"""
Stage 2 of the chopify pipeline - the built-in "brain".

Reads work/transcript.json, picks the most viral self-contained moments and
writes work/segments.json (a bare JSON array render_clips.py consumes):

    [{"start": 134.2, "end": 187.6, "hook": "short title", "overall": 8.4}]

Two scoring modes, both free:

  1. heuristic (default) - pure-Python scoring over the transcript: words per
     second (energy), lexical cue detection (hook / shock / humour /
     controversy / insight / emotion) and arc completion. Deterministic,
     zero dependency.
  2. --llm MODEL - asks a LOCAL Ollama server (http://localhost:11434) in JSON
     mode to pick the moments. Falls back to the heuristic automatically if
     Ollama is unreachable or returns junk.

No paid APIs, no API keys. Usage:
    python score_clips.py work
    python score_clips.py work --llm qwen2.5:7b
    python score_clips.py work --min-score 6.0 --max-clips 12
"""
import re
import json
import argparse
import os
import urllib.request
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
OPENROUTER_HOST = "https://openrouter.ai/api/v1"

FILLER_TOKENS = {"um", "uh", "uhh", "umm", "erm", "er", "hmm", "mmm", "mm", "ah", "eh"}
FILLER_SEQS = {("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")}

HOOK_CUES = ("the secret", "nobody tells you", "here's the thing", "here is the thing",
             "what most people", "the truth is", "here's why", "here is why", "imagine",
             "the problem is", "the biggest mistake", "let me tell you", "you need to",
             "here's how", "here is how", "the trick", "the key", "listen", "story")
SHOCK_CUES = ("insane", "crazy", "shocking", "unbelievable", "never", "nobody", "everyone",
              "banned", "illegal", "mistake", "wrong", "destroyed", "worst", "best",
              "huge", "million", "billion", "dead", "killed")
HUMOUR_CUES = ("haha", "lol", "laugh", "joke", "funny", "hilarious", "kidding", "ridiculous")
CONTROVERSY_CUES = ("i disagree", "unpopular opinion", "hot take", "controversial",
                    "stop doing", "don't do", "do not do", "overrated", "underrated",
                    "everyone is wrong", "myth", "lie", "scam")
INSIGHT_CUES = ("because", "the reason", "how it works", "step one", "first", "second",
                "third", "the way", "what happens", "if you", "the difference",
                "framework", "system", "lesson", "learned", "the point")
EMOTION_CUES = ("love", "hate", "afraid", "scared", "amazing", "terrible", "proud",
                "embarrassed", "angry", "excited", "grateful", "hurts", "cried",
                "dream", "hope", "believe")


# ---------------------------------------------------------------- transcript

def load_transcript(workdir):
    path = Path(workdir) / "transcript.json"
    if not path.exists():
        raise SystemExit(f"No transcript at {path} - run download_and_transcribe.py first")
    # utf-8-sig transparently handles BOM'd files from Windows editors too
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"transcript.json is not valid JSON: {e}")
    if not data.get("words"):
        raise SystemExit("transcript.json has no words - was the video silent?")
    return data


ARC_RE = re.compile(r"[.!?\u2026][\"')\]]?$")


def is_arc_complete(text):
    """True when a sentence ends with terminal punctuation (a complete thought)."""
    return bool(ARC_RE.search((text or "").strip()))


def sentenceize(words, gap=1.2):
    """Split a word list into sentences on terminal punctuation or >gap-second pauses."""
    sentences, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"]) > gap:
            sentences.append(cur)
            cur = []
        cur.append(w)
        if is_arc_complete(w["word"]):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    out = []
    for s in sentences:
        out.append({
            "start": s[0]["start"],
            "end": s[-1]["end"],
            "text": " ".join(w["word"] for w in s).strip(),
            "words": s,
        })
    return out


# ------------------------------------------------------------- heuristic core

# criterion -> weight (hook/energy/arc dominate; matches the README's 8 criteria)
WEIGHTS = {"hook": 0.20, "shock": 0.10, "humour": 0.05, "controversy": 0.10,
           "insight": 0.15, "emotion": 0.10, "energy": 0.15, "arc": 0.15}

def _cues(low, group):
    return min(1.0, sum(c in low for c in group) / 2.0)


def score_sentence(sent):
    text = sent["text"]
    low = " " + text.lower() + " "
    dur = max(0.1, sent["end"] - sent["start"])
    wps = len(sent["words"]) / dur
    arc = 1.0 if is_arc_complete(text) else 0.55
    scores = {
        "hook": _cues(low, HOOK_CUES),
        "shock": _cues(low, SHOCK_CUES),
        "humour": _cues(low, HUMOUR_CUES),
        "controversy": _cues(low, CONTROVERSY_CUES),
        "insight": _sm_cues(low),
        "emotion": _cues(low, EMOTION_CUES),
        "energy": min(1.0, wps / 3.0),
        "arc": arc,
    }
    overall = sum(WEIGHTS[k] * v for k, v in scores.items()) * 10.0
    return scores, round(overall, 2)


def _sm_cues(low):
    return _cues(low, INSIGHT_CUES)


def make_hook(text, max_words=7):
    words = [_norm_token(w) for w in text.split()]
    words = [w for w in words if w and w not in FILLER_TOKENS]
    return " ".join(words[:max_words]) or "clip"


def _norm_token(w):
    return re.sub(r"[^\w']", "", w.lower())


def _grow_window(sentences, seed, taken, min_len, max_len):
    """Grow a window from `seed` to >= min_len seconds, then extend or trim it so
    it always ends on a complete sentence (the #1 complaint about AI clippers is
    clips that start or end mid-thought). Returns (seed, end_idx) or None."""
    n = len(sentences)

    def span(a, b):
        return sentences[b]["end"] - sentences[a]["start"]

    # Do not stop as soon as the minimum is reached. A technically valid
    # 20-second cut often feels chopped-off in conversation. Aim for a useful
    # editorial body (~34s by default), then finish on a complete thought.
    target_len = min(max_len, max(min_len, 34))
    j = seed
    while span(seed, j) < target_len and j + 1 < n and not taken[j + 1]:
        j += 1
    while span(seed, j) > max_len and j > seed:
        j -= 1
    if span(seed, j) < min_len * 0.6:
        return None
    # complete-thought rule: keep growing (within max_len) until the last
    # sentence ends with terminal punctuation
    while not is_arc_complete(sentences[j]["text"]) and j + 1 < n \
            and not taken[j + 1] and span(seed, j + 1) <= max_len:
        j += 1
    # still incomplete -> back off to the last complete sentence in the window
    while j > seed and not is_arc_complete(sentences[j]["text"]):
        j -= 1
    if not is_arc_complete(sentences[j]["text"]):
        return None
    if span(seed, j) < min_len * 0.6:
        return None
    return seed, j


def _make_segment(sentences, a, b):
    """Score the sentence window sentences[a..b] and build a segment dict."""
    win = sentences[a:b + 1]
    words = [w for s in win for w in s["words"]]
    text = " ".join(w["word"] for w in words)
    scores, _ = score_sentence({"start": win[0]["start"], "end": win[-1]["end"],
                                "text": text, "words": words})
    return {
        "start": round(win[0]["start"], 2),
        "end": round(win[-1]["end"], 2),
        "hook": make_hook(win[0]["text"]),
        "overall": round(sum(WEIGHTS[k] * v for k, v in scores.items()) * 10.0, 1),
    }


def _segment_text(sentences, a, b):
    return " ".join(s["text"] for s in sentences[a:b + 1])


def _fingerprint(text):
    """Only collapse effectively identical transcript cuts; similar variants survive."""
    return " ".join(_norm_token(w) for w in text.split() if _norm_token(w))


def build_candidates(sentences, min_len, max_len, max_clips, allow_overlap=True):
    """Seed windows at the highest-scoring sentences, grow to min_len, dedup, sort.
    Windows always end on a complete sentence (complete-thought rule)."""
    scored = {i: score_sentence(s)[1] for i, s in enumerate(sentences)}
    order = sorted(range(len(sentences)), key=lambda i: scored[i], reverse=True)
    taken = [False] * len(sentences)
    accepted = []
    seen_windows = set()
    seen_text = set()

    for seed in order:
        if (not allow_overlap and taken[seed]) or len(accepted) >= max_clips:
            continue
        growth_taken = taken if not allow_overlap else [False] * len(sentences)
        w = _grow_window(sentences, seed, growth_taken, min_len, max_len)
        if w is None:
            continue
        a, b = w
        key = (round(sentences[a]["start"], 2), round(sentences[b]["end"], 2))
        fp = _fingerprint(_segment_text(sentences, a, b))
        if key in seen_windows or fp in seen_text:
            continue
        seen_windows.add(key)
        seen_text.add(fp)
        accepted.append(_make_segment(sentences, a, b))
        if not allow_overlap:
            for k in range(a, b + 1):
                taken[k] = True
    accepted.sort(key=lambda s: s["start"])
    return accepted


# ----------------------------------------------------------------- search/pick

def search_hits(sentences, query):
    """Indexes of sentences containing ALL whitespace-separated query tokens
    (case-insensitive). Empty-token queries match nothing."""
    tokens = [_norm_token(t) for t in (query or "").split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    hits = []
    for i, s in enumerate(sentences):
        low = " " + s["text"].lower() + " "
        if all(t in low for t in tokens):
            hits.append(i)
    return hits


def search_candidates(sentences, query, min_len, max_len, max_clips):
    """Build scored clips around every sentence matching `query` (keyword search
    over the transcript - the feature users call a dealbreaker when missing)."""
    taken = [False] * len(sentences)
    accepted = []
    for seed in search_hits(sentences, query):
        if taken[seed] or len(accepted) >= max_clips:
            continue
        w = _grow_window(sentences, seed, taken, min_len, max_len)
        if w is None:
            continue
        a, b = w
        accepted.append(_make_segment(sentences, a, b))
        for k in range(a, b + 1):
            taken[k] = True
    accepted.sort(key=lambda s: s["start"])
    return accepted


def pick_segments(segments, spec):
    """Keep only the given 1-based indexes ('1,3,5') of the start-sorted list -
    the review workflow: look at the printed table, render just the winners."""
    if not spec:
        return segments
    want = set()
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        if not tok.isdigit():
            raise SystemExit(f"--clips: '{tok}' is not a number (use e.g. --clips 1,3,5)")
        want.add(int(tok))
    n = len(segments)
    bad = sorted(i for i in want if i < 1 or i > n)
    if bad:
        raise SystemExit(
            f"--clips: index(es) {bad} out of range 1..{n} "
            f"(see the numbered table above)")
    return [s for i, s in enumerate(segments, 1) if i in want]


def print_table(segments):
    """Numbered candidate table - pairs with --clips for the review workflow."""
    print("\n  #    start ->      end  score  type          hook", flush=True)
    for i, s in enumerate(segments, 1):
        print(f"  {i:>2}  {s['start']:8.2f} -> {s['end']:8.2f}  {s['overall']:4.1f}/10"
              f"  {s.get('viral_type', 'heuristic'):<12}  {s['hook']}", flush=True)
    print("  render a subset with --clips 1,3,5", flush=True)

LLM_SYSTEM = (
    "You are the lead viral-content editor for TikTok, Instagram Reels, YouTube Shorts and X. "
    "Your job is to identify moments with VIEW POTENTIAL, not merely clean transcript excerpts. "
    "Assume the viewer has zero prior context and is deciding whether to swipe every second.\n"
    "VIRAL CONTENT LENSES - actively search every transcript chunk for these:\n"
    "1. CONTROVERSY/DEBATE: polarising opinions, disagreement, taboo/unpopular takes, accusations, conflict, strong claims, moments that make viewers choose a side. Do not manufacture controversy that is not actually present.\n"
    "2. PODCAST/QUOTE: a sharp insight, story, confession, lesson, surprising fact, memorable quote or answer that stands alone and feels worth sharing.\n"
    "3. CLIFFHANGER/CURIOSITY: an unresolved question, looming consequence, reveal, challenge or 'what happens next?' moment. Preserve a satisfying local beat, but the ending may intentionally create curiosity for the source/next part when truthful. Never fabricate missing events.\n"
    "4. CHAOS/REACTION: arguments, fails, shocks, sudden changes, embarrassment, laughter, disbelief, physical reactions or high-energy exchanges.\n"
    "5. STORY/PAYOFF: setup -> escalation -> twist/reveal/result. Include enough setup for the payoff to land.\n"
    "6. RELATABLE/EMOTIONAL: fear, ambition, pain, pride, vulnerability, awkwardness, friendship, rivalry or a situation viewers strongly recognise.\n"
    "7. CHALLENGE/STAKES: a goal, timer, bet, risk, competition, punishment, attempt, win/loss or clear consequence.\n"
    "8. EDUCATIONAL/HOW-TO: useful explanation, framework, mistake, lesson or surprising mechanism with a concrete takeaway.\n"
    "EDITORIAL RULES:\n"
    "- Every selected clip must make sense to someone who has seen NOTHING before it. Start before the context needed to understand the hook.\n"
    "- Prefer an immediate spoken/visual hook in the first 1-3 seconds. If the best hook occurs later, start at the shortest earlier setup that makes it land.\n"
    "- Preserve the question before a great answer, the challenge before a reaction, the accusation before a defence, and the setup before a punchline.\n"
    "- End after the payoff/reaction unless the clip is deliberately classified cliffhanger, where the unresolved curiosity must be genuine and compelling.\n"
    "- Reject greetings, introductions, sponsor reads, navigation, repetitive instructions, dead air, generic chatter and scene transitions.\n"
    "- Never treat weak words like 'listen', 'yeah', 'okay', 'bro' as the hook unless the actual following event is exceptional.\n"
    "- Clip length must be {min_len}-{max_len}s. Prefer 28-40s when context earns it; never pad a weak moment.\n"
    "- Seek VARIETY. Do not fill the batch with ten versions of the same type of moment.\n"
    "SCORING - score VIEW POTENTIAL, not grammar:\n"
    "- 9-10: must-post; powerful hook + clear stakes/curiosity + memorable payoff/reaction.\n"
    "- 8: strong post with obvious audience appeal.\n"
    "- 7: useful secondary post.\n"
    "- 5-6: average/filler. Below 5: weak.\n"
    "- Rank the strongest available moments even when the source is imperfect; do not return zero by default.\n"
    "HOOK TEXT RULES:\n"
    "- hook must describe the actual reason to watch, not simply copy the first transcript words.\n"
    "- Keep it specific, truthful and punchy; no fabricated claims or fake quotes.\n"
    'Respond with JSON only: [{{"start": <sec>, "end": <sec>, "hook": "<3-10 word hook>", '
    '"overall": <0-10>, "viral_type": "<controversy|podcast|cliffhanger|chaos|story|emotional|challenge|educational>", '
    '"why": "<one short reason this could earn views>"}}]'
)

# ------------------------------------------------------- cloud/free provider

def openrouter_select(data, model, api_key, host, min_score, max_clips,
                      min_len, max_len, campaign="", timeout=240):
    """Use OpenRouter's OpenAI-compatible API, then reuse the same viral prompt.

    Returns None on provider/rate-limit/JSON failure so the caller can fall back
    to local Ollama or heuristics without breaking the pipeline.
    """
    sentences = sentenceize(data["words"])
    if not sentences:
        return None
    chunks = [sentences[i:i + 240] for i in range(0, len(sentences), 240)] or [[]]
    raw = []
    print(f"  Cloud viral brain: {len(chunks)} transcript chunk(s), model={model}", flush=True)

    try:
        for batch_no, batch in enumerate(chunks, 1):
            per_chunk_target = max(3, (max_clips + len(chunks) - 1) // len(chunks) + 2)
            print(f"  Cloud viral brain: analysing chunk {batch_no}/{len(chunks)} "
                  f"({len(batch)} sentences, target {per_chunk_target})...", flush=True)
            lines = "\n".join(
                f"{s['start']:.1f}-{s['end']:.1f}: {s['text']}" for s in batch
            )
            sys_msg = LLM_SYSTEM.format(
                min_len=min_len, max_len=max_len,
                max_clips=max_clips, min_score=min_score
            )
            sys_msg += (
                f"\n\nFor THIS transcript chunk, return the best {per_chunk_target} "
                "usable candidates if possible. Rank the strongest available moments."
            )
            if campaign:
                sys_msg += (
                    "\n\nCAMPAIGN BRIEF:\n" + campaign +
                    "\nTreat campaign relevance as a major selection criterion."
                )
            payload = {
                "model": model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": sys_msg +
                     '\nReturn an object with one key named "clips" containing the array.'},
                    {"role": "user", "content":
                     f"Video duration: {data['duration']:.0f}s\n\nTranscript:\n{lines}"}
                ],
            }
            req = urllib.request.Request(
                host.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/mikailyousuf-sketch/chopify",
                    "X-Title": "Creator Rewards Clipper",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            clips = parsed.get("clips", []) if isinstance(parsed, dict) else parsed
            if not isinstance(clips, list):
                clips = []
            print(f"  Cloud viral brain: chunk {batch_no}/{len(chunks)} returned "
                  f"{len(clips)} candidate(s).", flush=True)
            raw.extend(clips)
    except Exception as exc:  # noqa: BLE001
        print(f"  Cloud viral brain unavailable ({exc})", flush=True)
        return None

    return _normalise_ai_clips(
        raw, data, sentences, max_clips, min_len, max_len,
        source_name="openrouter"
    )


def _normalise_ai_clips(raw, data, sentences, max_clips, min_len, max_len,
                        source_name="ai"):
    """Validate provider output and preserve viral metadata."""
    valid = []
    allowed_types = {"controversy", "podcast", "cliffhanger", "chaos",
                     "story", "emotional", "challenge", "educational"}
    for it in raw:
        try:
            start, end = float(it["start"]), float(it["end"])
            overall = float(it.get("overall", 0))
            hook = str(it.get("hook") or "clip")
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0.0, min(start, data["duration"] - min_len))
        end = max(start + min_len, min(end, data["duration"]))
        if end - start > max_len + 15:
            end = start + max_len
        viral_type = str(it.get("viral_type") or "story").strip().lower()
        if viral_type not in allowed_types:
            viral_type = "story"
        valid.append({
            "start": round(start, 2), "end": round(end, 2),
            "hook": make_hook(hook, 10), "overall": round(overall, 1),
            "viral_type": viral_type,
            "why": str(it.get("why") or "").strip()[:180],
            "selection_source": source_name,
        })

    starts = [s["start"] for s in sentences]
    ends = [s["end"] for s in sentences]
    for seg in valid:
        si = _nearest(starts, seg["start"])
        ei = _nearest(ends, seg["end"])
        if 2.0 < ends[ei] - starts[si] < max_len + 15:
            seg["start"], seg["end"] = round(starts[si], 2), round(ends[ei], 2)

    valid.sort(key=lambda s: (-s["overall"], s["start"]))
    dedup, seen = [], set()
    for seg in valid:
        key = (round(seg["start"], 2), round(seg["end"], 2))
        if key not in seen:
            seen.add(key)
            dedup.append(seg)
        if len(dedup) >= max_clips:
            break
    return dedup or None


# ------------------------------------------------------------------- ollama

def ollama_select(data, model, host, min_score, max_clips, min_len, max_len, campaign="", timeout=1800):
    """Ask local Ollama for clips. Returns list of dicts, or None on any failure."""
    sentences = sentenceize(data["words"])
    if not sentences:
        return None

    def call(batch, batch_no, total_batches):
        per_chunk_target = max(3, (max_clips + total_batches - 1) // total_batches + 2)
        print(f"  AI editor: analysing chunk {batch_no}/{total_batches} ({len(batch)} sentences, target {per_chunk_target})...", flush=True)
        lines = "\n".join(f"{s['start']:.1f}-{s['end']:.1f}: {s['text']}" for s in batch)
        sys_msg = LLM_SYSTEM.format(min_len=min_len, max_len=max_len,
                                    max_clips=max_clips, min_score=min_score)
        sys_msg += (f"\n\nFor THIS transcript chunk, return the best {per_chunk_target} usable candidates if possible. "
                    "Do not return zero merely because the material is imperfect; rank the strongest available moments.")
        if campaign:
            sys_msg += ("\n\nCAMPAIGN BRIEF:\n" + campaign +
                        "\nTreat campaign relevance as a major selection criterion. "
                        "Reject moments that violate the brief.")
        payload = {
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content":
                          f"Video duration: {data['duration']:.0f}s\n\nTranscript:\n{lines}"}],
        }
        req = urllib.request.Request(
            host.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = resp.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", content, re.S)
            parsed = json.loads(m.group(0)) if m else []
        if isinstance(parsed, dict):
            for key in ("clips", "segments", "candidates", "results"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed] if "start" in parsed and "end" in parsed else []
        clips = parsed if isinstance(parsed, list) else []
        print(f"  AI editor: chunk {batch_no}/{total_batches} returned {len(clips)} candidate(s).", flush=True)
        return clips

    try:
        chunks = [sentences[i:i + 240] for i in range(0, len(sentences), 240)] or [[]]
        raw = []
        print(f"  AI editor: {len(chunks)} transcript chunk(s), model={model}", flush=True)
        for batch_no, batch in enumerate(chunks, 1):
            raw.extend(call(batch, batch_no, len(chunks)) or [])
    except Exception as e:                                   # noqa: BLE001
        print(f"  Ollama unavailable ({e}) -> heuristic scoring", flush=True)
        return None

    valid = []
    for it in raw:
        try:
            start, end = float(it["start"]), float(it["end"])
            overall = float(it.get("overall", 0))
            hook = str(it.get("hook") or "clip")
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0.0, min(start, data["duration"] - min_len))
        end = max(start + min_len, min(end, data["duration"]))
        if end - start > max_len + 15:
            end = start + max_len
        viral_type = str(it.get("viral_type") or "story").strip().lower()
        allowed_types = {"controversy", "podcast", "cliffhanger", "chaos",
                         "story", "emotional", "challenge", "educational"}
        if viral_type not in allowed_types:
            viral_type = "story"
        why = str(it.get("why") or "").strip()[:180]
        valid.append({"start": round(start, 2), "end": round(end, 2),
                      "hook": make_hook(hook, 10), "overall": round(overall, 1),
                      "viral_type": viral_type, "why": why})

    # snap to sentence boundaries (nearest within 3s) so clips never cut mid-word
    starts = [s["start"] for s in sentences]
    ends = [s["end"] for s in sentences]
    for seg in valid:
        si = _nearest(starts, seg["start"])
        ei = _nearest(ends, seg["end"])
        if 2.0 < ends[ei] - starts[si] < max_len + 15:
            seg["start"], seg["end"] = round(starts[si], 2), round(ends[ei], 2)

    valid.sort(key=lambda s: (-s["overall"], s["start"]))
    dedup, seen = [], set()
    for seg in valid:
        key = (round(seg["start"], 2), round(seg["end"], 2))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(seg)
    dedup = dedup[:max_clips]
    if len(dedup) < max_clips:
        need = max_clips - len(dedup)
        print(f"  AI editor produced {len(dedup)}/{max_clips}; "
              f"filling {need} slot(s) with complete-thought candidates.", flush=True)
        pool = build_candidates(sentences, min_len, max_len, max(max_clips * 4, 40))
        used = {(round(s["start"], 2), round(s["end"], 2)) for s in dedup}
        # Prefer heuristic candidates that do not substantially duplicate an AI cut.
        for seg in sorted(pool, key=lambda s: (-s["overall"], s["start"])):
            key = (round(seg["start"], 2), round(seg["end"], 2))
            if key in used:
                continue
            overlap = any(
                max(0.0, min(seg["end"], x["end"]) - max(seg["start"], x["start"]))
                / max(1.0, min(seg["end"] - seg["start"], x["end"] - x["start"])) > 0.82
                for x in dedup
            )
            if overlap:
                continue
            seg["selection_source"] = "heuristic_fill"
            dedup.append(seg)
            used.add(key)
            if len(dedup) >= max_clips:
                break
    for seg in dedup:
        seg.setdefault("selection_source", "ollama")
    dedup = dedup[:max_clips]
    dedup.sort(key=lambda s: s["start"])
    return dedup if dedup else None


def _nearest(values, t):
    lo, hi = 0, len(values) - 1
    best = 0
    best_d = float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        d = abs(values[mid] - t)
        if d < best_d:
            best_d, best = d, mid
        if values[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return best

# ----------------------------------------------------------------- run / cli

def run(workdir, llm=None, host=OLLAMA_HOST, min_score=None, max_clips=45,
        min_len=20, max_len=60, search=None, pick=None, campaign="", original_mode=False):
    """Score the transcript in workdir and write workdir/segments.json.

    Returns (segments, mode) where mode is 'ollama' or 'heuristic'.
    search='keywords' overrides scoring with keyword-search clip finding;
    pick='1,3,5' keeps only the numbered candidates (review workflow).
    """
    workdir = Path(workdir)
    data = load_transcript(workdir)
    sentences = sentenceize(data["words"])
    print(f"Transcript: {len(sentences)} sentences over {data['duration']:.0f}s", flush=True)

    mode = "heuristic"
    segments = None
    if search:
        segments = search_candidates(sentences, search, min_len, max_len, max_clips)
        if not segments:
            raise SystemExit(
                f"--search '{search}': no transcript sentence matched. "
                "Try fewer or different keywords.")
        print(f"Keyword search '{search}': {len(segments)} match(es)", flush=True)
    elif llm:
        ms = min_score if min_score is not None else 8.0
        segments = ollama_select(data, llm, host, ms, max_clips, min_len, max_len, campaign=campaign)
        if segments is not None:
            mode = "ollama"
    if segments is None:
        ms = min_score if min_score is not None else 6.5
        pool = build_candidates(sentences, min_len, max_len, max(max_clips * 3, max_clips))
        strong = [s for s in pool if s["overall"] >= ms]
        # Aim for the requested volume. If a long source has fewer clips above the
        # quality floor, backfill with the next-best complete-thought candidates.
        ranked = sorted(pool, key=lambda s: (-s["overall"], s["start"]))
        segments = list(strong)
        used = {(s["start"], s["end"]) for s in segments}
        for seg in ranked:
            if len(segments) >= max_clips:
                break
            if (seg["start"], seg["end"]) not in used:
                segments.append(seg)
                used.add((seg["start"], seg["end"]))
        segments = sorted(segments[:max_clips], key=lambda s: s["start"])

    if not segments:                       # never ship zero clips silently
        print("No clip reached the threshold - keeping the single best candidate.",
              flush=True)
        segments = build_candidates(sentences, min_len, max_len, 1)

    if original_mode:
        for seg in segments:
            seg["original_mode"] = True
            seg["editorial_angle"] = "context_hook_takeaway"
    segments = pick_segments(segments, pick)
    if not segments:
        if pick:
            raise SystemExit("Nothing to render: --clips removed every candidate.")
        raise SystemExit("No usable clips found - the transcript may be too short "
                         f"for --min-len {min_len}s.")

    print_table(segments)
    out = workdir / "segments.json"
    out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}  (mode={mode}, {len(segments)} clips)", flush=True)
    return segments, mode


def main():
    ap = argparse.ArgumentParser(description="Score a chopify transcript for virality.")
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--campaign", default="",
                    help="campaign brief/instructions used by local LLM selection")
    ap.add_argument("--original-mode", action="store_true",
                    help="mark candidates for substantive original editorial treatment")
    ap.add_argument("--llm", nargs="?", const="qwen2.5:7b", default=None, metavar="MODEL",
                    help="score with a local Ollama model (e.g. qwen2.5:7b, llama3.1:8b)")
    ap.add_argument("--host", default=OLLAMA_HOST,
                    help="Ollama host (default http://localhost:11434)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="keep clips at/above this score (default 6.5 heuristic, 8.0 llm)")
    ap.add_argument("--max-clips", type=int, default=45)
    ap.add_argument("--min-len", type=int, default=20, help="min clip seconds")
    ap.add_argument("--max-len", type=int, default=60, help="max clip seconds")
    ap.add_argument("--search", default=None, metavar="KEYWORDS",
                    help="skip virality scoring: build clips around transcript "
                         "sentences containing ALL keywords (e.g. --search \"pricing\")")
    ap.add_argument("--clips", default=None, metavar="1,3,5",
                    help="render only these numbered candidates from the table "
                         "(review workflow)")
    args = ap.parse_args()
    run(args.workdir, args.llm, args.host, args.min_score, args.max_clips,
        args.min_len, args.max_len, search=args.search, pick=args.clips,
        campaign=args.campaign, original_mode=args.original_mode)
    print("SCORING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
