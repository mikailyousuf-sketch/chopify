"""
Find story-remix opportunities across already-discovered moments.

This is intentionally a planning pass: it does not render yet. It finds truthful
2-4 timestamp combinations that become a stronger short when assembled together
and writes work/remix_candidates.json for review/render integration.
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

DEFAULT_HOST = "https://openrouter.ai/api/v1"

REMIX_PROMPT = """You are a short-form story editor. Find cases where 2-4 DIFFERENT
timestamp ranges from the same source can be combined into a stronger truthful short.

Good remix patterns:
- earlier prediction/claim -> later attempt/result/reaction
- question/setup -> later answer/reveal
- accusation/opinion -> later response/consequence
- challenge announced -> attempt -> payoff
- promise/goal -> later outcome
- repeated joke/theme -> strongest final payoff

Rules:
- Every piece must refer to the same event/person/topic or a clearly connected story.
- Never splice unrelated statements to imply something the speaker did not mean.
- Preserve chronology by default. A later moment may open as a flash-forward hook only
  if the final sequence remains truthful and understandable.
- Do not create a remix if one continuous candidate is already stronger.
- INTRO + RANDOM LATER ACTION is not a story. The later beat must resolve, answer,
  contradict, escalate or pay off something specifically established by the earlier beat.
- Reject combinations whose only connection is that they happen in the same video,
  involve the same guest, or are both workouts/challenges.
- hook must be a NEW 3-10 word editorial hook, never a transcript dump or paragraph.
- Total assembled duration should usually be 25-50 seconds.
- Use 2-4 pieces only. Prefer fewer cuts.
- Each proposal needs a specific reason why the combination is stronger.
- Return only genuinely useful proposals; zero is acceptable.

Return JSON object {"remixes":[{"hook":"...", "viral_type":"story",
"score":8.8,"reason":"...","segments":[{"start":1.2,"end":7.5,"role":"hook"},
{"start":80.1,"end":105.0,"role":"payoff"}]}]}.
"""

def text_for(words, start, end):
    return " ".join(
        w["word"] for w in words
        if float(w["end"]) > start and float(w["start"]) < end
    ).strip()

def call(host, model, key, payload, timeout=240):
    req = urllib.request.Request(
        host.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/mikailyousuf-sketch/chopify",
            "X-Title": "Creator Rewards Clipper",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser(description="Find multi-timestamp story remixes")
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--model", default="openrouter/free")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--campaign", default="")
    ap.add_argument("--max-remixes", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required for remix discovery.")

    wd = Path(args.workdir)
    transcript = json.loads((wd / "transcript.json").read_text(encoding="utf-8-sig"))
    # Remix discovery needs smaller beats than standalone 25-45s clips. Build a
    # compact timeline from transcript words (~10s windows) so an early promise
    # can be related to a later payoff even when neither beat was a full clip.
    words = transcript["words"]
    duration = float(transcript.get("duration") or (words[-1]["end"] if words else 0))
    moments = []
    window, step = 12.0, 10.0
    t, idx = 0.0, 1
    while t < duration:
        end = min(duration, t + window)
        txt = text_for(words, t, end)
        if len(txt.split()) >= 6:
            moments.append({
                "id": idx, "start": round(t, 2), "end": round(end, 2),
                "transcript": txt,
            })
            idx += 1
        t += step

    # Keep judged/discovered hooks as editorial landmarks too.
    source_path = wd / "segments.json"
    if source_path.exists():
        for seg in json.loads(source_path.read_text(encoding="utf-8-sig")):
            moments.append({
                "id": idx, "start": seg["start"], "end": seg["end"],
                "type": seg.get("viral_type"), "hook": seg.get("hook"),
                "score": seg.get("overall"),
                "transcript": text_for(words, float(seg["start"]), float(seg["end"])),
                "landmark": True,
            })
            idx += 1

    payload = {
        "model": args.model,
        "temperature": 0.15,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": REMIX_PROMPT},
            {"role": "user", "content": json.dumps({
                "campaign": args.campaign or "(none)",
                "max_remixes": args.max_remixes,
                "moments": moments,
            }, ensure_ascii=False)},
        ],
    }
    print(f"Remix finder: relating {len(moments)} timeline beats with {args.model}...", flush=True)
    parsed = json.loads(call(args.host, args.model, key, payload))
    remixes = parsed.get("remixes", []) if isinstance(parsed, dict) else []

    clean = []
    for remix in remixes[:args.max_remixes]:
        pieces = remix.get("segments") if isinstance(remix, dict) else None
        if not isinstance(pieces, list) or not 2 <= len(pieces) <= 4:
            continue
        try:
            total = sum(float(x["end"]) - float(x["start"]) for x in pieces)
        except (KeyError, TypeError, ValueError):
            continue
        if not 15 <= total <= 60:
            continue
        hook = str(remix.get("hook") or "").strip()
        # Transcript-dump hooks are a strong signal that the free model ignored
        # the editorial schema; reject them rather than rendering garbage.
        if len(hook.split()) < 3 or len(hook.split()) > 12:
            continue
        out = dict(remix)
        out["hook"] = hook
        out["assembled_duration"] = round(total, 2)
        out["viral_type"] = "story_remix"
        clean.append(out)

    clean.sort(key=lambda x: -float(x.get("score", 0)))
    out_path = wd / "remix_candidates.json"
    out_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Found {len(clean)} valid remix candidate(s):", flush=True)
    for i, r in enumerate(clean, 1):
        spans = " + ".join(
            f"{float(x['start']):.1f}-{float(x['end']):.1f}" for x in r["segments"]
        )
        print(f"  {i:>2}  {float(r.get('score',0)):>4.1f}/10  {r.get('hook','')}  [{spans}]", flush=True)
    print(f"Wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
