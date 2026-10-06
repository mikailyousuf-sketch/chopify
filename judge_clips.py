"""
Second-pass viral judge for Creator Rewards Clipper.

Reads transcript.json + segments.json, asks the configured OpenAI-compatible
cloud brain to compare candidates against each other, writes judged_segments.json
and replaces segments.json with the ranked winners. The original discovery output
is preserved as segments.discovery.json.
"""
import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

DEFAULT_HOST = "https://openrouter.ai/api/v1"

JUDGE_PROMPT = """You are the final commissioning editor for short-form viral video.
You are judging candidates against EACH OTHER, not being polite.

For each candidate score these 0-10:
hook: stops a cold viewer in first seconds
context: understandable with zero prior knowledge
curiosity: creates a reason to keep watching
payoff: reaction/reveal/result is worth the wait
emotion: laughter/shock/anger/awe/empathy/etc
comment: likelihood of genuine discussion/debate
rewatch: density/quotability/replay value
campaign: fit to the campaign brief

Rules:
- A shocking word alone is NOT controversy. Controversy needs an actual claim,
  disagreement, conflict, accusation, taboo opinion or position viewers can debate.
- Profanity alone is not viral.
- Suicide/death/sex/crime references do not earn points merely for being sensitive.
- A cliffhanger must contain genuine unresolved stakes or curiosity; vague 'something
  happens next' is not enough.
- Podcast clips need a memorable insight/story/confession/quote, not ordinary chatter.
- Challenges need understandable stakes plus an attempt/result/reaction.
- Penalize missing setup, missing payoff, generic conversation and duplicate angles.
- Use the full scale. Do NOT bunch everything at 8 or 9.
- 9.3+ exceptional/must-post; 8.5-9.2 excellent; 7.5-8.4 strong; 6.5-7.4 usable;
  below 6.5 weak.
- Never invent events or meaning not present in the supplied transcript.
- viral_type may be corrected to controversy, podcast, cliffhanger, chaos, story,
  emotional, challenge, or educational.

Return JSON object {"clips":[{"id":1,"score":8.7,"viral_type":"challenge",
"hook":"specific truthful hook","reason":"short reason","scores":{"hook":9.0,
"context":8.5,"curiosity":9.0,"payoff":8.8,"emotion":7.5,"comment":7.0,
"rewatch":8.2,"campaign":9.0}}]}.
Return every supplied id exactly once.
"""

def clip_text(words, start, end):
    return " ".join(
        w["word"] for w in words
        if float(w["end"]) > start and float(w["start"]) < end
    ).strip()

def call_cloud(host, model, key, payload, timeout=240):
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
    ap = argparse.ArgumentParser(description="Second-pass comparative viral judge")
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--model", default="openrouter/free")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--campaign", default="")
    ap.add_argument("--keep", type=int, default=45)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required for the comparative judge.")

    wd = Path(args.workdir)
    transcript = json.loads((wd / "transcript.json").read_text(encoding="utf-8-sig"))
    seg_path = wd / "segments.json"
    segments = json.loads(seg_path.read_text(encoding="utf-8-sig"))
    if not segments:
        raise SystemExit("segments.json is empty")

    candidates = []
    for i, seg in enumerate(segments, 1):
        candidates.append({
            "id": i,
            "start": seg["start"], "end": seg["end"],
            "discovery_score": seg.get("overall"),
            "discovery_type": seg.get("viral_type"),
            "discovery_hook": seg.get("hook"),
            "transcript": clip_text(
                transcript["words"], float(seg["start"]), float(seg["end"])
            ),
        })

    user = {
        "campaign": args.campaign or "(no extra campaign brief)",
        "candidates": candidates,
    }
    payload = {
        "model": args.model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    print(f"Viral judge: comparing {len(candidates)} candidates with {args.model}...", flush=True)
    raw = call_cloud(args.host, args.model, key, payload)
    parsed = json.loads(raw)
    judged = parsed.get("clips", []) if isinstance(parsed, dict) else []
    by_id = {int(x["id"]): x for x in judged if isinstance(x, dict) and "id" in x}

    merged = []
    for i, seg in enumerate(segments, 1):
        verdict = by_id.get(i)
        if not verdict:
            continue
        out = dict(seg)
        out["discovery_score"] = seg.get("overall")
        out["overall"] = round(float(verdict.get("score", seg.get("overall", 0))), 1)
        out["viral_type"] = str(verdict.get("viral_type") or seg.get("viral_type") or "story")
        out["hook"] = str(verdict.get("hook") or seg.get("hook") or "clip")
        out["judge_reason"] = str(verdict.get("reason") or "")[:240]
        out["judge_scores"] = verdict.get("scores") or {}
        out["selection_source"] = str(seg.get("selection_source") or "discovery") + "+judge"
        merged.append(out)

    if not merged:
        raise SystemExit("Judge returned no valid candidate IDs; discovery file left unchanged.")

    merged.sort(key=lambda x: (-float(x.get("overall", 0)), float(x["start"])))
    merged = merged[:max(1, args.keep)]

    backup = wd / "segments.discovery.json"
    if not backup.exists():
        shutil.copy2(seg_path, backup)
    judged_path = wd / "judged_segments.json"
    judged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    seg_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  #  score  type          hook", flush=True)
    for i, seg in enumerate(merged, 1):
        print(f"  {i:>2}  {seg['overall']:>4.1f}   {seg['viral_type']:<12}  {seg['hook']}", flush=True)
    print(f"Wrote {judged_path} and promoted winners to {seg_path}", flush=True)
    print(f"Original discovery preserved at {backup}", flush=True)

if __name__ == "__main__":
    main()
