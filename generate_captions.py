"""
Generate platform-specific posting copy from judged clip metadata/transcript.
Writes work/captions.json. No posting credentials or social APIs involved.
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

DEFAULT_HOST = "https://openrouter.ai/api/v1"

PROMPT = """You write truthful high-retention social copy for short-form video.
For every supplied clip generate:
- tiktok: concise natural caption; optional genuine discussion question; 3-5 relevant hashtags
- instagram: concise caption with slightly more context; 3-6 relevant hashtags
- youtube_title: searchable, curiosity-driven, <=70 characters, no fake claim
- youtube_description: 1-2 sentences plus 2-4 relevant hashtags
- x: punchy standalone post, <=240 characters before hashtags
Rules: no fake quotes, no fabricated facts, no spammy ALL CAPS, no irrelevant #fyp pile,
no pretending controversy exists when it doesn't. Match the clip's actual transcript.
Return JSON {"captions":[{"id":1,"tiktok":"...","instagram":"...",
"youtube_title":"...","youtube_description":"...","x":"..."}]}.
"""

def clip_text(words, start, end):
    return " ".join(w["word"] for w in words
                    if float(w["end"]) > start and float(w["start"]) < end).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--model", default="openrouter/free")
    ap.add_argument("--host", default=DEFAULT_HOST)
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY","").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required for caption generation.")
    wd=Path(args.workdir)
    tr=json.loads((wd/"transcript.json").read_text(encoding="utf-8-sig"))
    segs=json.loads((wd/"segments.json").read_text(encoding="utf-8-sig"))
    clips=[]
    for i,s in enumerate(segs,1):
        clips.append({"id":i,"hook":s.get("hook"),"viral_type":s.get("viral_type"),
                      "score":s.get("overall"),"reason":s.get("judge_reason") or s.get("why"),
                      "transcript":clip_text(tr["words"],float(s["start"]),float(s["end"]))})
    payload={"model":args.model,"temperature":0.3,"response_format":{"type":"json_object"},
             "messages":[{"role":"system","content":PROMPT},
                         {"role":"user","content":json.dumps({"clips":clips},ensure_ascii=False)}]}
    req=urllib.request.Request(args.host.rstrip("/")+"/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {key}",
                 "HTTP-Referer":"https://github.com/mikailyousuf-sketch/chopify",
                 "X-Title":"Creator Rewards Clipper"})
    with urllib.request.urlopen(req,timeout=240) as resp:
        body=json.loads(resp.read().decode("utf-8"))
    parsed=json.loads(body["choices"][0]["message"]["content"])
    caps=parsed.get("captions",[]) if isinstance(parsed,dict) else []
    (wd/"captions.json").write_text(json.dumps(caps,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {wd/'captions.json'} ({len(caps)} clip caption set(s))",flush=True)

if __name__=="__main__":
    main()
