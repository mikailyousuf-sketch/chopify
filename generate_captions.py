"""
Generate platform-specific posting copy from judged clip metadata/transcript.
Robust against flaky free-router responses: batches clips, retries null/non-JSON
responses, and preserves successful batches.
"""
import argparse, json, os, re, time, urllib.request
from pathlib import Path

DEFAULT_HOST="https://openrouter.ai/api/v1"
PROMPT="""You write truthful high-retention social copy for short-form video.
For every supplied clip generate:
- tiktok: concise natural caption; optional genuine discussion question; 3-5 relevant hashtags
- instagram: concise caption with slightly more context; 3-6 relevant hashtags
- youtube_title: searchable, curiosity-driven, <=70 characters, no fake claim
- youtube_description: 1-2 sentences plus 2-4 relevant hashtags
- x: punchy standalone post, <=240 characters before hashtags
No fake quotes/facts, spammy ALL CAPS, irrelevant #fyp piles, or invented controversy.
Return JSON object {"captions":[{"id":1,"tiktok":"...","instagram":"...",
"youtube_title":"...","youtube_description":"...","x":"..."}]}."""

def clip_text(words,start,end):
    return " ".join(w["word"] for w in words
        if float(w["end"])>start and float(w["start"])<end).strip()

def request_json(host,model,key,user,retries=3):
    last=""
    for attempt in range(1,retries+1):
        payload={"model":model,"temperature":0.25,
                 "response_format":{"type":"json_object"},
                 "messages":[{"role":"system","content":PROMPT},
                             {"role":"user","content":json.dumps(user,ensure_ascii=False)}]}
        req=urllib.request.Request(host.rstrip("/")+"/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type":"application/json","Authorization":f"Bearer {key}",
                     "HTTP-Referer":"https://github.com/mikailyousuf-sketch/chopify",
                     "X-Title":"Creator Rewards Clipper"})
        try:
            with urllib.request.urlopen(req,timeout=240) as resp:
                body=json.loads(resp.read().decode("utf-8"))
            msg=((body.get("choices") or [{}])[0].get("message") or {})
            content=msg.get("content")
            if isinstance(content,str) and content.strip():
                try: return json.loads(content)
                except json.JSONDecodeError:
                    m=re.search(r"\{.*\}",content,re.S)
                    if m: return json.loads(m.group(0))
            last=f"empty/non-JSON response (finish_reason={(body.get('choices') or [{}])[0].get('finish_reason')})"
        except Exception as exc:
            last=str(exc)
        print(f"  caption brain attempt {attempt}/{retries} failed: {last}",flush=True)
        if attempt<retries: time.sleep(1.5*attempt)
    raise RuntimeError(last or "caption provider returned no usable content")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("workdir",nargs="?",default="work")
    ap.add_argument("--model",default="openrouter/free")
    ap.add_argument("--host",default=DEFAULT_HOST)
    ap.add_argument("--batch-size",type=int,default=4)
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY","").strip()
    if not key: raise SystemExit("OPENROUTER_API_KEY is required for caption generation.")
    wd=Path(args.workdir)
    tr=json.loads((wd/"transcript.json").read_text(encoding="utf-8-sig"))
    segs=json.loads((wd/"segments.json").read_text(encoding="utf-8-sig"))
    clips=[]
    for i,s in enumerate(segs,1):
        clips.append({"id":i,"hook":s.get("hook"),"viral_type":s.get("viral_type"),
          "score":s.get("overall"),"reason":s.get("judge_reason") or s.get("why"),
          "transcript":clip_text(tr["words"],float(s["start"]),float(s["end"]))})
    all_caps=[]
    size=max(1,args.batch_size)
    for n in range(0,len(clips),size):
        batch=clips[n:n+size]
        print(f"Caption brain: batch {n//size+1}/{(len(clips)+size-1)//size}...",flush=True)
        try:
            parsed=request_json(args.host,args.model,key,{"clips":batch})
            got=parsed.get("captions",[]) if isinstance(parsed,dict) else []
            if not isinstance(got,list): got=[]
            all_caps.extend(got)
        except Exception as exc:
            print(f"  batch skipped after retries: {exc}",flush=True)
    out=wd/"captions.json"
    out.write_text(json.dumps(all_caps,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {out} ({len(all_caps)}/{len(clips)} clip caption set(s))",flush=True)

if __name__=="__main__": main()
