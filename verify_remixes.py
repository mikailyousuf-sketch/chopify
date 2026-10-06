"""
Verify proposed story remixes against the exact source transcript passages.
Only semantically faithful, materially stronger remixes survive.
"""
import argparse,json,os,re,time,urllib.request
from pathlib import Path

HOST="https://openrouter.ai/api/v1"
PROMPT="""You are a skeptical senior video editor verifying a proposed multi-timestamp remix.
You receive ONLY the exact source transcript for each proposed piece.

Decide whether the proposed edit is truthful and genuinely stronger.
FAIL it if:
- the title/hook invents a redemption, conflict, transformation, result, relationship or claim
  that the supplied words do not establish;
- pieces are connected only because they share a person/location/general topic;
- the later piece does not specifically answer, resolve, escalate, contradict or pay off the earlier one;
- removing chronology changes the speaker's meaning;
- it is merely an intro followed by unrelated action;
- the combined edit is not materially better than the strongest continuous piece.

A valid remix should have a clear editorial relation such as setup->payoff,
prediction->result, question->answer, challenge->attempt/result, claim->response,
or repeated motif->payoff.

Return JSON {"verified":[{"id":1,"pass":true,"score":8.6,
"relation":"challenge->result","hook":"3-10 word truthful hook",
"reason":"specific evidence-based reason"}]}.
Use the full score scale. pass=true requires score >= 8.0 and a specific relation.
Return every id exactly once."""

def txt(words,a,b):
    return " ".join(w["word"] for w in words if float(w["end"])>a and float(w["start"])<b).strip()

def request(host,model,key,data,retries=3):
    last=""
    for n in range(1,retries+1):
        payload={"model":model,"temperature":0.05,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":PROMPT},
                      {"role":"user","content":json.dumps(data,ensure_ascii=False)}]}
        req=urllib.request.Request(host.rstrip("/")+"/chat/completions",
          data=json.dumps(payload).encode(),headers={"Content-Type":"application/json",
          "Authorization":f"Bearer {key}","HTTP-Referer":"https://github.com/mikailyousuf-sketch/chopify",
          "X-Title":"Creator Rewards Clipper"})
        try:
            with urllib.request.urlopen(req,timeout=240) as r: body=json.loads(r.read().decode())
            content=((body.get("choices") or [{}])[0].get("message") or {}).get("content")
            if content:
                try:return json.loads(content)
                except json.JSONDecodeError:
                    m=re.search(r"\{.*\}",content,re.S)
                    if m:return json.loads(m.group())
            last="empty/non-JSON response"
        except Exception as e:last=str(e)
        print(f"  verifier retry {n}/{retries}: {last}",flush=True);time.sleep(n)
    raise RuntimeError(last)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("workdir",nargs="?",default="work")
    ap.add_argument("--model",default="openrouter/free")
    ap.add_argument("--host",default=HOST)
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY","").strip()
    if not key:raise SystemExit("OPENROUTER_API_KEY required")
    wd=Path(args.workdir)
    tr=json.loads((wd/"transcript.json").read_text(encoding="utf-8-sig"))
    rem=json.loads((wd/"remix_candidates.json").read_text(encoding="utf-8-sig"))
    cases=[]
    for i,r in enumerate(rem,1):
        pieces=[]
        for p in r["segments"]:
            a,b=float(p["start"]),float(p["end"])
            pieces.append({"start":a,"end":b,"role":p.get("role"),"transcript":txt(tr["words"],a,b)})
        cases.append({"id":i,"proposed_hook":r.get("hook"),"proposed_reason":r.get("reason"),
                      "proposed_score":r.get("score"),"pieces":pieces})
    if not cases:
        print("No remix candidates to verify.");return
    print(f"Remix verifier: checking {len(cases)} candidate(s)...",flush=True)
    parsed=request(args.host,args.model,key,{"remixes":cases})
    verdicts=parsed.get("verified",[]) if isinstance(parsed,dict) else []
    byid={int(v["id"]):v for v in verdicts if isinstance(v,dict) and "id" in v}
    passed=[]
    for i,r in enumerate(rem,1):
        v=byid.get(i,{})
        ok=bool(v.get("pass")) and float(v.get("score",0))>=8.0
        print(f"  {i:>2}  {'PASS' if ok else 'FAIL':<4} {float(v.get('score',0)):>4.1f}/10  "
              f"{v.get('relation','')}  {v.get('reason','')}",flush=True)
        if ok:
            x=dict(r);x["score"]=round(float(v["score"]),1)
            x["hook"]=str(v.get("hook") or r.get("hook"))
            x["verification"]={"relation":v.get("relation"),"reason":v.get("reason")}
            passed.append(x)
    out=wd/"verified_remixes.json"
    out.write_text(json.dumps(passed,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {out} ({len(passed)}/{len(rem)} passed)",flush=True)

if __name__=="__main__":main()
