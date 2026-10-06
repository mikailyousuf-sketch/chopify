"""
Chopify V1 local dashboard. Zero extra web framework dependencies.
Run: python dashboard.py
Then open http://127.0.0.1:8765
"""
import json, os, shutil, subprocess, sys, threading, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
WEB=ROOT/"dashboard"
WORK=ROOT/"work"
PY=sys.executable
STATE={"running":False,"stage":"Ready","log":[],"error":None}
LOCK=threading.Lock()

def log(s):
    with LOCK:
        STATE["log"].append(str(s)); STATE["log"]=STATE["log"][-300:]

def run(cmd):
    log("> "+" ".join(map(str,cmd)))
    p=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        log(line.rstrip())
    if p.wait()!=0: raise RuntimeError(f"Command failed ({p.returncode})")

def pipeline(data):
    try:
        STATE.update(running=True,error=None,stage="Transcribing")
        source=data["source"].strip()
        run([PY,ROOT/"download_and_transcribe.py",source,"--workdir",WORK,"--model",data.get("whisper","small")])
        STATE["stage"]="Finding viral moments"
        score=[PY,ROOT/"score_clips.py",WORK,"--cloud-model",data.get("model","openrouter/free"),
               "--llm","qwen2.5:3b","--max-clips",str(data.get("max_clips",45)),
               "--min-len","25","--max-len","45"]
        if data.get("campaign"): score += ["--campaign",data["campaign"]]
        if data.get("original"): score += ["--original-mode"]
        run(score)
        STATE["stage"]="Judging candidates"
        judge=[PY,ROOT/"judge_clips.py",WORK,"--model",data.get("model","openrouter/free"),
               "--keep",str(data.get("max_clips",45))]
        if data.get("campaign"): judge += ["--campaign",data["campaign"]]
        run(judge)
        STATE["stage"]="Writing captions"
        run([PY,ROOT/"generate_captions.py",WORK,"--model",data.get("model","openrouter/free")])
        STATE["stage"]="Ready to review"
    except Exception as e:
        STATE["error"]=str(e); STATE["stage"]="Failed"; log("ERROR: "+str(e))
    finally: STATE["running"]=False

def render_selected(data):
    try:
        STATE.update(running=True,error=None,stage="Rendering selected clips")
        ids={int(x) for x in data.get("ids",[])}
        segs=json.loads((WORK/"segments.json").read_text(encoding="utf-8-sig"))
        chosen=[s for i,s in enumerate(segs,1) if i in ids]
        if not chosen: raise RuntimeError("Select at least one clip.")
        rw=WORK/"dashboard_render"; rw.mkdir(exist_ok=True)
        shutil.copy2(WORK/"transcript.json",rw/"transcript.json")
        (rw/"segments.json").write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding="utf-8")
        cmd=[PY,ROOT/"render_clips.py",rw,"--aspect","9:16","--out",ROOT/"clips","--style",data.get("style","default")]
        if data.get("tighten"): cmd.append("--tighten")
        if data.get("loudnorm"): cmd.append("--loudnorm")
        run(cmd); STATE["stage"]="Export complete"
    except Exception as e:
        STATE["error"]=str(e); STATE["stage"]="Failed"; log("ERROR: "+str(e))
    finally: STATE["running"]=False

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        rel=urlparse(path).path.lstrip("/") or "index.html"
        if rel.startswith("clips/"): return str(ROOT/rel)
        return str(WEB/rel)
    def send_json(self,obj,status=200):
        b=json.dumps(obj,ensure_ascii=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=="/api/status": return self.send_json(STATE)
        if p=="/api/clips":
            try:
                segs=json.loads((WORK/"segments.json").read_text(encoding="utf-8-sig"))
                caps=[]
                cp=WORK/"captions.json"
                if cp.exists(): caps=json.loads(cp.read_text(encoding="utf-8-sig"))
                cm={int(x.get("id",i+1)):x for i,x in enumerate(caps) if isinstance(x,dict)}
                return self.send_json([dict(s,id=i,caption=cm.get(i,{})) for i,s in enumerate(segs,1)])
            except Exception:return self.send_json([])
        return super().do_GET()
    def do_POST(self):
        n=int(self.headers.get("Content-Length","0")); data=json.loads(self.rfile.read(n) or b"{}")
        if STATE["running"]: return self.send_json({"error":"A job is already running"},409)
        if self.path=="/api/analyze":
            threading.Thread(target=pipeline,args=(data,),daemon=True).start()
            return self.send_json({"ok":True})
        if self.path=="/api/render":
            threading.Thread(target=render_selected,args=(data,),daemon=True).start()
            return self.send_json({"ok":True})
        self.send_json({"error":"Not found"},404)
    def log_message(self,*args): pass

if __name__=="__main__":
    WEB.mkdir(exist_ok=True)
    url="http://127.0.0.1:8765"
    print(f"CHOPIFY DASHBOARD -> {url}")
    try:webbrowser.open(url)
    except Exception:pass
    ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
