"""
Chopify V1 local dashboard.
Run: python dashboard.py
Open: http://127.0.0.1:8765
"""
import json, shutil, subprocess, sys, threading, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

import render_clips

ROOT=Path(__file__).resolve().parent
WEB=ROOT/"dashboard"
WORK=ROOT/"work"
UPLOADS=WORK/"uploads"
PREVIEWS=WORK/"dashboard_previews"
PY=sys.executable
STATE={"running":False,"stage":"Ready","log":[],"error":None}
LOCK=threading.Lock()

def log(s):
    with LOCK:
        STATE["log"].append(str(s)); STATE["log"]=STATE["log"][-300:]

def set_state(**kw):
    with LOCK: STATE.update(**kw)

def run(cmd):
    log("> "+" ".join(map(str,cmd)))
    p=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:
        log(line.rstrip())
    if p.wait()!=0:
        raise RuntimeError(f"Command failed ({p.returncode})")

def read_json(path, default):
    try:return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:return default

def pipeline(data):
    try:
        set_state(running=True,error=None,stage="Transcribing")
        source=data["source"].strip()
        run([PY,ROOT/"download_and_transcribe.py",source,"--workdir",WORK,
             "--model",data.get("whisper","small")])
        set_state(stage="Finding viral moments")
        score=[PY,ROOT/"score_clips.py",WORK,"--cloud-model",data.get("model","openrouter/free"),
               "--llm","qwen2.5:3b","--max-clips",str(data.get("max_clips",45)),
               "--min-len","25","--max-len","45"]
        if data.get("campaign"): score += ["--campaign",data["campaign"]]
        if data.get("original"): score += ["--original-mode"]
        run(score)
        set_state(stage="Judging candidates")
        judge=[PY,ROOT/"judge_clips.py",WORK,"--model",data.get("model","openrouter/free"),
               "--keep",str(data.get("max_clips",45))]
        if data.get("campaign"): judge += ["--campaign",data["campaign"]]
        run(judge)
        set_state(stage="Writing platform captions")
        run([PY,ROOT/"generate_captions.py",WORK,"--model",data.get("model","openrouter/free")])
        set_state(stage="Ready to review")
    except Exception as e:
        set_state(error=str(e),stage="Failed");log("ERROR: "+str(e))
    finally:set_state(running=False)

def preview_name(seg):
    return render_clips.sanitize(seg.get("hook","clip"))+f"-{int(float(seg['start'])*1000):08d}.preview.mp4"

def preview_clip(data):
    try:
        set_state(running=True,error=None,stage="Building smart preview")
        idx=int(data["id"])-1
        segs=read_json(WORK/"segments.json",[])
        if idx<0 or idx>=len(segs):raise RuntimeError("Clip not found.")
        tr=read_json(WORK/"transcript.json",{})
        source=Path(tr.get("video",""))
        if not source.exists():
            cands=sorted(WORK.glob("source.*"));source=cands[0] if cands else source
        if not source.exists():raise RuntimeError("Source video is missing.")
        PREVIEWS.mkdir(parents=True,exist_ok=True)
        W,H=render_clips.ffprobe_dims(source)
        render_clips.render(segs[idx],tr["words"],source,W,H,WORK,"9:16",
                            PREVIEWS,data.get("style","default"),False,False,True)
        set_state(stage="Preview ready")
    except Exception as e:
        set_state(error=str(e),stage="Failed");log("ERROR: "+str(e))
    finally:set_state(running=False)

def render_selected(data):
    try:
        set_state(running=True,error=None,stage="Rendering selected clips")
        ids={int(x) for x in data.get("ids",[])}
        segs=read_json(WORK/"segments.json",[])
        chosen=[s for i,s in enumerate(segs,1) if i in ids]
        if not chosen:raise RuntimeError("Select at least one clip.")
        rw=WORK/"dashboard_render";rw.mkdir(exist_ok=True)
        shutil.copy2(WORK/"transcript.json",rw/"transcript.json")
        (rw/"segments.json").write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding="utf-8")
        cmd=[PY,ROOT/"render_clips.py",rw,"--aspect","9:16","--out",ROOT/"clips",
             "--style",data.get("style","default")]
        if data.get("tighten"):cmd.append("--tighten")
        if data.get("loudnorm"):cmd.append("--loudnorm")
        run(cmd);set_state(stage="Export complete")
    except Exception as e:
        set_state(error=str(e),stage="Failed");log("ERROR: "+str(e))
    finally:set_state(running=False)

def save_caption(data):
    clip_id=int(data["id"])
    caps=read_json(WORK/"captions.json",[])
    found=False
    for x in caps:
        if isinstance(x,dict) and int(x.get("id",-1))==clip_id:
            for k in ("tiktok","instagram","youtube_title","youtube_description","x"):
                if k in data:x[k]=str(data[k])
            found=True;break
    if not found:
        item={"id":clip_id}
        for k in ("tiktok","instagram","youtube_title","youtube_description","x"):
            item[k]=str(data.get(k,""))
        caps.append(item)
    (WORK/"captions.json").write_text(json.dumps(caps,ensure_ascii=False,indent=2),encoding="utf-8")

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        rel=urlparse(path).path.lstrip("/") or "index.html"
        if rel.startswith("clips/"):return str(ROOT/rel)
        if rel.startswith("previews/"):return str(PREVIEWS/rel.removeprefix("previews/"))
        return str(WEB/rel)
    def send_json(self,obj,status=200):
        b=json.dumps(obj,ensure_ascii=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=="/api/status":return self.send_json(STATE)
        if p=="/api/clips":
            segs=read_json(WORK/"segments.json",[])
            caps=read_json(WORK/"captions.json",[])
            cm={int(x.get("id",i+1)):x for i,x in enumerate(caps) if isinstance(x,dict)}
            rows=[]
            for i,s in enumerate(segs,1):
                pn=preview_name(s)
                rows.append(dict(s,id=i,caption=cm.get(i,{}),
                                 preview=(f"/previews/{pn}" if (PREVIEWS/pn).exists() else None)))
            return self.send_json(rows)
        return super().do_GET()
    def do_POST(self):
        if self.path=="/api/upload":
            if STATE["running"]:return self.send_json({"error":"A job is already running"},409)
            n=int(self.headers.get("Content-Length","0"))
            name=Path(unquote(self.headers.get("X-Filename","video.mp4"))).name
            safe="".join(c for c in name if c.isalnum() or c in " ._-").strip() or "video.mp4"
            UPLOADS.mkdir(parents=True,exist_ok=True)
            dest=UPLOADS/safe
            with dest.open("wb") as f:
                left=n
                while left:
                    chunk=self.rfile.read(min(left,1024*1024))
                    if not chunk:break
                    f.write(chunk);left-=len(chunk)
            return self.send_json({"ok":True,"path":str(dest.resolve())})
        n=int(self.headers.get("Content-Length","0"))
        try:data=json.loads(self.rfile.read(n) or b"{}")
        except Exception:return self.send_json({"error":"Invalid request"},400)
        if self.path=="/api/caption":
            save_caption(data);return self.send_json({"ok":True})
        if STATE["running"]:return self.send_json({"error":"A job is already running"},409)
        if self.path=="/api/analyze":
            threading.Thread(target=pipeline,args=(data,),daemon=True).start()
            return self.send_json({"ok":True})
        if self.path=="/api/preview":
            threading.Thread(target=preview_clip,args=(data,),daemon=True).start()
            return self.send_json({"ok":True})
        if self.path=="/api/render":
            threading.Thread(target=render_selected,args=(data,),daemon=True).start()
            return self.send_json({"ok":True})
        self.send_json({"error":"Not found"},404)
    def log_message(self,*args):pass

if __name__=="__main__":
    WEB.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    url="http://127.0.0.1:8765"
    print(f"CHOPIFY DASHBOARD -> {url}")
    print("Keep this terminal open while using Chopify.")
    try:webbrowser.open(url)
    except Exception:pass
    ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
