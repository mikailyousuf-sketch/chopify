"""
Render approved multi-timestamp remix candidates.

Each piece is rendered independently through render_clips.render(), so every
piece gets its own speaker-crop vs safe-full-frame decision. Finished pieces
are concatenated losslessly at the MP4 level where possible. This V1 burns
captions per piece; timestamps naturally restart at each editorial cut.
"""
import argparse, json, subprocess
from pathlib import Path
import render_clips as rc

def main():
    ap=argparse.ArgumentParser(description="Render multi-timestamp story remixes")
    ap.add_argument("workdir",nargs="?",default="work")
    ap.add_argument("--aspect",default="9:16",choices=list(rc.ASPECTS))
    ap.add_argument("--out",default="remix_clips")
    ap.add_argument("--style",default="default")
    ap.add_argument("--preview",action="store_true")
    ap.add_argument("--remixes",default=None,help="1-based list, e.g. 1,3")
    args=ap.parse_args()

    wd=Path(args.workdir).resolve()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    temp=out/"_remix_parts"; temp.mkdir(parents=True,exist_ok=True)
    tr=json.loads((wd/"transcript.json").read_text(encoding="utf-8-sig"))
    remixes=json.loads((wd/"remix_candidates.json").read_text(encoding="utf-8-sig"))
    if args.remixes:
        ids={int(x) for x in args.remixes.split(",") if x.strip().isdigit()}
        remixes=[r for i,r in enumerate(remixes,1) if i in ids]
    source=Path(tr["video"])
    if not source.exists():
        cands=sorted(wd.glob("source.*")); source=cands[0] if cands else source
    source=source.resolve()
    W,H=rc.ffprobe_dims(source)

    for ri,remix in enumerate(remixes,1):
        print(f"=== Remix {ri}/{len(remixes)}: {remix.get('hook','')} ===",flush=True)
        parts=[]
        for pi,piece in enumerate(remix["segments"],1):
            seg={"start":float(piece["start"]),"end":float(piece["end"]),
                 "hook":f"remix-{ri:02d}-part-{pi:02d}",
                 "overall":remix.get("score"),"viral_type":"story_remix"}
            part=rc.render(seg,tr["words"],source,W,H,wd,args.aspect,
                           out_dir=temp,style=args.style,preview=args.preview)
            parts.append(Path(part).resolve())
        concat=out/f"_concat_{ri:02d}.txt"
        concat.write_text("\n".join("file '"+str(p).replace("'","'\\''")+"'" for p in parts)+"\n",
                          encoding="utf-8")
        final=out/(rc.sanitize(remix.get("hook") or f"remix-{ri}")+
                   (".preview.mp4" if args.preview else ".mp4"))
        cmd=["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),
             "-c","copy","-movflags","+faststart",str(final)]
        try:
            subprocess.run(cmd,check=True)
        except subprocess.CalledProcessError:
            cmd=["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),
                 "-c:v","libx264","-preset","veryfast","-crf","20",
                 "-c:a","aac","-b:a","160k","-movflags","+faststart",str(final)]
            subprocess.run(cmd,check=True)
        meta={"hook":remix.get("hook"),"score":remix.get("score"),
              "viral_type":"story_remix","reason":remix.get("reason"),
              "segments":remix.get("segments"),"assembled_duration":remix.get("assembled_duration")}
        final.with_suffix(".meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"SAVED REMIX {final}",flush=True)
        try: concat.unlink()
        except OSError: pass

if __name__=="__main__": main()
