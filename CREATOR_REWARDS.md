# Creator Rewards Clipper — V1

Personal, local-first long-form → short-form pipeline for TikTok, Instagram Reels, YouTube Shorts and X.

## Goal
Fastest path to usable campaign clips with zero recurring processing cost. No SaaS, auth, database or cloud renderer.

## Quick start (Windows)
1. Install Python 3.11+ and FFmpeg (full build with libx264 + AAC).
2. `pip install -r requirements.txt`
3. Optional but recommended: install Ollama and pull a local model such as `qwen2.5:3b`.

### Local video → vertical clips
```
python chopify.py "C:\\Videos\\podcast.mp4" --platform vertical --tighten --loudnorm --preview
```

### Campaign-aware selection
```
python chopify.py "C:\\Videos\\podcast.mp4" --platform vertical --llm qwen2.5:3b --campaign "20-60 second funny or controversial standalone moments. Strong first 2 seconds. Avoid sponsor reads and inside jokes requiring earlier context." --min-len 20 --max-len 60 --max-clips 45 --preview
```

Review the numbered candidates, then render selected winners with `--clips 1,3,5` and without `--preview`.

### Export vertical + X landscape
```
python chopify.py "C:\\Videos\\podcast.mp4" --platform all --out exports --llm qwen2.5:3b --campaign "..." --tighten --loudnorm
```

## Platform outputs
- vertical: 1080×1920 master for TikTok, Instagram Reels and YouTube Shorts.
- x: 1920×1080 landscape master.
- all: both masters.

## Cost
Core processing is local: faster-whisper + Ollama + OpenCV + FFmpeg. No paid API is required.


## Laptop preset
The V1 defaults are tuned for an 8 GB Windows laptop: faster-whisper `small`, CPU INT8, sequential rendering and Ollama `qwen2.5:3b`. A matching transcript is cached so caption/style/original-mode reruns do not repeat Whisper.

## Original Mode
Add `--original-mode` to create transformed editorial clips. V1 adds an editorial opening hook while preserving the source quote, alongside smart reframing, captions and tightening. Leave the flag out for standard clipping.

### First recommended test
```powershell
python chopify.py "C:\Videos\test.mp4" --platform vertical --llm qwen2.5:3b --campaign "Find strong standalone moments with a clear hook and payoff. Avoid sponsor reads and moments that need earlier context." --original-mode --max-clips 45 --preview
```
