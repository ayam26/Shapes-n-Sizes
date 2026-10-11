"""Unfiltered frame sampling from the user's dashcam drives (spec §4.4).

Every drive in the collection period is used; frames are taken every N seconds with N
chosen so the total reaches the minimum; nothing is selected by hand. Only unreadable
frames (decode failure, no frame at that time, or all black) are dropped, each logged.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import av
from PIL import Image

BLACK_MEAN = 2.0  # 0-255 scale


def choose_interval(total_seconds: float, min_frames: int = 2000) -> int:
    return max(1, math.floor(total_seconds / min_frames))


def sample_times(duration: float, every: float) -> list[float]:
    n = math.ceil(duration / every - 1e-9)
    return [i * every for i in range(n)]


def frame_name(drive: str, t: float) -> str:
    return f"{drive}_{round(t * 1000):08d}ms.jpg"


def video_duration(video: Path) -> float:
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        if s.duration is not None:
            return float(s.duration * s.time_base)
        return c.duration / av.time_base


def extract_frames(video: Path, drive: str, every: float, out_dir: Path
                   ) -> tuple[list[dict], list[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sample_times(video_duration(video), every)
    rows, drops, i = [], [], 0
    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                while i < len(targets) and targets[i] <= frame.time + 1e-6:
                    t = targets[i]
                    img = frame.to_ndarray(format="rgb24")
                    if img.mean() < BLACK_MEAN:
                        drops.append({"drive": drive, "t": t, "reason": "black"})
                    else:
                        path = out_dir / frame_name(drive, t)
                        Image.fromarray(img).save(path, quality=95)
                        rows.append({"name": path.name,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "drive": drive, "t": t})
                    i += 1
                if i == len(targets):
                    break
    except av.error.FFmpegError as e:
        drops += [{"drive": drive, "t": t, "reason": f"decode error: {e}"} for t in targets[i:]]
        i = len(targets)
    drops += [{"drive": drive, "t": t, "reason": "no frame"} for t in targets[i:]]
    return rows, drops


def sample_footage(drives_csv: Path, out_dir: Path, min_frames: int = 2000) -> dict:
    drives_csv, out_dir = Path(drives_csv), Path(out_dir)
    with open(drives_csv, newline="") as f:
        drives = list(csv.DictReader(f))
    for d in drives:
        d["path"] = (drives_csv.parent / d["file"]).resolve()
        if not d["path"].exists():
            raise FileNotFoundError(f"drive {d['drive_id']}: video not found: {d['path']}")
    total = sum(video_duration(d["path"]) for d in drives)
    every = choose_interval(total, min_frames)
    rows, drops = [], []
    for d in drives:
        r, x = extract_frames(d["path"], d["drive_id"], every, out_dir / "frames")
        rows += [{**row, "time_of_day": d["time_of_day"], "weather": d["weather"]} for row in r]
        drops += x
    with open(out_dir / "footage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "sha256", "drive", "t", "time_of_day",
                                          "weather"])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "footage_drops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drive", "t", "reason"])
        w.writeheader()
        w.writerows(drops)
    return {"drives": len(drives), "total_seconds": total, "every_seconds": every,
            "frames": len(rows), "dropped": len(drops)}
