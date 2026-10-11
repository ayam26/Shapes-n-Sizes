"""Unfiltered frame sampling from the user's dashcam drives (spec §4.4).

Every drive in the collection period is used; frames are taken every N seconds with N
chosen so the total reaches the minimum; nothing is selected by hand. Only unreadable
frames (decode failure, no frame at that time, or all black) are dropped, each logged.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
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


def video_duration(video: Path, fps: float | None = None) -> float:
    """Seconds of video. With `fps` (raw elementary streams such as comma2k19's video.hevc,
    which have no container timing), duration = frames / fps."""
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        if fps:
            return sum(1 for p in c.demux(s) if p.size) / fps
        if s.duration is not None:
            return float(s.duration * s.time_base)
        if c.duration is not None:
            return c.duration / av.time_base
    raise ValueError(f"{video}: duration unknown (damaged or still-recording file?)")


def extract_frames(video: Path, drive: str, every: float, out_dir: Path,
                   fps: float | None = None) -> tuple[list[dict], list[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sample_times(video_duration(video, fps), every)
    rows, drops, i = [], [], 0
    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            # sample times are measured from the first frame, not from pts 0 (MPEG-TS and
            # stream-copied clips often start at a large timestamp)
            t0 = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
            for k, frame in enumerate(c.decode(stream)):
                # declared fps: time from the frame index (raw streams have no usable pts)
                ftime = k / fps + t0 if fps else frame.time
                if ftime is None:
                    continue
                while i < len(targets) and targets[i] + t0 <= ftime + 1e-6:
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
        d["fps"] = float(d["fps"]) if d.get("fps") else None
        d["path"] = (drives_csv.parent / d["file"]).resolve()
        if not d["path"].exists():
            raise FileNotFoundError(f"drive {d['drive_id']}: video not found: {d['path']}")
    total = sum(video_duration(d["path"], d["fps"]) for d in drives)
    every = choose_interval(total, min_frames)
    rows, drops = [], []
    for d in drives:
        r, x = extract_frames(d["path"], d["drive_id"], every, out_dir / "frames", d["fps"])
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
    summary = {"drives": len(drives), "total_seconds": total, "every_seconds": every,
               "frames": len(rows), "dropped": len(drops)}
    if len(rows) < min_frames:
        summary["warning"] = (f"{len(rows)} frames is below the preregistered minimum of "
                              f"{min_frames}; record more drives in the collection period")
    return summary


def label_sample(footage_manifest: Path, k: int = 100, seed: int = 0) -> list[str]:
    """The frames to hand-label for recall: a seeded random sample, never hand-picked."""
    with open(footage_manifest, newline="") as f:
        names = sorted(r["name"] for r in csv.DictReader(f))
    return sorted(random.Random(seed).sample(names, min(k, len(names))))


COMMA2K19_FPS = 20  # comma2k19 road camera (README of the dataset)


def comma2k19_drives(root: Path, out_csv: Path) -> list[dict]:
    """drives.csv for comma2k19: every segment of every route under `root` is one drive.

    Paths in the CSV are relative to the CSV's folder; segments are ordered by route then
    segment number; nothing is selected.
    """
    root, out_csv = Path(root), Path(out_csv)
    rows = []
    for video in root.glob("*/*/video.hevc"):
        route, seg = video.parent.parent.name, video.parent.name
        dongle, _, stamp = route.partition("|")
        rows.append({"drive_id": f"{dongle[:8]}_{stamp}_{seg}",
                     "file": video.relative_to(out_csv.parent).as_posix(),
                     "time_of_day": "unknown", "weather": "unknown", "fps": COMMA2K19_FPS,
                     "_key": (route, int(seg))})
    rows.sort(key=lambda r: r["_key"])
    for r in rows:
        del r["_key"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drive_id", "file", "time_of_day", "weather", "fps"])
        w.writeheader()
        w.writerows(rows)
    return rows
