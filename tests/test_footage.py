import csv

import av
import numpy as np
import pytest

from snz.cli import main
from snz.data.footage import choose_interval, extract_frames, frame_name, sample_times


def make_video(path, seconds=10, fps=5, black=(4.0, 5.0)):
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            t = i / fps
            value = 0 if black[0] <= t < black[1] else 60 + 15 * int(t)
            img = np.full((48, 64, 3), value, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_choose_interval():
    assert choose_interval(10_000, 2000) == 5
    assert choose_interval(9_000, 2000) == 4
    assert choose_interval(1_000, 2000) == 1


def test_sample_times():
    assert sample_times(10.0, 2) == [0, 2, 4, 6, 8]
    assert sample_times(9.99, 5) == [0, 5]


def test_extract_frames_drops_and_logs_black(tmp_path):
    video = make_video(tmp_path / "drive.mp4")
    rows, drops = extract_frames(video, "d1", 2, tmp_path / "frames")
    assert [r["t"] for r in rows] == [0, 2, 6, 8]
    assert drops == [{"drive": "d1", "t": 4, "reason": "black"}]
    assert (tmp_path / "frames" / frame_name("d1", 6)).exists()
    assert len(rows[0]["sha256"]) == 64


def test_cli_footage_sample(tmp_path):
    make_video(tmp_path / "a.mp4")
    make_video(tmp_path / "b.mp4", seconds=6, black=(99, 99))
    drives = tmp_path / "drives.csv"
    drives.write_text("drive_id,file,time_of_day,weather\n"
                      "a,a.mp4,daytime,clear\nb,b.mp4,night,rainy\n")
    out = tmp_path / "foot"
    assert main(["footage-sample", "--drives", str(drives), "--out", str(out),
                 "--min-frames", "4"]) == 0
    with open(out / "footage.csv") as f:
        rows = list(csv.DictReader(f))
    # 16 s total, 4 frames minimum -> every 4 s: a at 0,4(black),8; b at 0,4
    assert [(r["drive"], float(r["t"])) for r in rows] == [("a", 0), ("a", 8), ("b", 0), ("b", 4)]
    assert rows[2]["time_of_day"] == "night" and rows[2]["weather"] == "rainy"
    assert "black" in (out / "footage_drops.csv").read_text()
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        drives.write_text("drive_id,file,time_of_day,weather\nc,missing.mp4,daytime,clear\n")
        main(["footage-sample", "--drives", str(drives), "--out", str(tmp_path / "x")])
