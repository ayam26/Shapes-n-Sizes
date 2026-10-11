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


def make_offset_video(path, seconds=10, fps=5, start=600):
    with av.open(str(path), "w", format="mpegts",
                 options={"output_ts_offset": str(start)}) as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            img = np.full((48, 64, 3), 40 + 20 * (i // fps), np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_timestamps_not_starting_at_zero(tmp_path):
    video = make_offset_video(tmp_path / "d.ts")
    rows, drops = extract_frames(video, "d", 2, tmp_path / "f")
    assert [r["t"] for r in rows] == [0, 2, 4, 6, 8] and drops == []
    assert len({r["sha256"] for r in rows}) == 5  # five different frames, not one repeated


def test_short_footage_warns(tmp_path):
    from snz.data.footage import sample_footage

    make_video(tmp_path / "a.mp4", seconds=4, black=(99, 99))
    (tmp_path / "drives.csv").write_text("drive_id,file,time_of_day,weather\na,a.mp4,day,clear\n")
    r = sample_footage(tmp_path / "drives.csv", tmp_path / "out", min_frames=2000)
    assert r["frames"] == 4 and "below" in r["warning"]


def test_label_sample_is_random_and_reproducible(tmp_path):
    from snz.data.footage import label_sample
    from snz.data.manifest import write_manifest

    write_manifest(tmp_path / "footage.csv", [(f"f{i}.jpg", "h") for i in range(500)])
    a = label_sample(tmp_path / "footage.csv", 100, seed=0)
    assert len(a) == 100 == len(set(a)) and a == label_sample(tmp_path / "footage.csv", 100, 0)
    assert a != [f"f{i}.jpg" for i in range(100)]


def make_raw_stream(path, seconds=10, fps=5):
    """Raw MPEG-4 elementary stream (no container, no reliable timestamps), like comma2k19's
    raw HEVC files."""
    with av.open(str(path), "w", format="m4v") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            t = i / fps
            value = 0 if 4.0 <= t < 5.0 else 60 + 15 * int(t)
            img = np.full((48, 64, 3), value, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_raw_stream_with_declared_fps(tmp_path):
    from snz.data.footage import video_duration

    video = make_raw_stream(tmp_path / "video.m4v")
    assert video_duration(video, fps=5) == pytest.approx(10.0)
    rows, drops = extract_frames(video, "seg", 2, tmp_path / "f", fps=5)
    assert [r["t"] for r in rows] == [0, 2, 6, 8]
    assert drops == [{"drive": "seg", "t": 4, "reason": "black"}]
    assert len({r["sha256"] for r in rows}) == 4


def test_drives_csv_fps_column(tmp_path):
    from snz.data.footage import sample_footage

    make_raw_stream(tmp_path / "a.m4v")
    (tmp_path / "drives.csv").write_text(
        "drive_id,file,time_of_day,weather,fps\na,a.m4v,daytime,unknown,5\n")
    r = sample_footage(tmp_path / "drives.csv", tmp_path / "out", min_frames=5)
    assert r["every_seconds"] == 2 and r["frames"] == 4


def test_comma2k19_drives_csv(tmp_path):
    from snz.data.footage import comma2k19_drives

    root = tmp_path / "Chunk_1"
    for route, segs in (("b0c9d2329ad1606b|2018-07-27--06-03-57", (3, 10)),
                        ("99c94dc769b5d96e|2018-05-01--08-13-53", (1,))):
        for seg in segs:
            (root / route / str(seg)).mkdir(parents=True)
            (root / route / str(seg) / "video.hevc").write_bytes(b"x")
    (root / "stray").mkdir()
    rows = comma2k19_drives(root, tmp_path / "drives.csv")
    assert [r["drive_id"] for r in rows] == [
        "99c94dc7_2018-05-01--08-13-53_1", "b0c9d232_2018-07-27--06-03-57_3",
        "b0c9d232_2018-07-27--06-03-57_10"]
    assert rows[0]["file"] == "Chunk_1/99c94dc769b5d96e|2018-05-01--08-13-53/1/video.hevc"
    assert rows[0]["fps"] == 20 and rows[0]["time_of_day"] == "unknown"
    assert (tmp_path / "drives.csv").read_text().startswith("drive_id,file,time_of_day,weather,fps")
