import os

from snz.fsutil import atomic_replace


def test_atomic_replace_fsyncs_file_and_directory(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    tmp = tmp_path / "x.tmp"
    tmp.write_bytes(b"data")
    atomic_replace(tmp, tmp_path / "x")
    assert (tmp_path / "x").read_bytes() == b"data" and not tmp.exists()
    assert len(synced) == 2  # the file, then its directory
