import numpy as np
import pytest
from PIL import Image

from helpers import write_image
from snz.corrupt import CORRUPTIONS, build_t_corrupt, choose_subset, corrupt_image, corrupt_name


def test_all_fifteen_corruptions_run_and_are_deterministic():
    assert len(CORRUPTIONS) == 15
    rng = np.random.default_rng(0)
    img = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    for c in CORRUPTIONS:
        a = corrupt_image(img, c, 3, key="x.jpg")
        assert a.shape == img.shape and a.dtype == np.uint8, c
        assert (a == corrupt_image(img, c, 3, key="x.jpg")).all(), c


def test_choose_subset():
    names = [f"{i}.jpg" for i in range(1000)]
    a = choose_subset(names, 300, seed=0)
    assert len(a) == 300 == len(set(a)) and a == sorted(a)
    assert a == choose_subset(list(reversed(names)), 300, seed=0)
    with pytest.raises(ValueError):
        choose_subset(names[:10], 300, seed=0)


def test_build_t_corrupt_writes_and_resumes(tmp_path):
    for n in ("a.jpg", "b.jpg"):
        write_image(tmp_path / "img" / n, size=(128, 96))
    kw = dict(image_dir=tmp_path / "img", out_dir=tmp_path / "out",
              corruptions=["gaussian_noise", "fog"], severities=(1, 5), workers=1)
    rows = build_t_corrupt(["a.jpg", "b.jpg"], **kw)
    assert len(rows) == 8
    f = tmp_path / "out" / corrupt_name("a.jpg", "fog", 5)
    assert Image.open(f).size == (128, 96)
    mtime = f.stat().st_mtime_ns
    build_t_corrupt(["a.jpg", "b.jpg"], **kw)
    assert f.stat().st_mtime_ns == mtime


def test_cli_corrupt_writes_manifest_and_images(tmp_path):
    from snz.cli import main
    from snz.data.manifest import read_manifest, write_manifest

    for i in range(5):
        write_image(tmp_path / "val" / f"v{i}.jpg", size=(128, 96), color=(30 * i, 90, 120))
    write_manifest(tmp_path / "m" / "t-clean.csv", [(f"v{i}.jpg", "h") for i in range(5)])
    assert main(["corrupt", "--manifests", str(tmp_path / "m"), "--images", str(tmp_path / "val"),
                 "--out", str(tmp_path / "tc"), "--k", "3", "--severities", "1", "5",
                 "--corruptions", "fog", "snow", "--workers", "1"]) == 0
    names = [n for n, _ in read_manifest(tmp_path / "m" / "t-corrupt.csv")]
    assert len(names) == 3
    assert len(list((tmp_path / "tc").glob("*.jpg"))) == 3 * 2 * 2
