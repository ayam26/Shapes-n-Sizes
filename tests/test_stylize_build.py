import csv

import pytest
from PIL import Image

from helpers import write_image
from snz.data.manifest import write_manifest
from snz.stylize.build import (
    StylizeConfig,
    build_stylized,
    run_stylize,
    style_for,
    stylized_name,
)

STYLES = [f"s{i}.jpg" for i in range(10)]


class CountingStylizer:
    """Returns the style image resized to the content size, and counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, content, style):
        self.calls += 1
        return style.resize(content.size)


def test_style_for_is_order_and_subset_independent():
    a = {(n, k): style_for(n, k, STYLES, seed=0) for n in ("x.jpg", "y.jpg") for k in (1, 2)}
    assert style_for("y.jpg", 2, STYLES, seed=0) == a[("y.jpg", 2)]
    assert len({style_for(f"{i}.jpg", 1, STYLES, 0) for i in range(100)}) > 5
    assert any(style_for(f"{i}.jpg", 1, STYLES, 0) != style_for(f"{i}.jpg", 1, STYLES, 1)
               for i in range(20))


def test_stylized_name():
    assert stylized_name("abc-123.jpg", 2) == "abc-123_s2.jpg"


def _setup(tmp_path):
    for n in ("a.jpg", "b.jpg"):
        write_image(tmp_path / "img" / n, size=(40, 30))
    for i, s in enumerate(STYLES):
        write_image(tmp_path / "styles" / s, size=(20, 20), color=(i * 20, 0, 0))


def test_build_stylized_writes_resumes_and_keeps_size(tmp_path):
    _setup(tmp_path)
    st = CountingStylizer()
    kw = dict(image_dir=tmp_path / "img", style_root=tmp_path / "styles", styles=STYLES,
              out_dir=tmp_path / "out", copies=(1, 2), seed=0)
    rows = build_stylized(["a.jpg", "b.jpg"], stylizer=st, **kw)
    assert st.calls == 4 and len(rows) == 4
    assert Image.open(tmp_path / "out" / "a_s1.jpg").size == (40, 30)
    assert not list((tmp_path / "out").glob("*.tmp"))
    again = build_stylized(["b.jpg", "a.jpg"], stylizer=st, **kw)
    assert st.calls == 4  # everything existed; nothing recomputed
    key = lambda r: (r["name"], r["copy"])  # noqa: E731
    assert sorted(again, key=key) == sorted(rows, key=key)


def test_build_stylized_rejects_size_change(tmp_path):
    _setup(tmp_path)
    with pytest.raises(ValueError, match="size"):
        build_stylized(["a.jpg"], tmp_path / "img", tmp_path / "styles", STYLES,
                       tmp_path / "out", (1,), 0, lambda c, s: s)


def test_run_stylize_uses_disjoint_pools(tmp_path):
    _setup(tmp_path)
    m = tmp_path / "manifests"
    write_manifest(m / "train.csv", [("a.jpg", "h")])
    write_manifest(m / "t-clean.csv", [("b.jpg", "h")])
    cfg = StylizeConfig(seed=0, test_fraction=0.2, train_copies=2, alpha=1.0, style_size=16)
    tr = run_stylize("train", m, tmp_path / "img", tmp_path / "styles", tmp_path / "o1", cfg,
                     CountingStylizer())
    te = run_stylize("t-texture", m, tmp_path / "img", tmp_path / "styles", tmp_path / "o2",
                     cfg, CountingStylizer())
    with open(m / "styles_test.csv") as f:
        test_styles = {r["name"] for r in csv.DictReader(f)}
    assert {r["copy"] for r in tr} == {1, 2} and {r["copy"] for r in te} == {1}
    assert all(r["style"] not in test_styles for r in tr)
    assert all(r["style"] in test_styles for r in te)
    assert (m / "stylized_train.csv").exists() and (m / "stylized_t-texture.csv").exists()
