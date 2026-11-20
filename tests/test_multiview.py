"""Юнит-тесты мультивью-режима.

Этап 1 плана: доменная логика видов (validate_views/prepare_views) — TDD.
Этап 3 добавит сюда же тесты загрузчика папки видов для smoke-CLI.
"""
import numpy as np
import pytest
from PIL import Image

from img2stl import preprocess
from img2stl.preprocess import prepare_views, validate_views


def make_image(kind: str = "transparent", size=(8, 8)) -> Image.Image:
    """kind: 'transparent' — RGBA с честной прозрачностью; 'opaque' — RGB."""
    if kind == "opaque":
        return Image.new("RGB", size, 200)
    arr = np.zeros((size[1], size[0], 4), np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 255
    arr[: size[1] // 2, :, 3] = 0  # верхняя половина прозрачная
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def stub_remover(monkeypatch):
    calls = []

    def fake_remover(img):
        calls.append(img.mode)
        return Image.new("RGBA", img.size, (10, 20, 30, 255))

    monkeypatch.setattr(preprocess, "_remover", fake_remover)
    return calls


# ------------------------------------------------------------- validate_views

def test_validate_views_empty_raises():
    with pytest.raises(ValueError):
        validate_views({})


def test_validate_views_unknown_key_raises():
    with pytest.raises(ValueError):
        validate_views({"front": make_image(), "top": make_image()})


def test_validate_views_single_front_ok():
    img = make_image()
    assert validate_views({"front": img}) == {"front": img}


def test_validate_views_all_four_canonical_order():
    views = {k: make_image() for k in ("right", "back", "front", "left")}
    assert list(validate_views(views)) == ["front", "back", "left", "right"]


def test_validate_views_drops_none_slots():
    """Слоты UI без фото приходят как None — отбрасываются молча."""
    views = validate_views({"front": make_image(), "back": None, "left": None})
    assert list(views) == ["front"]


# -------------------------------------------------------------- prepare_views

def test_prepare_views_removes_background_per_opaque_view(stub_remover):
    rgba_views, white_views = prepare_views(
        {"front": make_image("opaque"), "back": make_image("opaque")}
    )
    assert stub_remover == ["RGB", "RGB"]  # rembg ровно один раз на вид
    assert sorted(rgba_views) == ["back", "front"]
    assert sorted(white_views) == ["back", "front"]


def test_prepare_views_skips_remover_for_real_alpha(stub_remover):
    rgba_views, _ = prepare_views(
        {"front": make_image("transparent"), "back": make_image("opaque")}
    )
    assert stub_remover == ["RGB"]  # для вида с альфой rembg не вызывается
    assert sorted(rgba_views) == ["back", "front"]


def test_prepare_views_output_modes(stub_remover):
    rgba_views, white_views = prepare_views({"left": make_image("opaque")})
    assert rgba_views["left"].mode == "RGBA"
    assert white_views["left"].mode == "RGB"


# --------------------------------------------- загрузчик папки видов (smoke-CLI)

def test_load_views_from_dir_two_files(tmp_path):
    from img2stl.smoke import load_views_from_dir

    make_image("opaque").save(tmp_path / "front.png")
    make_image("opaque").save(tmp_path / "back.jpg")
    views = load_views_from_dir(tmp_path)
    assert sorted(views) == ["back", "front"]
    assert all(img.mode == "RGBA" for img in views.values())


def test_load_views_from_dir_jpeg_variant(tmp_path):
    from img2stl.smoke import load_views_from_dir

    make_image("opaque").save(tmp_path / "left.jpeg", format="JPEG")
    assert list(load_views_from_dir(tmp_path)) == ["left"]


def test_load_views_from_dir_empty_raises(tmp_path):
    from img2stl.smoke import load_views_from_dir

    with pytest.raises(ValueError):
        load_views_from_dir(tmp_path)


def test_load_views_from_dir_foreign_names_raise(tmp_path):
    from img2stl.smoke import load_views_from_dir

    make_image("opaque").save(tmp_path / "top.png")
    with pytest.raises(ValueError):
        load_views_from_dir(tmp_path)


def test_load_views_from_dir_missing_dir_raises(tmp_path):
    from img2stl.smoke import load_views_from_dir

    with pytest.raises(ValueError):
        load_views_from_dir(tmp_path / "nope")
