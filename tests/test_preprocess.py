"""Юнит-тесты препроцессинга (по замечанию ревью №3: единственный модуль
с логикой ветвления без покрытия). rembg стабится — сеть не нужна.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from img2stl import preprocess


def make_image(mode: str, size=(8, 8), opaque: bool = False) -> Image.Image:
    """Синтетическая картинка: половина прозрачная; opaque=True — полностью непрозрачная."""
    if mode in ("RGBA", "LA"):
        arr = np.zeros((size[1], size[0], 4), np.uint8)
        arr[..., :3] = 200
        arr[..., 3] = 255 if opaque else 128
        if not opaque:
            arr[: size[1] // 2, :, 3] = 0  # верхняя половина прозрачная
        return Image.fromarray(arr, mode)
    if mode == "P":
        img = Image.new("P", size)
        img.putpalette([200, 0, 0] + [0] * (255 * 3))
        img.info["transparency"] = 1  # честная палитровая прозрачность
        img.putdata([1] * (size[0] * size[1]))  # пиксели прозрачного индекса
        return img
    return Image.new(mode, size, 200)


@pytest.fixture
def stub_remover(monkeypatch):
    """Стаб BackgroundRemover: помечает результат, считает вызовы."""
    calls = []

    def fake_remover(img):
        calls.append(img.mode)
        out = Image.new("RGBA", img.size, (10, 20, 30, 255))
        return out

    monkeypatch.setattr(preprocess, "_remover", fake_remover)
    return calls


# ------------------------------------------------------------------ has_alpha

def test_has_alpha_true_for_real_transparency():
    assert preprocess.has_alpha(make_image("RGBA")) is True


def test_has_alpha_false_for_fully_opaque_rgba():
    """JPEG, сконвертированный Gradio в RGBA: альфа есть, но непрозрачна → rembg нужен."""
    assert preprocess.has_alpha(make_image("RGBA", opaque=True)) is False


def test_has_alpha_false_for_plain_rgb():
    assert preprocess.has_alpha(make_image("RGB")) is False


def test_has_alpha_true_for_palette_with_transparency():
    assert preprocess.has_alpha(make_image("P")) is True


# ------------------------------------------------------------- on_white_background

def test_on_white_background_composites_alpha():
    rgba = make_image("RGBA")
    out = preprocess.on_white_background(rgba)
    assert out.mode == "RGB"
    arr = np.asarray(out)
    assert (arr[: rgba.size[1] // 2] == 255).all()  # прозрачная половина → белый


# ---------------------------------------------------------------- prepare_image

def test_prepare_image_uses_alpha_without_remover(stub_remover):
    rgba, white = preprocess.prepare_image(make_image("RGBA"))
    assert stub_remover == []  # rembg не вызывается: уже есть альфа
    assert rgba.mode == "RGBA"
    assert white.mode == "RGB"


def test_prepare_image_runs_remover_for_opaque(stub_remover):
    rgba, white = preprocess.prepare_image(make_image("RGB"))
    assert stub_remover == ["RGB"]  # rembg вызвался ровно один раз
    assert rgba.mode == "RGBA"
    assert white.mode == "RGB"


def test_prepare_image_runs_remover_for_opaque_rgba(stub_remover):
    """Полностью непрозрачный RGBA (случай JPEG из UI) → тоже снимаем фон."""
    rgba, _ = preprocess.prepare_image(make_image("RGBA", opaque=True))
    assert stub_remover == ["RGB"]
    assert rgba.mode == "RGBA"
