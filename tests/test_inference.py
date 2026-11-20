"""Юнит-тесты инференс-слоя: реестр пайплайнов и views_to_mesh.

Пайплайны стабятся через шов _load_pipeline — GPU/веса не нужны.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from img2stl import inference
from img2stl.inference import get_pipeline, image_to_mesh, views_to_mesh


def make_rgba():
    """RGBA с честной прозрачностью — идёт в пайплайн без rembg."""
    arr = np.zeros((4, 4, 4), np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 255
    arr[:2, :, 3] = 0
    return Image.fromarray(arr, "RGBA")


class StubPipeline:
    """Стаб пайплайна: пишет вызовы; fail_first первых вызовов падают OOM."""

    def __init__(self, fail_first: int = 0):
        self.calls = []
        self.fail_first = fail_first

    def __call__(self, *, image, num_inference_steps, guidance_scale,
                 octree_resolution, **kwargs):
        self.calls.append({
            "image": image,
            "steps": num_inference_steps,
            "guidance": guidance_scale,
            "octree": octree_resolution,
        })
        if len(self.calls) <= self.fail_first:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return [object()]  # output_type="trimesh" → список из одного меша


@pytest.fixture
def fresh_registry(monkeypatch):
    monkeypatch.setattr(inference, "_pipelines", {})
    return inference._pipelines


def use_stub(monkeypatch, stub: StubPipeline):
    monkeypatch.setattr(inference, "_load_pipeline", lambda name: stub)
    return stub


# -------------------------------------------------------------- get_pipeline

def test_get_pipeline_loads_once(monkeypatch, fresh_registry):
    loads = []

    def fake_load(name):
        loads.append(name)
        return StubPipeline()

    monkeypatch.setattr(inference, "_load_pipeline", fake_load)
    p1 = get_pipeline("mv")
    p2 = get_pipeline("mv")
    assert p1 is p2
    assert loads == ["mv"]


def test_get_pipeline_caches_both_pipelines(monkeypatch, fresh_registry):
    stubs = {"one": StubPipeline(), "mv": StubPipeline()}
    monkeypatch.setattr(inference, "_load_pipeline", lambda name: stubs[name])
    assert get_pipeline("one") is stubs["one"]
    assert get_pipeline("mv") is stubs["mv"]
    assert set(inference._pipelines) == {"one", "mv"}


def test_get_pipeline_rejects_unknown_name(monkeypatch, fresh_registry):
    with pytest.raises(ValueError):
        get_pipeline("stereo")


# -------------------------------------------------------------- views_to_mesh

def test_views_to_mesh_passes_prepared_dict(monkeypatch, fresh_registry):
    stub = use_stub(monkeypatch, StubPipeline())
    views = {"front": make_rgba(), "back": None, "left": make_rgba()}
    mesh = views_to_mesh(views, seed=1, num_inference_steps=30,
                         guidance_scale=5.0, octree_resolution=256)
    call = stub.calls[0]
    assert list(call["image"]) == ["front", "left"]  # dict видов, None отброшен
    assert all(img.mode == "RGBA" for img in call["image"].values())
    assert call["steps"] == 30 and call["octree"] == 256
    assert mesh is not None


def test_views_to_mesh_empty_raises_before_pipeline(monkeypatch, fresh_registry):
    stub = use_stub(monkeypatch, StubPipeline())
    with pytest.raises(ValueError):
        views_to_mesh({})
    assert stub.calls == []  # пайплайн не вызывался


def test_oom_retry_with_smaller_params(monkeypatch, fresh_registry):
    stub = use_stub(monkeypatch, StubPipeline(fail_first=1))
    views_to_mesh({"front": make_rgba()}, num_inference_steps=30, octree_resolution=256)
    assert [(c["steps"], c["octree"]) for c in stub.calls] == [(30, 256), (20, 196)]


def test_oom_ladder_never_raises_octree():
    """Третья ступень не должна поднимать разрешение выше запрошенного (128 < 144)."""
    from img2stl.inference import _oom_fallbacks

    assert _oom_fallbacks(30, 128)[-1] == (10, 128)


def test_oom_ladder_exhaustion_raises(monkeypatch, fresh_registry):
    """Исчерпание лестницы OOM → RuntimeError с причиной."""
    stub = use_stub(monkeypatch, StubPipeline(fail_first=99))
    with pytest.raises(RuntimeError, match="out of memory"):
        views_to_mesh({"front": make_rgba()}, num_inference_steps=30, octree_resolution=256)
    assert len(stub.calls) == 3  # исчерпаны все ступени лестницы


# --------------------------------------------------- регрессия one-view

def test_image_to_mesh_passes_single_image(monkeypatch, fresh_registry):
    stub = use_stub(monkeypatch, StubPipeline())
    img = make_rgba()
    image_to_mesh(img, seed=0, num_inference_steps=10, octree_resolution=128)
    call = stub.calls[0]
    assert not isinstance(call["image"], dict)  # одиночное изображение, не dict
    assert call["image"].mode == "RGBA"


# ------------------------------------------------ путь preprocessed=True (UI)

def test_views_to_mesh_preprocessed_passes_dict_as_is(monkeypatch, fresh_registry):
    """preprocessed=True: prepare_views не зовётся — в пайплайн уходит тот же dict."""
    stub = use_stub(monkeypatch, StubPipeline())
    sentinel_front, sentinel_left = object(), object()  # не-Image: prepare_views упал бы
    mesh = views_to_mesh(
        {"front": sentinel_front, "left": sentinel_left},
        preprocessed=True, num_inference_steps=10, octree_resolution=128,
    )
    assert stub.calls[0]["image"] == {"front": sentinel_front, "left": sentinel_left}
    assert mesh is not None


def test_views_to_mesh_preprocessed_empty_raises(monkeypatch, fresh_registry):
    """preprocessed=True не отменяет валидацию: пустой набор — ValueError."""
    stub = use_stub(monkeypatch, StubPipeline())
    with pytest.raises(ValueError):
        views_to_mesh({}, preprocessed=True)
    assert stub.calls == []


def test_views_to_mesh_preprocessed_unknown_key_raises(monkeypatch, fresh_registry):
    stub = use_stub(monkeypatch, StubPipeline())
    with pytest.raises(ValueError):
        views_to_mesh({"front": object(), "top": object()}, preprocessed=True)
    assert stub.calls == []


def test_image_to_mesh_preprocessed_passes_image_as_is(monkeypatch, fresh_registry):
    """preprocessed=True: prepare_image не зовётся — изображение уходит как есть."""
    stub = use_stub(monkeypatch, StubPipeline())
    sentinel = object()
    image_to_mesh(sentinel, preprocessed=True, num_inference_steps=10, octree_resolution=128)
    assert stub.calls[0]["image"] is sentinel
