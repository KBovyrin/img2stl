"""Инференс-обёртка Hunyuan3D-2 shape-пайплайнов: one-view и мультивью (2mv).

Оба пайплайна лениво грузятся при первом использовании (fp16, cuda) и остаются
в VRAM; доступ к GPU сериализуется общим lock; при CUDA OOM — лестница ретраев
с уменьшенными параметрами. Верификация — smoke-CLI (не юнит-тестами), слой
реестра покрыт тестами через шов _load_pipeline.
"""
from __future__ import annotations

import logging
import threading
import time

import torch
import trimesh
from PIL import Image

from .preprocess import prepare_image, prepare_views, validate_views

log = logging.getLogger("img2stl.inference")

#: one-view: репозиторий весов shape-модели (качается в HF_HOME при первом запуске)
MODEL_REPO = "tencent/Hunyuan3D-2"
SUBFOLDER = "hunyuan3d-dit-v2-0"
#: мультивью: 1–4 ракурса одного объекта, закрывает «додуманный» тыл
MODEL_REPO_MV = "tencent/Hunyuan3D-2mv"
SUBFOLDER_MV = "hunyuan3d-dit-v2-mv"

#: доступные пайплайны: ключ → (HF-репо, subfolder)
PIPELINE_SPECS = {
    "one": (MODEL_REPO, SUBFOLDER),
    "mv": (MODEL_REPO_MV, SUBFOLDER_MV),
}

#: параметры генерации по умолчанию (шаги/разрешение — баланс качества и VRAM)
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 5.0
DEFAULT_OCTREE = 256
DEFAULT_NUM_CHUNKS = 8000

_pipelines: dict = {}
_pipeline_lock = threading.Lock()  # потокобезопасное создание пайплайнов
gpu_lock = threading.Lock()  # GPU один: сериализация всех генераций


def _load_pipeline(name: str):
    """Загрузка пайплайна по ключу PIPELINE_SPECS (шов для тестов)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    repo, subfolder = PIPELINE_SPECS[name]
    t0 = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        repo,
        subfolder=subfolder,
        use_safetensors=True,
        variant="fp16",
        dtype=torch.float16,
        device="cuda",
    )
    log.info("Пайплайн %s (%s/%s) загружен за %.1f c",
             name, repo, subfolder, time.perf_counter() - t0)
    return pipe


def get_pipeline(name: str = "one"):
    """Пайплайн по ключу ('one' | 'mv'): лениво, грузится один раз и остаётся в VRAM."""
    if name not in PIPELINE_SPECS:
        raise ValueError(f"неизвестный пайплайн {name!r}; доступны {sorted(PIPELINE_SPECS)}")
    if name not in _pipelines:
        with _pipeline_lock:
            if name not in _pipelines:
                log.info("Загрузка пайплайна %s (fp16, cuda)…", name)
                _pipelines[name] = _load_pipeline(name)
    return _pipelines[name]


def _oom_fallbacks(steps: int, octree: int) -> list[tuple[int, int]]:
    """Лестница параметров для ретрая при OOM: (шаги, разрешение октодерева)."""
    plan = [
        (steps, octree),
        (min(steps, 20), min(octree, 196)),
        (10, min(octree, 144)),  # не поднимаем разрешение выше запрошенного
    ]
    seen, unique = set(), []
    for item in plan:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _generate(
    pipe,
    image,
    *,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    octree_resolution: int,
):
    """Общий цикл генерации с OOM-ретраями; image — RGBA или dict[RGBA] для mv."""
    last_error: BaseException | None = None
    for steps_try, octree_try in _oom_fallbacks(num_inference_steps, octree_resolution):
        try:
            generator = torch.Generator().manual_seed(int(seed))
            log.info(
                "Генерация: steps=%d guidance=%.1f octree=%d …",
                steps_try, guidance_scale, octree_try,
            )
            t0 = time.perf_counter()
            outputs = pipe(
                image=image,
                num_inference_steps=steps_try,
                guidance_scale=guidance_scale,
                octree_resolution=octree_try,
                num_chunks=DEFAULT_NUM_CHUNKS,
                generator=generator,
                output_type="trimesh",
                enable_pbar=False,
            )
            log.info("Генерация заняла %.1f c", time.perf_counter() - t0)
            if not outputs:
                raise RuntimeError("пайплайн вернул пустой результат")
            # output_type="trimesh": список мешей на батч; берём первый,
            # вложенность отличается от аннотации — обрабатываем оба варианта
            first = outputs[0]
            return first[0] if isinstance(first, list) else first
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            log.warning(
                "CUDA OOM (steps=%d, octree=%d) — ретрай с меньшими параметрами",
                steps_try, octree_try,
            )
            torch.cuda.empty_cache()
    raise RuntimeError(
        f"CUDA out of memory на всех наборах параметров: {last_error}"
    ) from last_error


def image_to_mesh(
    image: Image.Image,
    *,
    seed: int = 0,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    octree_resolution: int = DEFAULT_OCTREE,
    preprocessed: bool = False,
) -> trimesh.Trimesh:
    """One-view: PIL.Image → сырой меш (до постобработки meshops).

    preprocessed=True — изображение уже прошло prepare_image (фон снят):
    препроцессинг не повторяется, ответственность за вызов на стороне UI.
    """
    rgba = image if preprocessed else prepare_image(image)[0]
    with gpu_lock:
        return _generate(
            get_pipeline("one"), rgba,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )


def views_to_mesh(
    views: dict,
    *,
    seed: int = 0,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    octree_resolution: int = DEFAULT_OCTREE,
    preprocessed: bool = False,
) -> trimesh.Trimesh:
    """Мультивью: dict {front|back|left|right: PIL.Image} → сырой меш (Hunyuan3D-2mv).

    Допустим 1–4 вида (None-слоты отбрасываются, пустой набор — ValueError);
    фон снимается каждому виду отдельно. preprocessed=True — виды уже прошли
    prepare_views (фон снят), но валидация набора выполняется всегда.
    """
    views = validate_views(views)
    rgba_views = views if preprocessed else prepare_views(views)[0]
    with gpu_lock:
        return _generate(
            get_pipeline("mv"), rgba_views,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )
