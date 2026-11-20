"""Smoke-прогон инференса: python -m img2stl.smoke [--img FILE | --views DIR] --out out

Наблюдаемая верификация: в логах видно загрузку весов (первый запуск), имя GPU,
время генерации, метрики меша до/после ремонта; на диске появляются
out/raw.glb и out/final.stl.

--img    — одно фото (one-view пайплайн);
--views  — каталог с видами front/back/left/right (.png/.jpg/.jpeg, 1–4 шт.,
          мультивью-пайплайн Hunyuan3D-2mv). Аргументы взаимоисключаемы.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from .inference import image_to_mesh, views_to_mesh
from .meshops import export_mesh, postprocess, repair_mesh, validate_mesh
from .preprocess import VIEW_KEYS

log = logging.getLogger("img2stl.smoke")

#: допустимые расширения файлов видов
VIEW_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_views_from_dir(directory: Path) -> dict:
    """Каталог → dict видов {front|back|left|right: RGBA}; ни одного файла → ValueError."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"каталог видов не найден: {directory}")
    views: dict = {}
    for key in VIEW_KEYS:
        found = None
        for ext in VIEW_EXTENSIONS:
            candidate = directory / f"{key}{ext}"
            if candidate.exists():
                if found is not None:
                    log.warning("Вид %s: найдены дубликаты %s и %s — берётся %s",
                                key, found.name, candidate.name, found.name)
                    break
                found = candidate
        if found is not None:
            views[key] = Image.open(found).convert("RGBA")
    if not views:
        raise ValueError(
            f"в {directory} нет видов {list(VIEW_KEYS)} с расширениями {list(VIEW_EXTENSIONS)}"
        )
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description="img2stl smoke: фото/виды → STL")
    parser.add_argument("--img", type=Path, default=None, help="входное фото (one-view)")
    parser.add_argument("--views", type=Path, default=None,
                        help="каталог видов front/back/left/right (мультивью)")
    parser.add_argument("--out", type=Path, default=Path("out"), help="каталог результатов")
    parser.add_argument("--target-mm", type=float, default=60.0, help="целевой размер по макс. оси, мм")
    parser.add_argument("--steps", type=int, default=30, help="шаги диффузии")
    parser.add_argument("--octree", type=int, default=256, help="разрешение октодерева")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (args.img is None) == (args.views is None):
        parser.error("укажите ровно один из аргументов --img / --views")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    log.info("GPU: %s", torch.cuda.get_device_name(0))

    t0 = time.perf_counter()
    if args.img is not None:
        image = Image.open(args.img)
        log.info("Вход: %s (%s, %dx%d) — режим one-view", args.img, image.mode, *image.size)
        raw = image_to_mesh(
            image, seed=args.seed, num_inference_steps=args.steps, octree_resolution=args.octree
        )
    else:
        views = load_views_from_dir(args.views)
        log.info("Вход: %s — режим мультивью, виды: %s", args.views, sorted(views))
        raw = views_to_mesh(
            views, seed=args.seed, num_inference_steps=args.steps, octree_resolution=args.octree
        )
    t_gen = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    raw.export(args.out / "raw.glb")

    rep_raw = validate_mesh(raw)
    repaired = repair_mesh(raw)
    rep_repaired = validate_mesh(repaired)

    final, rep_final = postprocess(raw, target_mm=args.target_mm, axis="max")
    export_mesh(final, args.out / "final.stl")

    log.info("--- метрики ---------------------------------------------")
    log.info("генерация:          %.1f c (steps=%d, octree=%d)", t_gen, args.steps, args.octree)
    log.info("сырой меш:          faces=%d, watertight=%s, volume=%.4f",
             rep_raw.faces, rep_raw.watertight, rep_raw.volume)
    log.info("после ремонта:      faces=%d, watertight=%s, volume=%.4f",
             rep_repaired.faces, rep_repaired.watertight, rep_repaired.volume)
    log.info("watertight:         %s → %s", rep_raw.watertight, rep_final.watertight)
    log.info("финал (мм):         %.2f x %.2f x %.2f, min Z=%.3f",
             *rep_final.extents, final.bounds[0][2])
    log.info("файлы:              %s, %s", args.out / "raw.glb", args.out / "final.stl")


if __name__ == "__main__":
    main()
