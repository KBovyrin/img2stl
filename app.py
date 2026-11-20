"""Gradio UI img2stl (этап 3+4): фото/виды → генерация → файлы/метрики/превью.

Режимы: мультивью (Hunyuan3D-2mv, 1–4 ракурса, по умолчанию) и одно фото.
Локальный запуск: scripts/run.sh → http://127.0.0.1:7860
(контейнер слушает 0.0.0.0:7860, на хост публикуется только 127.0.0.1:7860)
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gradio as gr

from img2stl.inference import (
    DEFAULT_GUIDANCE,
    DEFAULT_OCTREE,
    DEFAULT_STEPS,
    image_to_mesh,
    views_to_mesh,
)
from img2stl.meshops import EXPORT_SUFFIXES, export_mesh, postprocess, validate_mesh
from img2stl.preprocess import VIEW_KEYS, prepare_image, prepare_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("img2stl.app")

OUT_DIR = Path("/app/out")

MODE_CHOICES = [("Мультивью (1–4 фото)", "mv"), ("Одно фото", "one")]
AXIS_CHOICES = [("макс. габарит", "max"), ("X", "x"), ("Y", "y"), ("Z", "z")]
FORMAT_CHOICES = [("STL (для печати)", "stl"), ("OBJ", "obj"), ("3MF", "3mf")]
FORMAT_ORDER = ("stl", "obj", "3mf")  # порядок экспорта в списке файлов


def _export_and_report(raw, target_mm, axis, formats, elapsed, mode_label, n_views):
    """Общий хвост: постобработка → экспорт → (файлы, превью-путь, метрики)."""
    rep_raw = validate_mesh(raw)
    final, rep = postprocess(raw, target_mm=float(target_mm), axis=axis)

    formats = list(formats)
    unknown = [f for f in formats if f not in EXPORT_SUFFIXES]
    if unknown:
        raise gr.Error(f"Неизвестный формат экспорта: {', '.join(map(str, unknown))}")

    run_dir = OUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = [export_mesh(final, run_dir / f"model.{fmt}") for fmt in sorted(formats, key=FORMAT_ORDER.index)]
    preview_path = run_dir / "model.glb"
    final.export(preview_path)
    log.info("Готово: %s | %s | режим=%s видов=%d", run_dir, rep.extents, mode_label, n_views)

    if not rep.watertight:
        gr.Warning("Меш остался не-watertight после ремонта — проверьте модель в слайсере перед печатью.")

    metrics = {
        "режим": mode_label,
        "видов": n_views,
        "время, с": round(elapsed, 1),
        "сырой меш": {"граней": rep_raw.faces, "watertight": rep_raw.watertight},
        "финал": asdict(rep),
    }
    return [str(p) for p in paths], str(preview_path), metrics


def generate(mode, image, view_front, view_back, view_left, view_right,
             target_mm: float, axis: str, formats: list[str],
             steps: int, octree: int, guidance: float, seed: int):
    """Основной сценарий: фото/виды → сырой меш → постобработка → файлы + метрики."""
    if not formats:
        raise gr.Error("Выберите хотя бы один формат экспорта")

    t_start = time.perf_counter()
    if mode == "mv":
        views = {"front": view_front, "back": view_back,
                 "left": view_left, "right": view_right}
        filled = {k: v for k, v in views.items() if v is not None}
        if not filled:
            raise gr.Error("Загрузите хотя бы один вид: Front / Back / Left / Right")
        rgba_views, white_views = prepare_views(filled)  # фон снимается один раз на вид
        raw = views_to_mesh(
            rgba_views, seed=int(seed), num_inference_steps=int(steps),
            guidance_scale=float(guidance), octree_resolution=int(octree), preprocessed=True,
        )
        gallery = [(white_views[k], k) for k in VIEW_KEYS if k in white_views]
        mode_label, n_views = "мультивью", len(rgba_views)
    else:
        if image is None:
            raise gr.Error("Сначала загрузите изображение")
        rgba, white = prepare_image(image)  # фон снимаем ровно один раз
        raw = image_to_mesh(
            rgba, seed=int(seed), num_inference_steps=int(steps),
            guidance_scale=float(guidance), octree_resolution=int(octree), preprocessed=True,
        )
        gallery = [(white, "фото")]
        mode_label, n_views = "одно фото", 1

    elapsed = time.perf_counter() - t_start
    paths, preview_path, metrics = _export_and_report(
        raw, target_mm, axis, formats, elapsed, mode_label, n_views
    )
    return gallery, preview_path, paths, metrics


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="img2stl — фото → 3D-модель") as demo:
        gr.Markdown("# img2stl — фото → 3D-модель для печати\n"
                    "Мультивью: 2–4 ракурса одного объекта (front/back/left/right) дают точный тыл. "
                    "Одно фото — быстрый режим. Генерация: Hunyuan3D-2 (локально, GPU).")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(MODE_CHOICES, value="mv", label="Режим генерации")

                with gr.Group(visible=True) as mv_group:
                    gr.Markdown("*Ракурсы одного объекта: камера на уровне центра, вид строго сбоку/спереди/сзади.*")
                    with gr.Row():
                        view_front = gr.Image(label="Front", type="pil", image_mode="RGBA", height=150)
                        view_back = gr.Image(label="Back", type="pil", image_mode="RGBA", height=150)
                    with gr.Row():
                        view_left = gr.Image(label="Left", type="pil", image_mode="RGBA", height=150)
                        view_right = gr.Image(label="Right", type="pil", image_mode="RGBA", height=150)

                with gr.Group(visible=False) as one_group:
                    image_in = gr.Image(label="Фото объекта", type="pil", image_mode="RGBA", height=260)

                target_mm = gr.Slider(10, 300, value=60.0, step=1.0, label="Целевой размер, мм")
                axis = gr.Radio(AXIS_CHOICES, value="max", label="Ось масштабирования")
                formats = gr.CheckboxGroup(FORMAT_CHOICES, value=["stl"], label="Форматы экспорта")
                with gr.Accordion("Дополнительно", open=False):
                    steps = gr.Slider(10, 50, value=DEFAULT_STEPS, step=1, label="Шаги диффузии")
                    octree = gr.Slider(
                        128, 384, value=DEFAULT_OCTREE, step=64,
                        label="Детализация меша (октодерево)",
                        info="256 — стандарт; 384 — выше детализация, дольше и больше граней",
                    )
                    guidance = gr.Slider(
                        1.0, 10.0, value=DEFAULT_GUIDANCE, step=0.5,
                        label="Guidance scale (следование фото)",
                        info="5 — стандарт; выше — точнее по фото, но возможны артефакты",
                    )
                    seed = gr.Number(value=0, label="Seed", precision=0)
                btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                preview_gallery = gr.Gallery(label="После снятия фона", height=260)
                model_3d = gr.Model3D(label="3D-превью", camera_position=(90, -70, 120))
                files_out = gr.Files(label="Файлы для скачивания")
                metrics_out = gr.JSON(label="Метрики меша")

        def toggle_mode(selected):
            return (gr.update(visible=selected == "mv"),
                    gr.update(visible=selected == "one"))

        mode.change(toggle_mode, inputs=mode, outputs=[mv_group, one_group])

        btn.click(
            generate,
            inputs=[mode, image_in, view_front, view_back, view_left, view_right,
                    target_mm, axis, formats, steps, octree, guidance, seed],
            outputs=[preview_gallery, model_3d, files_out, metrics_out],
        )
    return demo


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    demo.queue()  # GPU один: очередь Gradio поверх gpu_lock инференса
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, allowed_paths=[str(OUT_DIR)])
