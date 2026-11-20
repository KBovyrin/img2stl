"""Препроцессинг входного изображения: снятие фона → объект для пайплайна.

Политика как в референс-приложении Hunyuan3D-2: если у картинки уже есть
альфа-канал (PNG с прозрачностью) — используем её, иначе снимаем фон rembg
(обёртка hy3dgen.rembg.BackgroundRemover, u2net кэшируется в U2NET_HOME).
"""
from __future__ import annotations

import logging
import threading

from PIL import Image

log = logging.getLogger("img2stl.preprocess")

#: канонические виды мультивью-режима (Hunyuan3D-2mv)
VIEW_KEYS = ("front", "back", "left", "right")

_remover = None
_remover_lock = threading.Lock()


def _get_remover():
    """Lazy singleton снятия фона (первый вызов скачивает ~170 МБ u2net)."""
    global _remover
    if _remover is not None:
        return _remover
    with _remover_lock:
        if _remover is None:
            from hy3dgen.rembg import BackgroundRemover

            _remover = BackgroundRemover()
    return _remover


def has_alpha(image: Image.Image) -> bool:
    """Есть ли у изображения значимая прозрачность."""
    if image.mode == "RGBA" or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").split()[3]
        return alpha.getextrema()[0] < 255
    return False


def on_white_background(image: Image.Image) -> Image.Image:
    """RGBA → объект на белом фоне (для превью UI)."""
    rgba = image.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def prepare_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Входное фото → (RGBA без фона — в пайплайн, RGB на белом — для UI)."""
    if image.mode != "RGB" and not has_alpha(image):
        image = image.convert("RGB")

    if has_alpha(image):
        log.info("Фон: используется альфа-канал изображения")
        rgba = image.convert("RGBA")
    else:
        log.info("Фон: снятие фона rembg/u2net")
        t0 = __import__("time").perf_counter()
        rgba = _get_remover()(image.convert("RGB"))
        log.info("Фон снят за %.1f c", __import__("time").perf_counter() - t0)

    return rgba, on_white_background(rgba)


def validate_views(views: dict) -> dict:
    """Нормализация набора видов: выкинуть None-слоты, проверить ключи и непустоту."""
    unknown = set(views) - set(VIEW_KEYS)
    if unknown:
        raise ValueError(f"неизвестные виды {sorted(unknown)}; допустимы {list(VIEW_KEYS)}")
    normalized = {key: views[key] for key in VIEW_KEYS if views.get(key) is not None}
    if not normalized:
        raise ValueError(f"нужен хотя бы один вид из {list(VIEW_KEYS)}")
    return normalized


def prepare_views(views: dict) -> tuple[dict, dict]:
    """Виды → (dict[RGBA без фона] для пайплайна, dict[RGB на белом] для UI).

    Фон снимается каждому виду отдельно, ровно один раз на вид.
    """
    views = validate_views(views)
    rgba_views, white_views = {}, {}
    for key, image in views.items():
        log.info("Вид %s: подготовка", key)
        rgba_views[key], white_views[key] = prepare_image(image)
    return rgba_views, white_views
