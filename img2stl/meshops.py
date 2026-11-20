"""Постобработка меша: валидация, ремонт (pymeshfix), чистка, масштабирование
в мм, укладка на стол и экспорт STL/OBJ/3MF.

Чистый Python + trimesh/pymeshfix, без GPU — полностью юнит-тестируемый слой.
Контракт описан тестами tests/test_meshops.py (этап 1 плана).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshfix
import trimesh

#: форматы экспорта, поддерживаемые приложением
EXPORT_SUFFIXES = {"stl", "obj", "3mf"}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class MeshReport:
    """Сводка состояния меша для UI и логов."""

    watertight: bool
    winding_consistent: bool
    volume: float
    faces: int
    vertices: int
    degenerate_faces: int
    duplicate_faces: int
    extents: tuple[float, float, float]


def _duplicate_face_count(mesh: trimesh.Trimesh) -> int:
    """Число граней-дубликатов (одинаковое множество вершин)."""
    if len(mesh.faces) == 0:
        return 0
    _, counts = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_counts=True)
    return int(np.clip(counts - 1, 0, None).sum())


def _degenerate_face_count(mesh: trimesh.Trimesh) -> int:
    """Число вырожденных граней (нулевая площадь)."""
    if len(mesh.faces) == 0:
        return 0
    return int(len(mesh.faces) - int(mesh.nondegenerate_faces().sum()))


def validate_mesh(mesh: trimesh.Trimesh) -> MeshReport:
    """Отчёт о состоянии меша: watertight, winding, объём, габариты, мусор."""
    extents = mesh.extents if len(mesh.faces) else np.zeros(3)
    return MeshReport(
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        volume=float(mesh.volume),
        faces=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        degenerate_faces=_degenerate_face_count(mesh),
        duplicate_faces=_duplicate_face_count(mesh),
        extents=(float(extents[0]), float(extents[1]), float(extents[2])),
    )


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Ремонт меша pymeshfix (дыры, ориентация) с гарантией нормалей наружу."""
    fixer = pymeshfix.MeshFix(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )
    fixer.repair()
    # pymeshfix >= 0.18: данные ремонта — .points/.faces (.mesh требует pyvista)
    repaired = trimesh.Trimesh(vertices=fixer.points, faces=fixer.faces)
    if len(repaired.faces) and repaired.volume < 0:
        repaired.invert()  # нормали наружу: положительный объём
    return repaired


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Удаление граней-дубликатов, вырожденных граней и висячих вершин."""
    cleaned = mesh.copy()
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    return cleaned


def _require_nonempty(mesh: trimesh.Trimesh, what: str) -> None:
    """Guard: операции над пустым мешом — контролируемая ошибка, не numpy-исключение."""
    if len(mesh.faces) == 0:
        raise ValueError(f"меш пуст (0 граней): нельзя {what}")


def scale_to_mm(
    mesh: trimesh.Trimesh, target_mm: float, axis: str = "max"
) -> trimesh.Trimesh:
    """Равномерное масштабирование: заданная ось становится target_mm.

    axis: 'max' — самая длинная сторона, либо явная 'x' | 'y' | 'z'.
    """
    _require_nonempty(mesh, "масштабировать")
    target_mm = float(target_mm)
    if not np.isfinite(target_mm) or target_mm <= 0:
        raise ValueError(f"target_mm должен быть конечным положительным числом: {target_mm}")

    if axis == "max":
        idx = int(np.argmax(mesh.extents))
    elif axis in _AXIS_INDEX:
        idx = _AXIS_INDEX[axis]
    else:
        raise ValueError(f"axis должен быть 'max' | 'x' | 'y' | 'z': {axis!r}")

    current = float(mesh.extents[idx])
    if not np.isfinite(current) or current <= 0:
        raise ValueError(f"габарит меша по оси {axis!r} некорректен: {current}")

    scaled = mesh.copy()
    scaled.apply_scale(target_mm / current)
    return scaled


def place_on_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Укладка на стол печати: min Z = 0."""
    _require_nonempty(mesh, "уложить на стол")
    placed = mesh.copy()
    placed.apply_translation([0.0, 0.0, -float(placed.bounds[0][2])])
    return placed


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Экспорт меша в STL (binary) / OBJ / 3MF по расширению файла."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in EXPORT_SUFFIXES:
        raise ValueError(f"поддерживаются форматы {sorted(EXPORT_SUFFIXES)}: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def postprocess(
    mesh: trimesh.Trimesh, target_mm: float = 60.0, axis: str = "max"
) -> tuple[trimesh.Trimesh, MeshReport]:
    """Полный конвейер постобработки: чистка → ремонт → масштаб → стол."""
    result = place_on_bed(scale_to_mm(repair_mesh(clean_mesh(mesh)), target_mm, axis))
    return result, validate_mesh(result)
