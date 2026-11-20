"""Юнит-тесты постобработки меша (этап 1 плана, TDD).

Все меши синтетические, на базе trimesh.creation.box — без GPU и нейросети.
Ожидаемое поведение — из плана img2stl-plan.md, раздел «Этап 1».
"""
from pathlib import Path

import numpy as np
import pytest
import trimesh.creation

from img2stl.meshops import (
    MeshReport,
    clean_mesh,
    export_mesh,
    place_on_bed,
    postprocess,
    repair_mesh,
    scale_to_mm,
    validate_mesh,
)

EPS = 1e-6


def make_box(extents=(2.0, 3.0, 4.0)):
    return trimesh.creation.box(extents=extents)


def make_box_with_hole(extents=(2.0, 3.0, 4.0), n_faces_to_drop=2):
    """Бокс с дырой: удаляем n смежных граней → не-watertight, объём < полного."""
    box = make_box(extents)
    mask = np.ones(len(box.faces), dtype=bool)
    mask[:n_faces_to_drop] = False
    box.update_faces(mask)
    return box


# ---------------------------------------------------------------- validate_mesh

def test_validate_ok_box():
    box = make_box()
    rep = validate_mesh(box)
    assert isinstance(rep, MeshReport)
    assert rep.watertight is True
    assert rep.winding_consistent is True
    assert rep.volume == pytest.approx(24.0, abs=1e-6)
    assert rep.faces == 12
    assert rep.vertices == 8
    assert rep.degenerate_faces == 0
    assert rep.duplicate_faces == 0
    assert np.allclose(rep.extents, (2.0, 3.0, 4.0), atol=EPS)


def test_validate_detects_hole():
    holed = make_box_with_hole()
    rep = validate_mesh(holed)
    assert rep.watertight is False
    assert rep.faces == 10


def test_validate_detects_inverted_normals():
    box = make_box()
    box.invert()
    rep = validate_mesh(box)
    assert rep.watertight is True
    assert rep.volume < 0  # инвернутые нормали → отрицательный объём


# ----------------------------------------------------------------- repair_mesh

def test_repair_fixes_hole():
    holed = make_box_with_hole()
    rep_before = validate_mesh(holed)
    assert rep_before.watertight is False

    repaired = repair_mesh(holed)
    rep_after = validate_mesh(repaired)
    assert rep_after.watertight is True
    assert rep_after.volume > 0
    # дыру заросли — объём вернулся к объёму бокса (с допуском на шов ремонта)
    assert rep_after.volume == pytest.approx(24.0, rel=0.1)


def test_repair_fixes_winding():
    box = make_box()
    box.invert()
    assert validate_mesh(box).volume < 0

    repaired = repair_mesh(box)
    rep = validate_mesh(repaired)
    assert rep.watertight is True
    assert rep.volume > 0  # нормали наружу


# ------------------------------------------------------------------ clean_mesh

def test_clean_removes_degenerate_and_duplicate_faces():
    box = make_box()
    faces = np.vstack([
        box.faces,
        box.faces[:1],   # дубликат грани
        [0, 0, 1],       # вырожденная грань (нулевая площадь)
    ])
    dirty = type(box)(vertices=box.vertices, faces=faces, process=False)
    rep_dirty = validate_mesh(dirty)
    assert rep_dirty.degenerate_faces == 1
    assert rep_dirty.duplicate_faces == 1

    cleaned = clean_mesh(dirty)
    rep = validate_mesh(cleaned)
    assert rep.degenerate_faces == 0
    assert rep.duplicate_faces == 0
    assert rep.faces == 12
    assert rep.watertight is True


# ---------------------------------------------------------------- scale_to_mm

def test_scale_to_mm_max_axis():
    """План: бокс 2×3×4 → target 60 мм по макс. оси → габариты 30/45/60."""
    mesh = scale_to_mm(make_box(), target_mm=60.0, axis="max")
    rep = validate_mesh(mesh)
    assert np.allclose(sorted(rep.extents), [30.0, 45.0, 60.0], atol=1e-4)


def test_scale_to_mm_explicit_axis():
    mesh = scale_to_mm(make_box(), target_mm=40.0, axis="z")
    rep = validate_mesh(mesh)
    assert rep.extents[2] == pytest.approx(40.0, abs=1e-4)
    assert rep.extents[0] == pytest.approx(20.0, abs=1e-4)
    assert rep.extents[1] == pytest.approx(30.0, abs=1e-4)


def test_scale_to_mm_rejects_bad_axis():
    with pytest.raises(ValueError):
        scale_to_mm(make_box(), target_mm=60.0, axis="diagonal")


def test_scale_to_mm_rejects_bad_target():
    with pytest.raises(ValueError):
        scale_to_mm(make_box(), target_mm=0.0)
    with pytest.raises(ValueError):
        scale_to_mm(make_box(), target_mm=-5.0)


# --------------------------------------------------------------- place_on_bed

def test_place_on_bed():
    box = make_box()
    box.apply_translation([1.0, 2.0, 5.0])  # висит над столом
    placed = place_on_bed(box)
    assert placed.bounds[0][2] == pytest.approx(0.0, abs=1e-9)  # min Z = 0
    assert placed.bounds[1][2] == pytest.approx(4.0, abs=1e-9)  # высота бокса сохранилась


# ---------------------------------------------------------------- export_mesh

@pytest.mark.parametrize("suffix", ["stl", "obj", "3mf"])
def test_export_round_trip(tmp_path: Path, suffix: str):
    box = make_box()
    rep_src = validate_mesh(box)
    path = tmp_path / f"model.{suffix}"
    export_mesh(box, path)

    assert path.exists() and path.stat().st_size > 0
    loaded = trimesh_load(path)
    rep = validate_mesh(loaded)
    assert rep.watertight is True
    assert rep.volume == pytest.approx(rep_src.volume, rel=0.001)


def trimesh_load(path: Path):
    import trimesh
    return trimesh.load(path, force="mesh")


def test_export_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(ValueError):
        export_mesh(make_box(), tmp_path / "model.ply")


# ---------------------------------------------------------------- postprocess

def test_postprocess_full_pipeline():
    """Дырявый инвернутый бокс → ремонт + чистка + масштаб + укладка на стол."""
    dirty = make_box_with_hole(n_faces_to_drop=1)
    dirty.invert()

    mesh, rep = postprocess(dirty, target_mm=60.0, axis="max")
    assert rep.watertight is True
    assert rep.volume > 0
    assert np.allclose(sorted(rep.extents), [30.0, 45.0, 60.0], atol=1e-4)
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- guards пустого меша

def make_empty_mesh():
    return trimesh.creation.box().copy().__class__(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )


def test_validate_empty_mesh_does_not_crash():
    rep = validate_mesh(make_empty_mesh())
    assert rep.faces == 0
    assert rep.watertight is False
    assert rep.extents == (0.0, 0.0, 0.0)


def test_scale_to_mm_empty_mesh_raises():
    with pytest.raises(ValueError):
        scale_to_mm(make_empty_mesh(), target_mm=60.0)


def test_place_on_bed_empty_mesh_raises():
    with pytest.raises(ValueError):
        place_on_bed(make_empty_mesh())


def test_postprocess_empty_mesh_raises():
    with pytest.raises(ValueError):
        postprocess(make_empty_mesh(), target_mm=60.0)
