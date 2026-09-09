from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_run_exports import prepare


def test_migrate_keeps_contents_and_old_paths(tmp_path):
    work, dest = tmp_path / "work", tmp_path / "export"
    source = work / "cvat_export_3d"
    source.mkdir(parents=True)
    (source / "task.zip").write_bytes(b"existing archive")
    prepare(work, dest)
    assert source.is_symlink()
    assert (dest / "cvat_export_3d/task.zip").read_bytes() == b"existing archive"
    assert (source / "task.zip").read_bytes() == b"existing archive"
    prepare(work, dest)
    (work / "cvat_export/new.json").write_text("new export")
    assert (dest / "cvat_export/new.json").read_text() == "new export"


def test_collision_refuses_before_any_migration(tmp_path):
    work, dest = tmp_path / "work", tmp_path / "export"
    (work / "cvat_export").mkdir(parents=True)
    (work / "cvat_export_road").mkdir()
    (dest / "cvat_export_road").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare(work, dest)
    assert not (work / "cvat_export").is_symlink()
