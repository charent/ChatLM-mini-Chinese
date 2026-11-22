"""路径模块冒烟测试。"""

from pathlib import Path

from chatlm.paths import ARTIFACT_DIR, DATA_DIR, PROJECT_ROOT


def test_project_root_is_repo_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "chatlm").is_dir()


def test_dirs_are_absolute_paths() -> None:
    assert isinstance(DATA_DIR, Path) and DATA_DIR.is_absolute()
    assert isinstance(ARTIFACT_DIR, Path) and ARTIFACT_DIR.is_absolute()


def test_default_dirs_live_under_project_root() -> None:
    assert DATA_DIR.parent == PROJECT_ROOT
    assert ARTIFACT_DIR.parent == PROJECT_ROOT
