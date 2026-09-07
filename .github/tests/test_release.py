"""Release metadata checks using only the Python 3.11+ standard library."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "astro_ndslice_check_release",
    Path(__file__).parents[1] / "scripts/check_release.py",
)
assert spec is not None and spec.loader is not None
release_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_checks)

_VERSION = "1.2.3"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create matching project and lock metadata with an unreleased section."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "example"\nversion = "{_VERSION}"\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname = "example"\nversion = "{_VERSION}"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n- Pending fix.\n", encoding="utf-8"
    )
    return tmp_path


def _write_release_notes(project: Path, body: str) -> None:
    (project / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## v{_VERSION} (2026-09-07)\n\n{body}",
        encoding="utf-8",
    )


def test_normal_ci_accepts_unreleased_notes_and_matching_tag(project: Path) -> None:
    assert release_checks.check_release(project) == _VERSION
    assert release_checks.check_release(project, tag=f"v{_VERSION}") == _VERSION


def test_release_requires_one_dated_nonempty_section(project: Path) -> None:
    _write_release_notes(project, "- Fixed slicing.\n")
    assert (
        release_checks.check_release(project, release=True, tag=f"v{_VERSION}")
        == _VERSION
    )


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3rc1", ""])
def test_wrong_tag_is_rejected(project: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        release_checks.check_release(project, tag=tag)


@pytest.mark.parametrize(
    "lock",
    [
        "package = []\n",
        '[[package]]\nname = "example"\nversion = "1.2.2"\n',
        '[[package]]\nname = "example"\nversion = "1.2.3"\n' * 2,
    ],
)
def test_invalid_lock_entry_is_rejected(project: Path, lock: str) -> None:
    (project / "uv.lock").write_text(lock, encoding="utf-8")
    with pytest.raises(ValueError, match="uv.lock"):
        release_checks.check_release(project)


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## Unreleased\n\n- Pending.\n",
        "# Changelog\n\n##   \n\n- Malformed heading.\n",
        "# Changelog\n\n## v1.2.3\n\n- Missing date.\n",
        "# Changelog\n\n## v1.2.3 (2026-02-30)\n\n- Invalid date.\n",
        "# Changelog\n\n## v1.2.3 (2026-09-07)\n\n",
        "# Changelog\n\n## v1.2.3 (2026-09-07)\n\n- First.\n\n"
        "## v1.2.3 (2026-09-07)\n\n- Duplicate.\n",
    ],
)
def test_invalid_release_notes_are_rejected(project: Path, changelog: str) -> None:
    (project / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError):
        release_checks.check_release(project, release=True)


def test_pep440_prerelease_is_supported(project: Path) -> None:
    for filename in ("pyproject.toml", "uv.lock"):
        path = project / filename
        path.write_text(
            path.read_text(encoding="utf-8").replace(_VERSION, "1.2.3rc1"),
            encoding="utf-8",
        )
    (project / "CHANGELOG.md").write_text(
        "# Changelog\n\n## v1.2.3rc1 (2026-09-07)\n\n- Candidate.\n",
        encoding="utf-8",
    )
    assert (
        release_checks.check_release(project, release=True, tag="v1.2.3rc1")
        == "1.2.3rc1"
    )


def test_cli_writes_github_output(project: Path, monkeypatch) -> None:
    output = project / "github-output"
    monkeypatch.chdir(project)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert release_checks.main([]) == 0
    assert output.read_text(encoding="utf-8") == "version=1.2.3\n"
