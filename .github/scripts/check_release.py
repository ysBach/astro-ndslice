"""Validate package versions and release notes before building distributions."""

from __future__ import annotations

import argparse
import logging
import os
import re
import tomllib
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

_SECTION_HEADING = re.compile(r"^##[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)


def _check_changelog(root: Path, version: str) -> None:
    """Require one dated, nonempty Markdown section for ``version``."""
    lines = (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    headings = [
        index
        for index, line in enumerate(lines)
        if _SECTION_HEADING.fullmatch(line) is not None
    ]
    matching = [
        index
        for index in headings
        if lines[index].split(maxsplit=2)[1:2] == [f"v{version}"]
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one dated changelog entry for {version}")

    start = matching[0]
    match = re.fullmatch(
        rf"##[ \t]+v{re.escape(version)}[ \t]+\("
        rf"(?P<date>[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})\)[ \t]*",
        lines[start],
    )
    if match is None:
        raise ValueError(f"Changelog heading must be v{version} (YYYY-MM-DD)")
    try:
        date.fromisoformat(match.group("date"))
    except ValueError as error:
        raise ValueError(
            f"Changelog entry for {version} has an invalid date"
        ) from error

    end = next((index for index in headings if index > start), len(lines))
    if not any(line.strip() for line in lines[start + 1 : end]):
        raise ValueError(f"Changelog entry for {version} is empty")


def check_release(root: Path, *, release: bool = False, tag: str | None = None) -> str:
    """Check project metadata and return the package version.

    Parameters
    ----------
    root : pathlib.Path
        Repository root containing ``pyproject.toml``, ``uv.lock``, and the
        Markdown changelog.
    release : bool, optional
        Require one dated, nonempty changelog entry for this version.
    tag : str, optional
        Tag to compare with ``v{version}``.

    Returns
    -------
    str
        Validated project version.

    Raises
    ------
    ValueError
        If metadata, the optional tag, or release notes are invalid.
    OSError
        If a required repository file cannot be read.
    """
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    name = project["name"]
    version = project["version"]
    packages = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))["package"]
    matching = [p for p in packages if p["name"] == name]
    if len(matching) != 1 or matching[0].get("version") != version:
        raise ValueError(f"uv.lock must contain one {name} entry at version {version}")

    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} must match v{version}")
    if release:
        _check_changelog(root, version)
    return version


def main(argv: list[str] | None = None) -> int:
    """Validate release metadata and write the optional GitHub output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        version = check_release(Path.cwd(), release=args.release, tag=args.tag)
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"version={version}\n")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
    log.info("Validated astro-ndslice %s", version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
