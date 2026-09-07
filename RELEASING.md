# Release astro-ndslice

**Publish a GitHub release to validate and upload to PyPI.** Tag pushes and manual
workflow runs do not publish.

## One-time setup

Create the GitHub environment `pypi`, restricted to `v*` tags. In
[PyPI Publishing settings](https://pypi.org/manage/project/astro-ndslice/settings/publishing/),
add a trusted publisher:

| Setting | Value |
| --- | --- |
| Owner / repository | `ysBach` / `astro-ndslice` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Only the upload job receives `id-token: write`; no API token is needed.
See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Each release

Use one Bash/zsh session in the repository root, with `uv` and authenticated `gh`.
Commit or merge the intended changes onto `main` first.

1. **Set the version.** Replace `1.2.3` with the intended release version.

   ```bash
   release_version=1.2.3
   git switch main &&
   uv version "$release_version" --no-sync
   ```

   `pyproject.toml` is the version source; this also updates `uv.lock` without
   syncing any environment. No separate Python version constant is maintained.

2. **Date the release notes.** Rename `## Unreleased` in `CHANGELOG.md` to
   `## v1.2.3 (YYYY-MM-DD)`, using the version above and actual release date.
   Keep this release's bullets below it. Validate with the project `.venv`:

   ```bash
   uv sync --locked --python 3.13 &&
   uv run --no-sync python .github/scripts/check_release.py \
     --release --tag "v$release_version"
   ```

3. **Commit and push.**

   ```bash
   git add pyproject.toml uv.lock CHANGELOG.md &&
   git commit -m "chore: prepare release $release_version" &&
   git push origin main
   ```

4. **Wait for CI** on that commit in
   [GitHub Actions](https://github.com/ysBach/astro-ndslice/actions).
   An optional **Publish to PyPI → Run workflow** also validates dated release
   notes. Manual runs never upload, including runs on tags.

5. **Tag that commit and publish the GitHub release.**

   ```bash
   release_version=1.2.3
   git tag -a "v$release_version" -m "Release $release_version" &&
   git push origin "v$release_version" &&
   gh release create "v$release_version" --verify-tag \
     --title "v$release_version" --generate-notes
   ```

   Check **Publish to PyPI** in Actions and the version on
   [PyPI](https://pypi.org/project/astro-ndslice/). Stop if any command fails.

## Validation

- Package and lockfile versions must match. Releases also require a matching
  `v{version}` tag and a dated, nonempty changelog entry.
- Hatchling builds one universal wheel and one sdist with `uv build --no-sources`.
- CI tests installed wheels on Python 3.10–3.14 and the sdist on Python 3.13,
  outside the checkout. Astropy tests, docstrings, formatting, and lint must pass.
- Publishing uploads those tested artifacts, without rebuilding. The sdist
  includes tests and public docs; local files and workflows are excluded.

If nothing uploaded, fix the cause and rerun failed jobs. For a partial upload,
compare hashes with the saved artifacts before uploading missing files.
Published files cannot be replaced; source fixes need a new version and tag.
