"""Tests for Kaggle dataset download path handling."""

import tempfile
from pathlib import Path

from src.data.kaggle_loader import _find_dataset_root, ensure_dataset


class _FakeKaggleHub:
    def __init__(self, on_download):
        self._on_download = on_download

    def dataset_download(self, handle, path=None, force_download=False, output_dir=None):
        return self._on_download(
            handle=handle,
            path=path,
            force_download=force_download,
            output_dir=output_dir,
        )


def test_ensure_dataset_downloads_into_download_dir(monkeypatch):
    calls = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir) / "data"
        configured_root = download_dir / "my-dataset"

        def _download(**kwargs):
            calls.update(kwargs)
            # Simulate kagglehub extracting into download_dir with nested wrapper
            target = Path(kwargs["output_dir"])
            nested = target / "my-dataset"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "1").mkdir()
            (nested / "1" / "file.bmp").write_text("ok", encoding="utf-8")
            return str(target)

        monkeypatch.setitem(__import__("sys").modules, "kagglehub", _FakeKaggleHub(_download))

        resolved = ensure_dataset(
            root_dir=str(configured_root),
            kaggle_enabled=True,
            dataset_id="owner/dataset",
            download_dir=str(download_dir),
        )

        # Should find person-ID dirs inside configured_root
        assert Path(resolved) == configured_root.resolve()
        assert calls["output_dir"] == str(download_dir.resolve())


def test_ensure_dataset_prefers_configured_root_after_download(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        download_dir = base / "data"
        configured_root = download_dir / "configured-root"
        alternate_download_path = base / "kaggle-cache-location"

        def _download(**kwargs):
            # Simulate downloader populating configured root but returning a different path.
            target = Path(kwargs["output_dir"])
            cr = target / "configured-root"
            cr.mkdir(parents=True, exist_ok=True)
            (cr / "1").mkdir()
            (cr / "1" / "file.bmp").write_text("done", encoding="utf-8")

            alternate_download_path.mkdir(parents=True, exist_ok=True)
            (alternate_download_path / "1").mkdir()
            (alternate_download_path / "1" / "cache.bmp").write_text("cache", encoding="utf-8")
            return str(alternate_download_path)

        monkeypatch.setitem(__import__("sys").modules, "kagglehub", _FakeKaggleHub(_download))

        resolved = ensure_dataset(
            root_dir=str(configured_root),
            kaggle_enabled=True,
            dataset_id="owner/dataset",
            download_dir=str(download_dir),
        )

        assert Path(resolved) == configured_root.resolve()


class TestFindDatasetRoot:
    def test_finds_person_dirs_at_top_level(self):
        """If root already contains person-ID dirs, return it as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "1").mkdir()
            (root / "2").mkdir()
            (root / "1" / "file.bmp").write_text("x", encoding="utf-8")

            result = _find_dataset_root(root)
            assert result == root

    def test_descends_single_wrapper(self):
        """Unwrap one wrapper directory to find person-ID dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inner = Path(tmpdir) / "wrapper"
            inner.mkdir()
            (inner / "1").mkdir()
            (inner / "2").mkdir()
            (inner / "1" / "file.bmp").write_text("x", encoding="utf-8")

            result = _find_dataset_root(Path(tmpdir))
            assert result == inner

    def test_descends_multiple_wrappers(self):
        """Recursively unwrap nested wrappers (e.g. zip-in-zip extraction)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir) / "outer" / "inner"
            deep.mkdir(parents=True)
            (deep / "1").mkdir()
            (deep / "2").mkdir()
            (deep / "1" / "file.bmp").write_text("x", encoding="utf-8")

            result = _find_dataset_root(Path(tmpdir))
            assert result == deep

    def test_stops_at_multiple_children(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a").mkdir()
            (root / "b").mkdir()

            result = _find_dataset_root(root)
            assert result == root

    def test_stops_when_files_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "child").mkdir()
            (root / "README.txt").write_text("hi", encoding="utf-8")

            result = _find_dataset_root(root)
            assert result == root

    def test_noop_for_flat_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file.txt").write_text("x", encoding="utf-8")

            result = _find_dataset_root(root)
            assert result == root

    def test_ensure_dataset_finds_nested_existing(self):
        """ensure_dataset auto-descends into nested wrappers for existing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outer = Path(tmpdir) / "dataset"
            inner = outer / "dataset"
            inner.mkdir(parents=True)
            (inner / "1").mkdir()
            (inner / "2").mkdir()
            (inner / "1" / "file.bmp").write_text("x", encoding="utf-8")
            (inner / "2" / "file.bmp").write_text("x", encoding="utf-8")

            resolved = ensure_dataset(
                root_dir=str(outer),
                kaggle_enabled=False,
                dataset_id="owner/dataset",
            )
            assert Path(resolved) == inner
