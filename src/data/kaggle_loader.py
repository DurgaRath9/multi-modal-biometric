"""Generic Kaggle dataset loader using kagglehub.

Authentication: set KAGGLE_API_TOKEN in env.dev.txt at the project root.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from env.dev.txt at project root
_ENV_FILE = Path(__file__).resolve().parents[2] / "env.dev.txt"
load_dotenv(_ENV_FILE)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(raw_path: str) -> Path:
    """Resolve a path: try as-is, then relative to project root, then Hydra cwd."""
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    candidate = _PROJECT_ROOT / p
    if candidate.exists():
        return candidate.resolve()
    # Try Hydra original cwd (Hydra changes cwd to an outputs/ dir)
    try:
        from hydra.utils import get_original_cwd

        hydra_candidate = Path(get_original_cwd()) / p
        if hydra_candidate.exists():
            return hydra_candidate.resolve()
    except Exception:
        pass
    return p.resolve()


def _find_dataset_root(path: Path) -> Path:
    """Walk into nested directories until we find person-ID subdirectories.

    Kaggle zip extraction often creates wrapper directories like:
        download_dir/dataset-name/actual-data/1/  2/  3/ ...
    This function descends through single-child directory chains and stops
    when it finds a directory whose children look like person-ID folders
    (i.e., directories named with digits).
    """
    if not path.exists():
        return path

    children = list(path.iterdir())
    subdirs = [c for c in children if c.is_dir()]

    # Check if current directory contains person-ID folders (e.g. "1", "2", ...)
    digit_dirs = [d for d in subdirs if d.name.isdigit()]
    if digit_dirs:
        logger.info("Dataset root found at %s (%d person dirs)", path, len(digit_dirs))
        return path

    # If there's exactly one subdirectory and no files, descend into it
    files = [c for c in children if c.is_file()]
    if len(subdirs) == 1 and not files:
        logger.info("Descending into wrapper directory: %s → %s", path, subdirs[0])
        return _find_dataset_root(subdirs[0])

    return path


def ensure_dataset(
    root_dir: str,
    *,
    kaggle_enabled: bool = False,
    dataset_id: str,
    download_dir: str | None = None,
) -> str:
    """Return a valid dataset path, downloading from Kaggle if needed.

    Args:
        root_dir: Expected path to the dataset (folder with person-ID subdirs).
        kaggle_enabled: Whether to auto-download from Kaggle if root_dir is missing.
        dataset_id: Kaggle dataset identifier (e.g. "owner/dataset-name").
        download_dir: Directory where kagglehub extracts the zip. Defaults to
            the parent of root_dir. After download, root_dir should point into
            the extracted contents.
    """
    resolved = _resolve_path(root_dir)
    resolved = _find_dataset_root(resolved)
    if resolved.exists() and any(resolved.iterdir()):
        logger.info("Dataset found at %s", resolved)
        return str(resolved)

    if not kaggle_enabled:
        raise FileNotFoundError(
            f"Dataset not found at {resolved}. "
            f"Set data.root_dir to the correct local folder, set data.kaggle.enabled=true, "
            f"or download manually."
        )

    # Download into download_dir (or parent of root_dir as fallback)
    dl_dir = download_dir if download_dir else str(Path(root_dir).parent)
    downloaded_path = download_kaggle_dataset(dataset_id, output_dir=dl_dir)

    # After download, search for person-ID dirs in both configured root and download path
    for search_path in [_resolve_path(root_dir), Path(downloaded_path).resolve()]:
        found = _find_dataset_root(search_path)
        if found.exists() and any(found.iterdir()):
            logger.info("Dataset ready at %s", found)
            return str(found)

    raise FileNotFoundError(
        "Kaggle download completed but dataset path is still invalid. "
        f"Configured root_dir: {root_dir}. Downloaded to: {downloaded_path}. "
        f"Check that root_dir points to the folder containing person-ID subdirectories."
    )


def download_kaggle_dataset(dataset_id: str, *, output_dir: str | None = None) -> str:
    """Download any Kaggle dataset by ID and return the local dataset path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Install kagglehub for automatic downloads: pip install kagglehub"
        ) from exc

    output_dir_resolved: str | None = None
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir_resolved = str(output_path)

    logger.info(
        "Downloading from Kaggle: %s%s",
        dataset_id,
        f" into {output_dir_resolved}" if output_dir_resolved else "",
    )
    try:
        path = kagglehub.dataset_download(dataset_id, output_dir=output_dir_resolved)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download dataset from Kaggle. "
            "If you are offline or behind a restricted network/DNS, either: "
            "(1) place the dataset at data.root_dir and run with data.kaggle.enabled=false, "
            "or (2) fix internet/DNS/proxy access and retry with data.kaggle.enabled=true."
        ) from exc

    logger.info("Dataset downloaded at %s", path)
    return path
