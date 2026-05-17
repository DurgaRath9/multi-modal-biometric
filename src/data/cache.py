"""PyArrow-based metadata caching for efficient dataset operations.

Using Apache Arrow's columnar format for:
- Fast metadata lookups without scanning the filesystem
- Low memory overhead
- Efficient batch reads and filtering
"""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def build_metadata_table(root_dir: str) -> pa.Table:
    """Scan dataset directory and build an Arrow table of all image metadata.

    Columns: person_id, modality, filename, filepath, file_size_bytes
    """
    root = Path(root_dir)
    rows: list[dict] = []

    image_exts = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.isdigit():
            continue
        person_id = int(person_dir.name)

        for modality_dir in sorted(person_dir.iterdir()):
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name  # "left", "right", "fingerprint"

            for img_file in sorted(modality_dir.iterdir()):
                if img_file.suffix.lower() not in image_exts:
                    continue
                rows.append(
                    {
                        "person_id": person_id,
                        "modality": modality,
                        "filename": img_file.name,
                        "filepath": str(img_file),
                        "file_size_bytes": img_file.stat().st_size,
                    }
                )

    table = pa.table(
        {
            "person_id": pa.array([r["person_id"] for r in rows], type=pa.int32()),
            "modality": pa.array([r["modality"] for r in rows], type=pa.string()),
            "filename": pa.array([r["filename"] for r in rows], type=pa.string()),
            "filepath": pa.array([r["filepath"] for r in rows], type=pa.string()),
            "file_size_bytes": pa.array([r["file_size_bytes"] for r in rows], type=pa.int64()),
        }
    )
    logger.info("Built metadata table: %d rows, %d columns", table.num_rows, table.num_columns)
    return table


def save_metadata_cache(table: pa.Table, cache_path: str) -> None:
    """Save Arrow table to Parquet for fast reload."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    logger.info("Metadata cache saved to %s", cache_path)


def load_metadata_cache(cache_path: str) -> pa.Table:
    """Load cached metadata from Parquet."""
    table = pq.read_table(cache_path)
    logger.info("Loaded metadata cache: %d rows", table.num_rows)
    return table


def get_image_paths_by_modality(table: pa.Table, modality: str) -> list[str]:
    """Filter metadata table and return image paths for a given modality."""
    mask = pa.compute.equal(table.column("modality"), modality)
    filtered = table.filter(mask)
    return filtered.column("filepath").to_pylist()


def get_dataset_summary(table: pa.Table) -> dict:
    """Return summary statistics from the metadata table."""
    df = table.to_pandas()
    return {
        "total_images": len(df),
        "num_persons": df["person_id"].nunique(),
        "modalities": df["modality"].unique().tolist(),
        "images_per_modality": df.groupby("modality").size().to_dict(),
        "total_size_mb": round(df["file_size_bytes"].sum() / (1024 * 1024), 2),
    }
