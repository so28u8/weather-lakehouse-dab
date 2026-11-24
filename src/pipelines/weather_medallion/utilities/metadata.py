from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import StructType

INGEST_METADATA_COL = "ingest_metadata"
# Legacy payload key retained for backward compatibility with pre-migration dev files.
LEGACY_METADATA_COL = "_metadata"
INGEST_METADATA_SCHEMA_HINT = (
    f"{INGEST_METADATA_COL} STRUCT<"
    "city: STRING, "
    "country: STRING, "
    "ingested_at_utc: STRING, "
    "source: STRING"
    ">"
)


def has_struct_field(df: DataFrame, col_name: str, field_name: str) -> bool:
    """Return True when df has a top-level StructType column containing field_name."""
    if col_name not in df.schema.fieldNames():
        return False
    dtype = df.schema[col_name].dataType
    return isinstance(dtype, StructType) and field_name in dtype.fieldNames()


def resolve_source_file_path(raw_df: DataFrame) -> Column:
    """Safely capture Spark hidden file_path unless legacy JSON `_metadata` shadows it."""
    if LEGACY_METADATA_COL not in raw_df.schema.fieldNames():
        return F.col(f"{LEGACY_METADATA_COL}.file_path")
    if has_struct_field(raw_df, LEGACY_METADATA_COL, "file_path"):
        return F.col(f"{LEGACY_METADATA_COL}.file_path")
    # Legacy JSON `_metadata` shadows Spark's hidden `_metadata` struct until old dev files are retired.
    return F.lit(None).cast("string")


def resolve_metadata_field(
    df: DataFrame,
    field_name: str,
    transform: Callable[[Column], Column] | None = None,
) -> Column:
    """Resolve a metadata subfield from `ingest_metadata`, falling back to legacy `_metadata` if present."""
    candidates: list[Column] = []
    for col_name in (INGEST_METADATA_COL, LEGACY_METADATA_COL):
        if has_struct_field(df, col_name, field_name):
            expr = F.col(f"{col_name}.{field_name}")
            if transform is not None:
                # Apply transform per branch before coalescing to avoid ANSI mixed-type errors.
                expr = transform(expr)
            candidates.append(expr)

    if not candidates:
        raise ValueError(
            f"Metadata field '{field_name}' not found under '{INGEST_METADATA_COL}' "
            f"or legacy '{LEGACY_METADATA_COL}' in Bronze schema: {df.schema.fieldNames()}"
        )
    if len(candidates) == 1:
        return candidates[0]
    return F.coalesce(*candidates)
