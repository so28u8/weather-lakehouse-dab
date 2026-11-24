from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, functions as F

from utilities.metadata import INGEST_METADATA_SCHEMA_HINT, resolve_source_file_path


def read_raw_json_stream(spark: SparkSession, landing_path: str) -> DataFrame:
    """Read raw JSON payloads via Auto Loader with ingest_metadata hints and lineage columns."""
    raw_df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("cloudFiles.schemaHints", INGEST_METADATA_SCHEMA_HINT)
        .load(landing_path)
    )
    return (
        raw_df
        .withColumn("_bronze_ingested_at", F.current_timestamp())
        .withColumn("_source_file_path", resolve_source_file_path(raw_df))
    )
