import dlt
from pyspark.sql import functions as F

from utilities.metadata import resolve_metadata_field

AIR_QUALITY_EXPECTATIONS = {
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_timestamp IS NOT NULL",
    "valid_pm2_5": "pm2_5 IS NULL OR pm2_5 >= 0",
    "valid_pm10": "pm10 IS NULL OR pm10 >= 0",
    "valid_no2": "nitrogen_dioxide IS NULL OR nitrogen_dioxide >= 0",
    "valid_aqi": "european_aqi IS NULL OR european_aqi >= 0",
}


@dlt.view(
    name="air_quality_silver_staged",
    comment="Flattened and validated hourly air quality data before deduplication.",
)
@dlt.expect_all_or_drop(AIR_QUALITY_EXPECTATIONS)
def air_quality_silver_staged():
    df = dlt.read_stream("air_quality_bronze")
    zipped = F.arrays_zip(
        F.col("hourly.time"),
        F.col("hourly.pm10"),
        F.col("hourly.pm2_5"),
        F.col("hourly.nitrogen_dioxide"),
        F.col("hourly.european_aqi"),
    )
    return (
        df.withColumn("hourly_row", F.explode(zipped))
        .select(
            resolve_metadata_field(df, "city", lambda c: c.cast("string")).alias("city"),
            resolve_metadata_field(df, "country", lambda c: c.cast("string")).alias("country"),
            F.to_timestamp(F.col("hourly_row.time")).alias("forecast_timestamp"),
            F.to_date(F.col("hourly_row.time")).alias("forecast_date"),
            F.col("hourly_row.pm10").cast("double").alias("pm10"),
            F.col("hourly_row.pm2_5").cast("double").alias("pm2_5"),
            F.col("hourly_row.nitrogen_dioxide").cast("double").alias("nitrogen_dioxide"),
            F.col("hourly_row.european_aqi").cast("double").alias("european_aqi"),
            resolve_metadata_field(df, "ingested_at_utc", F.to_timestamp).alias("ingested_at_utc"),
        )
    )


dlt.create_streaming_table(
    name="air_quality_silver",
    comment="Deduplicated hourly air quality measurements (SCD Type 1).",
    cluster_by=["city", "forecast_date"],
)

dlt.apply_changes(
    target="air_quality_silver",
    source="air_quality_silver_staged",
    keys=["city", "forecast_timestamp"],
    sequence_by=F.col("ingested_at_utc"),
    stored_as_scd_type=1,
)
