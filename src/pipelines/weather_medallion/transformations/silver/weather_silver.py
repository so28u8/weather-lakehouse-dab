import dlt
from pyspark.sql import functions as F

from utilities.metadata import resolve_metadata_field

WEATHER_EXPECTATIONS = {
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_timestamp IS NOT NULL",
    "valid_temperature": "temperature_2m BETWEEN -50 AND 60",
    "valid_humidity": "relative_humidity_2m BETWEEN 0 AND 100",
    "valid_precipitation": "precipitation >= 0",
    "valid_wind": "wind_speed_10m >= 0",
}


@dlt.view(
    name="weather_silver_staged",
    comment="Flattened and validated hourly weather data before deduplication.",
)
@dlt.expect_all_or_drop(WEATHER_EXPECTATIONS)
def weather_silver_staged():
    df = dlt.read_stream("weather_bronze")

    exploded = df.select(
        resolve_metadata_field(df, "city", lambda c: c.cast("string")).alias("city"),
        resolve_metadata_field(df, "country", lambda c: c.cast("string")).alias("country"),
        resolve_metadata_field(df, "ingested_at_utc", F.to_timestamp).alias("ingested_at_utc"),
        F.explode(
            F.arrays_zip(
                F.col("hourly.time"),
                F.col("hourly.temperature_2m"),
                F.col("hourly.relative_humidity_2m"),
                F.col("hourly.precipitation"),
                F.col("hourly.wind_speed_10m"),
            )
        ).alias("h"),
    )

    return exploded.select(
        "city",
        "country",
        F.to_timestamp(F.col("h.time")).alias("forecast_timestamp"),
        F.to_date(F.col("h.time")).alias("forecast_date"),
        F.col("h.temperature_2m").cast("double").alias("temperature_2m"),
        F.col("h.relative_humidity_2m").cast("double").alias("relative_humidity_2m"),
        F.col("h.precipitation").cast("double").alias("precipitation"),
        F.col("h.wind_speed_10m").cast("double").alias("wind_speed_10m"),
        "ingested_at_utc",
    )


dlt.create_streaming_table(
    name="weather_silver",
    comment="Deduplicated hourly weather measurements (SCD Type 1).",
    cluster_by=["city", "forecast_date"],
)

dlt.apply_changes(
    target="weather_silver",
    source="weather_silver_staged",
    keys=["city", "forecast_timestamp"],
    sequence_by=F.col("ingested_at_utc"),
    stored_as_scd_type=1,
)
