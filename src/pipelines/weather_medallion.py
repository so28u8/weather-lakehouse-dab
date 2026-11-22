import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

LANDING_PATH = spark.conf.get("landing_volume_path")
AIR_QUALITY_LANDING_PATH = spark.conf.get("air_quality_landing_volume_path")


@dlt.table(
    name="weather_bronze",
    comment="Nyers Open-Meteo JSON adatok a landing Volume-ból (Auto Loader)",
)
def weather_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(LANDING_PATH)
        .withColumn("_bronze_ingested_at", F.current_timestamp())
    )

@dlt.view(
    name="weather_silver_staged",
    comment="Kibontott és tisztított óránkénti adatok deduplikáció előtt",
)
@dlt.expect_all_or_drop({
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_time IS NOT NULL",
    "valid_temperature": "temperature_2m BETWEEN -50 AND 60",
    "valid_humidity": "relative_humidity_2m BETWEEN 0 AND 100",
    "valid_precipitation": "precipitation >= 0",
    "valid_wind": "wind_speed_10m >= 0",
})
def weather_silver_staged():
    df = dlt.read_stream("weather_bronze")

    exploded = df.select(
        F.col("_metadata.city").alias("city"),
        F.col("_metadata.country").alias("country"),
        F.to_timestamp(F.col("_metadata.ingested_at_utc")).alias("ingested_at_utc"),
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
        F.to_timestamp(F.col("h.time")).alias("forecast_time"),
        F.to_date(F.col("h.time")).alias("forecast_date"),  # <-- EZT ADD HOZZÁ
        F.col("h.temperature_2m").cast("double").alias("temperature_2m"),
        F.col("h.relative_humidity_2m").cast("double").alias("relative_humidity_2m"),
        F.col("h.precipitation").cast("double").alias("precipitation"),
        F.col("h.wind_speed_10m").cast("double").alias("wind_speed_10m"),
        "ingested_at_utc",
    )


dlt.create_streaming_table(
    name="weather_silver",
    comment="Deduplikált óránkénti időjárás-adatok (városonként és óránként csak a legfrissebb)",
)

dlt.apply_changes(
    target="weather_silver",
    source="weather_silver_staged",
    keys=["city", "forecast_time"],
    sequence_by=F.col("ingested_at_utc"),
    stored_as_scd_type=1,
)

@dlt.table(
    name="air_quality_bronze",
    comment="Raw Open-Meteo air quality JSON payloads ingested via Auto Loader.",
)
def air_quality_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(AIR_QUALITY_LANDING_PATH)
        .withColumn("_bronze_ingested_at", F.current_timestamp())
    )


@dlt.view(name="air_quality_silver_staged")
@dlt.expect_all_or_drop({
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_timestamp IS NOT NULL",
    "valid_pm2_5": "pm2_5 IS NULL OR pm2_5 >= 0",
    "valid_pm10": "pm10 IS NULL OR pm10 >= 0",
    "valid_aqi": "european_aqi IS NULL OR european_aqi >= 0",
})
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
            F.col("_metadata.city").alias("city"),
            F.col("_metadata.country").alias("country"),
            F.to_timestamp(F.col("hourly_row.time")).alias("forecast_timestamp"),
            F.to_date(F.col("hourly_row.time")).alias("forecast_date"),
            F.col("hourly_row.pm10").cast("double").alias("pm10"),
            F.col("hourly_row.pm2_5").cast("double").alias("pm2_5"),
            F.col("hourly_row.nitrogen_dioxide").cast("double").alias("nitrogen_dioxide"),
            F.col("hourly_row.european_aqi").cast("double").alias("european_aqi"),
            F.to_timestamp(F.col("_metadata.ingested_at_utc")).alias("ingested_at_utc"),
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


@dlt.table(
    name="weather_gold_daily_summary",
    comment="Napi időjárási összefoglaló városonként",
)
def weather_gold_daily_summary():
    return (
        dlt.read("weather_silver")
        .groupBy("city", "country", "forecast_date")
        .agg(
            F.round(F.avg("temperature_2m"), 2).alias("avg_temp_c"),
            F.round(F.min("temperature_2m"), 2).alias("min_temp_c"),
            F.round(F.max("temperature_2m"), 2).alias("max_temp_c"),
            F.round(F.sum("precipitation"), 2).alias("total_precipitation_mm"),
            F.round(F.avg("relative_humidity_2m"), 1).alias("avg_humidity_pct"),
            F.round(F.max("wind_speed_10m"), 1).alias("max_wind_speed_kmh"),
            F.count("*").alias("hours_count"),
        )
    )