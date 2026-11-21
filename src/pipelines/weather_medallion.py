import dlt
from pyspark.sql import functions as F

LANDING_PATH = spark.conf.get("landing_volume_path")


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

@dlt.table(
    name="weather_silver",
    comment="Tisztított, óránkénti bontású időjárási előrejelzések városonként",
)
@dlt.expect_or_drop("valid_city", "city IS NOT NULL")
@dlt.expect_or_drop("valid_temperature", "temperature_c BETWEEN -60 AND 60")
def weather_silver():
    bronze_df = dlt.read_stream("weather_bronze")

    exploded = bronze_df.select(
        F.col("_metadata.city").alias("city"),
        F.col("_metadata.country").alias("country"),
        F.col("latitude"),
        F.col("longitude"),
        F.to_timestamp(F.col("_metadata.ingested_at_utc")).alias("ingested_at_utc"),
        F.explode(
            F.arrays_zip(
                F.col("hourly.time").alias("forecast_time"),
                F.col("hourly.temperature_2m").alias("temperature_c"),
                F.col("hourly.relative_humidity_2m").alias("humidity_pct"),
                F.col("hourly.precipitation").alias("precipitation_mm"),
                F.col("hourly.wind_speed_10m").alias("wind_speed_kmh"),
            )
        ).alias("h"),
    )

    return exploded.select(
        "city",
        "country",
        "latitude",
        "longitude",
        F.to_timestamp(F.col("h.forecast_time")).alias("forecast_time_utc"),
        F.to_date(F.col("h.forecast_time")).alias("forecast_date"),
        F.col("h.temperature_c").cast("double").alias("temperature_c"),
        F.col("h.humidity_pct").cast("double").alias("humidity_pct"),
        F.col("h.precipitation_mm").cast("double").alias("precipitation_mm"),
        F.col("h.wind_speed_kmh").cast("double").alias("wind_speed_kmh"),
        "ingested_at_utc",
    )


@dlt.table(
    name="weather_gold_daily_summary",
    comment="Napi időjárási összesítő tábla városonként",
)
def weather_gold_daily_summary():
    return (
        dlt.read("weather_silver")
        .groupBy("city", "country", "forecast_date")
        .agg(
            F.round(F.avg("temperature_c"), 2).alias("avg_temp_c"),
            F.min("temperature_c").alias("min_temp_c"),
            F.max("temperature_c").alias("max_temp_c"),
            F.round(F.sum("precipitation_mm"), 2).alias("total_precipitation_mm"),
            F.max("wind_speed_kmh").alias("max_wind_speed_kmh"),
            F.count("*").alias("hourly_records"),
        )
    )