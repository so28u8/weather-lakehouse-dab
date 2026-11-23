import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

LANDING_PATH = spark.conf.get("landing_volume_path")
AIR_QUALITY_LANDING_PATH = spark.conf.get("air_quality_landing_volume_path")


@dlt.table(
    name="weather_bronze",
    comment="Raw Open-Meteo weather JSON payloads ingested via Auto Loader.",
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
    comment="Flattened and validated hourly weather data before deduplication.",
)
@dlt.expect_all_or_drop({
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_timestamp IS NOT NULL",
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


@dlt.view(
    name="air_quality_silver_staged",
    comment="Flattened and validated hourly air quality data before deduplication.",
)
@dlt.expect_all_or_drop({
    "valid_city": "city IS NOT NULL",
    "valid_timestamp": "forecast_timestamp IS NOT NULL",
    "valid_pm2_5": "pm2_5 IS NULL OR pm2_5 >= 0",
    "valid_pm10": "pm10 IS NULL OR pm10 >= 0",
    "valid_no2": "nitrogen_dioxide IS NULL OR nitrogen_dioxide >= 0",
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
    name="gold_hourly_environmental",
    comment="Hourly combined weather and air quality metrics with 24h rolling averages and smog risk.",
    cluster_by=["city", "forecast_date"],
)
def gold_hourly_environmental():
    w = dlt.read("weather_silver")
    aq = dlt.read("air_quality_silver")

    joined = (
        w.alias("w")
        .join(aq.alias("aq"), on=["city", "forecast_timestamp"], how="left")
        .select(
            F.col("city"),
            F.col("w.country"),
            F.col("forecast_timestamp"),
            F.col("w.forecast_date"),
            F.col("w.temperature_2m"),
            F.col("w.relative_humidity_2m"),
            F.col("w.precipitation"),
            F.col("w.wind_speed_10m"),
            F.col("aq.pm10"),
            F.col("aq.pm2_5"),
            F.col("aq.nitrogen_dioxide"),
            F.col("aq.european_aqi"),
        )
    )

    window_24h = (
        Window.partitionBy("city")
        .orderBy(F.col("forecast_timestamp").cast("long"))
        .rangeBetween(-23 * 3600, 0)
    )

    return (
        joined
        .withColumn("temp_24h_avg", F.round(F.avg("temperature_2m").over(window_24h), 2))
        .withColumn("pm2_5_24h_avg", F.round(F.avg("pm2_5").over(window_24h), 2))
        .withColumn(
            "aqi_category",
            F.when(F.col("european_aqi").isNull(), "Unknown")
            .when(F.col("european_aqi") < 20, "Good")
            .when(F.col("european_aqi") < 40, "Fair")
            .when(F.col("european_aqi") < 60, "Moderate")
            .when(F.col("european_aqi") < 80, "Poor")
            .when(F.col("european_aqi") < 100, "Very Poor")
            .otherwise("Extremely Poor"),
        )
        .withColumn(
            "is_stagnant_air_risk",
            F.coalesce(
                (F.col("wind_speed_10m") < 5.0) & (F.col("pm2_5") > 25.0),
                F.lit(False),
            ),
        )
    )

@dlt.table(
    name="gold_daily_city_summary",
    comment="Daily executive summary combining weather, air quality, day-over-day temp change, and daily city ranking.",
    cluster_by=["city", "forecast_date"],
)
def gold_daily_city_summary():
    daily = (
        dlt.read("gold_hourly_environmental")
        .groupBy("city", "country", "forecast_date")
        .agg(
            F.round(F.avg("temperature_2m"), 2).alias("avg_temp_c"),
            F.round(F.min("temperature_2m"), 2).alias("min_temp_c"),
            F.round(F.max("temperature_2m"), 2).alias("max_temp_c"),
            F.round(F.sum("precipitation"), 2).alias("total_precipitation_mm"),
            F.round(F.avg("relative_humidity_2m"), 1).alias("avg_humidity_pct"),
            F.round(F.max("wind_speed_10m"), 1).alias("max_wind_speed_kmh"),
            F.round(F.avg("pm2_5"), 2).alias("avg_pm2_5"),
            F.round(F.max("pm2_5"), 2).alias("max_pm2_5"),
            F.round(F.avg("pm10"), 2).alias("avg_pm10"),
            F.round(F.avg("nitrogen_dioxide"), 2).alias("avg_no2"),
            F.round(F.avg("european_aqi"), 1).alias("avg_european_aqi"),
            F.round(F.max("european_aqi"), 1).alias("max_european_aqi"),
            F.sum(F.col("is_stagnant_air_risk").cast("int")).alias("stagnant_air_hours"),
            F.count("*").alias("hours_count"),
        )
    )

    city_time_win = Window.partitionBy("city").orderBy("forecast_date")
    daily_rank_win = Window.partitionBy("forecast_date").orderBy(
        F.col("avg_european_aqi").asc_nulls_last()
    )

    return (
        daily
        .withColumn("_prev_date", F.lag("forecast_date").over(city_time_win))
        .withColumn("_prev_temp", F.lag("avg_temp_c").over(city_time_win))
        .withColumn(
            "temp_diff_vs_prev_day_c",
            F.when(
                F.datediff("forecast_date", "_prev_date") == 1,
                F.round(F.col("avg_temp_c") - F.col("_prev_temp"), 2),
            ),
        )
        .withColumn(
            "daily_air_quality_rank",
            F.when(
                F.col("avg_european_aqi").isNotNull(),
                F.dense_rank().over(daily_rank_win),
            ),
        )
        .drop("_prev_date", "_prev_temp")
    )

@dlt.table(
    name="gold_environmental_alerts",
    comment="Operational alerts for extreme weather and poor air quality events.",
    cluster_by=["city", "forecast_date"],
)
def gold_environmental_alerts():
    df = dlt.read("gold_hourly_environmental")

    alerts = df.withColumn(
        "alert_reasons",
        F.array_compact(
            F.array(
                F.when(F.col("european_aqi") >= 60, F.lit("POOR_AIR_QUALITY")),
                F.when(F.col("pm2_5") > 25.0, F.lit("HIGH_PM2_5")),
                F.when(F.col("is_stagnant_air_risk"), F.lit("STAGNANT_AIR_SMOG_RISK")),
                F.when(F.col("wind_speed_10m") >= 50.0, F.lit("HIGH_WIND")),
                F.when(F.col("precipitation") >= 10.0, F.lit("HEAVY_PRECIPITATION")),
                F.when(
                    (F.col("temperature_2m") <= -10.0) | (F.col("temperature_2m") >= 35.0),
                    F.lit("EXTREME_TEMPERATURE"),
                ),
            )
        ),
    ).filter(F.size("alert_reasons") > 0)

    return alerts.withColumn(
        "alert_severity",
        F.when(
            (F.col("european_aqi") >= 80)
            | (F.col("wind_speed_10m") >= 70.0)
            | (F.col("temperature_2m") <= -15.0)
            | (F.col("temperature_2m") >= 38.0),
            F.lit("CRITICAL"),
        ).otherwise(F.lit("WARNING")),
    ).select(
        "city",
        "country",
        "forecast_timestamp",
        "forecast_date",
        "alert_severity",
        "alert_reasons",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "pm2_5",
        "european_aqi",
        "aqi_category",
    )