import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window


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
