import dlt
from pyspark.sql import Column, functions as F
from pyspark.sql.window import Window

ROLLING_24H_WINDOW_SECONDS = 23 * 3600
STAGNANT_WIND_MAX_KMH = 5.0
STAGNANT_PM2_5_MIN = 25.0


def _classify_aqi(aqi_col: Column) -> Column:
    """Map numeric European AQI values to standard qualitative bands."""
    return (
        F.when(aqi_col.isNull(), "Unknown")
        .when(aqi_col < 20, "Good")
        .when(aqi_col < 40, "Fair")
        .when(aqi_col < 60, "Moderate")
        .when(aqi_col < 80, "Poor")
        .when(aqi_col < 100, "Very Poor")
        .otherwise("Extremely Poor")
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
        .rangeBetween(-ROLLING_24H_WINDOW_SECONDS, 0)
    )

    return (
        joined
        .withColumn("temp_24h_avg", F.round(F.avg("temperature_2m").over(window_24h), 2))
        .withColumn("pm2_5_24h_avg", F.round(F.avg("pm2_5").over(window_24h), 2))
        .withColumn("aqi_category", _classify_aqi(F.col("european_aqi")))
        .withColumn(
            "is_stagnant_air_risk",
            F.coalesce(
                (F.col("wind_speed_10m") < STAGNANT_WIND_MAX_KMH)
                & (F.col("pm2_5") > STAGNANT_PM2_5_MIN),
                F.lit(False),
            ),
        )
    )
