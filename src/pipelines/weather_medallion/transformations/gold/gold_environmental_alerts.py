import dlt
from pyspark.sql import functions as F

ALERT_AQI_MIN = 60
ALERT_PM2_5_MIN = 25.0
ALERT_WIND_MIN_KMH = 50.0
ALERT_PRECIP_MIN_MM = 10.0
ALERT_TEMP_LOW_C = -10.0
ALERT_TEMP_HIGH_C = 35.0

CRITICAL_AQI_MIN = 80
CRITICAL_WIND_MIN_KMH = 70.0
CRITICAL_TEMP_LOW_C = -15.0
CRITICAL_TEMP_HIGH_C = 38.0


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
                F.when(F.col("european_aqi") >= ALERT_AQI_MIN, F.lit("POOR_AIR_QUALITY")),
                F.when(F.col("pm2_5") > ALERT_PM2_5_MIN, F.lit("HIGH_PM2_5")),
                F.when(F.col("is_stagnant_air_risk"), F.lit("STAGNANT_AIR_SMOG_RISK")),
                F.when(F.col("wind_speed_10m") >= ALERT_WIND_MIN_KMH, F.lit("HIGH_WIND")),
                F.when(F.col("precipitation") >= ALERT_PRECIP_MIN_MM, F.lit("HEAVY_PRECIPITATION")),
                F.when(
                    (F.col("temperature_2m") <= ALERT_TEMP_LOW_C)
                    | (F.col("temperature_2m") >= ALERT_TEMP_HIGH_C),
                    F.lit("EXTREME_TEMPERATURE"),
                ),
            )
        ),
    ).filter(F.size("alert_reasons") > 0)

    return alerts.withColumn(
        "alert_severity",
        F.when(
            (F.col("european_aqi") >= CRITICAL_AQI_MIN)
            | (F.col("wind_speed_10m") >= CRITICAL_WIND_MIN_KMH)
            | (F.col("temperature_2m") <= CRITICAL_TEMP_LOW_C)
            | (F.col("temperature_2m") >= CRITICAL_TEMP_HIGH_C),
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
