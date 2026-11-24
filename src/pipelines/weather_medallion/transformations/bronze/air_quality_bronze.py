import dlt

from utilities.autoloader import read_raw_json_stream

AIR_QUALITY_LANDING_PATH = spark.conf.get("air_quality_landing_volume_path")


@dlt.table(
    name="air_quality_bronze",
    comment="Raw Open-Meteo air quality JSON payloads ingested via Auto Loader.",
)
def air_quality_bronze():
    return read_raw_json_stream(spark, AIR_QUALITY_LANDING_PATH)
