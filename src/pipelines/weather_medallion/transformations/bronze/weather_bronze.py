import dlt

from utilities.autoloader import read_raw_json_stream

LANDING_PATH = spark.conf.get("landing_volume_path")


@dlt.table(
    name="weather_bronze",
    comment="Raw Open-Meteo weather JSON payloads ingested via Auto Loader.",
)
def weather_bronze():
    return read_raw_json_stream(spark, LANDING_PATH)
