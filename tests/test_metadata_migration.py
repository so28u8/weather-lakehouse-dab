from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPEN_METEO_CLIENT_PATH = REPO_ROOT / "src" / "ingestion" / "open_meteo_client.py"
AIR_QUALITY_CLIENT_PATH = REPO_ROOT / "src" / "ingestion" / "air_quality_client.py"
PIPELINE_ROOT = REPO_ROOT / "src" / "pipelines" / "weather_medallion"
TRANSFORMATIONS_ROOT = PIPELINE_ROOT / "transformations"
PIPELINE_RESOURCE_PATH = REPO_ROOT / "resources" / "weather_pipeline.yml"
LEGACY_PIPELINE_FILE_PATH = REPO_ROOT / "src" / "pipelines" / "weather_medallion.py"


def _load_module_from_path(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_requests_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight requests/urllib3 stubs to isolate HTTP Session tests."""
    urllib3_mod = types.ModuleType("urllib3")
    urllib3_util_mod = types.ModuleType("urllib3.util")
    urllib3_retry_mod = types.ModuleType("urllib3.util.retry")

    class Retry:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    urllib3_retry_mod.Retry = Retry
    urllib3_util_mod.retry = urllib3_retry_mod
    urllib3_mod.util = urllib3_util_mod

    requests_mod = types.ModuleType("requests")
    requests_adapters_mod = types.ModuleType("requests.adapters")

    class HTTPAdapter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Session:
        def __init__(self) -> None:
            self.mounted: dict[str, Any] = {}

        def mount(self, prefix: str, adapter: Any) -> None:
            self.mounted[prefix] = adapter

        def get(self, url: str, params: dict[str, Any] | None = None, timeout: Any = None) -> Any:
            raise AssertionError("Unexpected live HTTP call during unit test")

    requests_adapters_mod.HTTPAdapter = HTTPAdapter
    requests_mod.Session = Session
    requests_mod.adapters = requests_adapters_mod

    monkeypatch.setitem(sys.modules, "urllib3", urllib3_mod)
    monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_mod)
    monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_mod)
    monkeypatch.setitem(sys.modules, "requests", requests_mod)
    monkeypatch.setitem(sys.modules, "requests.adapters", requests_adapters_mod)


# ---------------------------------------------------------------------------
# Lightweight PySpark / DLT AST stubs for isolated pipeline unit tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeColumn:
    kind: str
    value: Any
    args: tuple[FakeColumn, ...] = ()
    alias_name: str | None = None

    def alias(self, name: str) -> FakeColumn:
        return FakeColumn(self.kind, self.value, self.args, alias_name=name)

    def cast(self, target_type: str) -> FakeColumn:
        return FakeColumn("cast", target_type, (self,))


class FakeDataType:
    def __init__(self, type_name: str) -> None:
        self.type_name = type_name


class FakeStructField:
    def __init__(self, name: str, data_type: Any) -> None:
        self.name = name
        self.dataType = data_type


class FakeStructType(FakeDataType):
    def __init__(self, fields: list[FakeStructField] | None = None) -> None:
        super().__init__("struct")
        self.fields = list(fields or [])
        self._by_name = {f.name: f for f in self.fields}

    def fieldNames(self) -> list[str]:
        return [f.name for f in self.fields]

    def __getitem__(self, key: str) -> FakeStructField:
        return self._by_name[key]


class FakeDataFrame:
    def __init__(
        self,
        schema: FakeStructType,
        *,
        source_format: str | None = None,
        options: dict[str, str] | None = None,
        load_path: str | None = None,
        added_columns: dict[str, FakeColumn] | None = None,
        selected_columns: list[FakeColumn | str] | None = None,
    ) -> None:
        self.schema = schema
        self.source_format = source_format
        self.options = dict(options or {})
        self.load_path = load_path
        self.added_columns = dict(added_columns or {})
        self.selected_columns = list(selected_columns or [])

    def withColumn(self, name: str, col: FakeColumn) -> FakeDataFrame:
        updated_added = dict(self.added_columns)
        updated_added[name] = col
        updated_fields = list(self.schema.fields)
        if name not in self.schema.fieldNames():
            updated_fields.append(FakeStructField(name, FakeDataType("derived")))
        return FakeDataFrame(
            FakeStructType(updated_fields),
            source_format=self.source_format,
            options=self.options,
            load_path=self.load_path,
            added_columns=updated_added,
            selected_columns=self.selected_columns,
        )

    def select(self, *cols: FakeColumn | str) -> FakeDataFrame:
        projected_by_name = {
            c.alias_name: c
            for c in self.selected_columns
            if isinstance(c, FakeColumn) and c.alias_name is not None
        }
        resolved = [
            projected_by_name[c] if isinstance(c, str) and c in projected_by_name else c
            for c in cols
        ]
        return FakeDataFrame(
            self.schema,
            source_format=self.source_format,
            options=self.options,
            load_path=self.load_path,
            added_columns=self.added_columns,
            selected_columns=resolved,
        )


class FakeStreamReader:
    def __init__(self, schemas_by_path: dict[str, FakeStructType]) -> None:
        self._schemas_by_path = schemas_by_path
        self._format: str | None = None
        self._options: dict[str, str] = {}
        self.captured_loads: list[tuple[str, str | None, dict[str, str]]] = []

    def format(self, fmt: str) -> FakeStreamReader:
        self._format = fmt
        return self

    def option(self, key: str, value: str) -> FakeStreamReader:
        self._options[key] = value
        return self

    def load(self, path: str) -> FakeDataFrame:
        self.captured_loads.append((path, self._format, dict(self._options)))
        schema = self._schemas_by_path.get(path, FakeStructType([]))
        df = FakeDataFrame(
            schema,
            source_format=self._format,
            options=dict(self._options),
            load_path=path,
        )
        self._format = None
        self._options = {}
        return df


class FakeSpark:
    def __init__(self, conf_values: dict[str, str], schemas_by_path: dict[str, FakeStructType]) -> None:
        self.conf = conf_values
        self.readStream = FakeStreamReader(schemas_by_path)


class FakeDltModule(types.ModuleType):
    def __init__(self, stream_registry: dict[str, FakeDataFrame]) -> None:
        super().__init__("dlt")
        self.registered_datasets: list[tuple[str, str]] = []
        self._stream_registry = stream_registry

    @property
    def dataset_names(self) -> set[str]:
        return {name for _, name in self.registered_datasets}

    def table(self, *args: Any, name: str | None = None, **kwargs: Any):
        def _wrap(fn):
            self.registered_datasets.append(("table", name or fn.__name__))
            return fn
        return _wrap

    def view(self, *args: Any, name: str | None = None, **kwargs: Any):
        def _wrap(fn):
            self.registered_datasets.append(("view", name or fn.__name__))
            return fn
        return _wrap

    def expect_all_or_drop(self, *args: Any, **kwargs: Any):
        def _wrap(fn):
            return fn
        return _wrap

    def create_streaming_table(self, *, name: str, **kwargs: Any) -> None:
        self.registered_datasets.append(("streaming_table", name))

    def apply_changes(self, **kwargs: Any) -> None:
        return None

    def read_stream(self, name: str) -> FakeDataFrame:
        return self._stream_registry[name]

    def read(self, name: str) -> FakeDataFrame:
        return self._stream_registry[name]


@pytest.fixture(autouse=True)
def _clean_utilities_modules():
    """Ensure dynamically imported pipeline `utilities` modules never leak across tests."""
    def _purge() -> None:
        for mod_name in list(sys.modules):
            if mod_name == "utilities" or mod_name.startswith("utilities."):
                sys.modules.pop(mod_name, None)

    _purge()
    try:
        yield
    finally:
        _purge()


def _install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    schemas_by_path: dict[str, FakeStructType] | None = None,
    dlt_streams: dict[str, FakeDataFrame] | None = None,
) -> tuple[FakeSpark, FakeDltModule]:
    conf_values = {
        "landing_volume_path": "/Volumes/weather_dev/raw/landing_weather",
        "air_quality_landing_volume_path": "/Volumes/weather_dev/raw/landing_air_quality",
    }
    fake_spark = FakeSpark(conf_values, schemas_by_path or {})
    fake_dlt = FakeDltModule(dict(dlt_streams or {}))

    pyspark_mod = types.ModuleType("pyspark")
    pyspark_sql_mod = types.ModuleType("pyspark.sql")
    pyspark_sql_fn_mod = types.ModuleType("pyspark.sql.functions")
    pyspark_sql_types_mod = types.ModuleType("pyspark.sql.types")
    pyspark_sql_window_mod = types.ModuleType("pyspark.sql.window")

    pyspark_sql_fn_mod.col = lambda name: FakeColumn("col", name)
    pyspark_sql_fn_mod.lit = lambda val: FakeColumn("lit", val)
    pyspark_sql_fn_mod.current_timestamp = lambda: FakeColumn("fn", "current_timestamp")
    pyspark_sql_fn_mod.to_timestamp = lambda col: FakeColumn("to_timestamp", None, (col,))
    pyspark_sql_fn_mod.to_date = lambda col: FakeColumn("to_date", None, (col,))
    pyspark_sql_fn_mod.coalesce = lambda *cols: FakeColumn("coalesce", None, cols)
    pyspark_sql_fn_mod.arrays_zip = lambda *cols: FakeColumn("arrays_zip", None, cols)
    pyspark_sql_fn_mod.explode = lambda col: FakeColumn("explode", None, (col,))

    pyspark_sql_types_mod.StructType = FakeStructType
    pyspark_sql_types_mod.StructField = FakeStructField
    pyspark_sql_types_mod.StringType = lambda: FakeDataType("string")
    pyspark_sql_types_mod.TimestampType = lambda: FakeDataType("timestamp")

    class FakeWindow:
        @staticmethod
        def partitionBy(*cols: Any) -> FakeWindow:
            return FakeWindow()

    pyspark_sql_window_mod.Window = FakeWindow

    pyspark_sql_mod.Column = FakeColumn
    pyspark_sql_mod.DataFrame = FakeDataFrame
    pyspark_sql_mod.SparkSession = FakeSpark
    pyspark_sql_mod.functions = pyspark_sql_fn_mod
    pyspark_sql_mod.types = pyspark_sql_types_mod
    pyspark_sql_mod.window = pyspark_sql_window_mod
    pyspark_mod.sql = pyspark_sql_mod

    monkeypatch.setitem(sys.modules, "dlt", fake_dlt)
    monkeypatch.setitem(sys.modules, "pyspark", pyspark_mod)
    monkeypatch.setitem(sys.modules, "pyspark.sql", pyspark_sql_mod)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", pyspark_sql_fn_mod)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", pyspark_sql_types_mod)
    monkeypatch.setitem(sys.modules, "pyspark.sql.window", pyspark_sql_window_mod)
    monkeypatch.setattr(builtins, "spark", fake_spark, raising=False)
    # Mirror Databricks SDP `root_path` behavior so `from utilities...` resolves cleanly.
    monkeypatch.syspath_prepend(str(PIPELINE_ROOT))

    return fake_spark, fake_dlt


def _load_transformation(relative_path: str) -> types.ModuleType:
    file_path = TRANSFORMATIONS_ROOT / relative_path
    mod_name = f"sdp_tx_{relative_path.replace('/', '_').removesuffix('.py')}"
    return _load_module_from_path(mod_name, file_path)


def _collect_col_refs(expr: FakeColumn) -> list[str]:
    refs = [expr.value] if expr.kind == "col" else []
    for child in expr.args:
        refs.extend(_collect_col_refs(child))
    return refs


def _ingest_metadata_struct(ingested_type: str = "string") -> FakeStructType:
    return FakeStructType([
        FakeStructField("city", FakeDataType("string")),
        FakeStructField("country", FakeDataType("string")),
        FakeStructField("ingested_at_utc", FakeDataType(ingested_type)),
        FakeStructField("source", FakeDataType("string")),
    ])


def _spark_file_metadata_struct() -> FakeStructType:
    return FakeStructType([
        FakeStructField("file_path", FakeDataType("string")),
        FakeStructField("file_name", FakeDataType("string")),
        FakeStructField("file_size", FakeDataType("bigint")),
        FakeStructField("file_modification_time", FakeDataType("timestamp")),
    ])


# ---------------------------------------------------------------------------
# Ingestion client tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


class _FakeHttpSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: Any = None) -> _FakeResponse:
        return _FakeResponse(self._payload)


def test_open_meteo_client_writes_ingest_metadata_without_legacy_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_requests_stubs(monkeypatch)
    client = _load_module_from_path("open_meteo_client_under_test", OPEN_METEO_CLIENT_PATH)

    sample_api_payload = {"hourly": {"time": ["2026-11-24T00:00"], "temperature_2m": [5.2]}}
    city = {"name": "Budapest", "country": "HU", "lat": 47.4979, "lon": 19.0402}

    result = client.fetch_city_weather(_FakeHttpSession(sample_api_payload), city)

    assert "_metadata" not in result
    assert "ingest_metadata" in result
    meta = result["ingest_metadata"]
    assert meta["city"] == "Budapest"
    assert meta["country"] == "HU"
    assert meta["source"] == "open-meteo-forecast-v1"
    parsed_ts = datetime.fromisoformat(meta["ingested_at_utc"])
    assert parsed_ts.utcoffset() is not None and parsed_ts.utcoffset().total_seconds() == 0

    monkeypatch.setattr(client, "CITIES", [city])
    monkeypatch.setattr(client, "build_session", lambda: _FakeHttpSession(sample_api_payload))
    client.run_ingestion(str(tmp_path))

    written_files = list(tmp_path.glob("ingest_date=*/*.json"))
    assert len(written_files) == 1
    persisted = json.loads(written_files[0].read_text(encoding="utf-8"))
    assert "_metadata" not in persisted
    assert persisted["ingest_metadata"]["city"] == "Budapest"
    assert persisted["ingest_metadata"]["source"] == "open-meteo-forecast-v1"


def test_air_quality_client_writes_ingest_metadata_without_legacy_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_requests_stubs(monkeypatch)
    client = _load_module_from_path("air_quality_client_under_test", AIR_QUALITY_CLIENT_PATH)

    sample_api_payload = {"hourly": {"time": ["2026-11-24T00:00"], "pm2_5": [12.4]}}
    city = {"name": "Vienna", "country": "AT", "lat": 48.2082, "lon": 16.3738}

    result = client.fetch_city_air_quality(_FakeHttpSession(sample_api_payload), city)

    assert "_metadata" not in result
    assert "ingest_metadata" in result
    meta = result["ingest_metadata"]
    assert meta["city"] == "Vienna"
    assert meta["country"] == "AT"
    assert meta["source"] == "open-meteo-air-quality-v1"
    parsed_ts = datetime.fromisoformat(meta["ingested_at_utc"])
    assert parsed_ts.utcoffset() is not None and parsed_ts.utcoffset().total_seconds() == 0

    out_path = client.save_payload(result, tmp_path, city["name"])
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert "_metadata" not in persisted
    assert persisted["ingest_metadata"]["city"] == "Vienna"
    assert persisted["ingest_metadata"]["source"] == "open-meteo-air-quality-v1"


# ---------------------------------------------------------------------------
# Modular Medallion utilities & transformation tests
# ---------------------------------------------------------------------------


def test_bronze_streams_configure_autoloader_hints_and_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_schema = FakeStructType([
        FakeStructField("hourly", FakeDataType("struct")),
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
    ])
    schemas_by_path = {
        "/Volumes/weather_dev/raw/landing_weather": clean_schema,
        "/Volumes/weather_dev/raw/landing_air_quality": clean_schema,
    }
    fake_spark, _ = _install_pipeline_stubs(monkeypatch, schemas_by_path=schemas_by_path)
    metadata_mod = importlib.import_module("utilities.metadata")
    weather_bronze_mod = _load_transformation("bronze/weather_bronze.py")
    aq_bronze_mod = _load_transformation("bronze/air_quality_bronze.py")

    weather_df = weather_bronze_mod.weather_bronze()
    aq_df = aq_bronze_mod.air_quality_bronze()

    assert len(fake_spark.readStream.captured_loads) == 2
    for path, df in (
        ("/Volumes/weather_dev/raw/landing_weather", weather_df),
        ("/Volumes/weather_dev/raw/landing_air_quality", aq_df),
    ):
        assert df.load_path == path
        assert df.source_format == "cloudFiles"
        assert df.options == {
            "cloudFiles.format": "json",
            "cloudFiles.inferColumnTypes": "true",
            "cloudFiles.schemaEvolutionMode": "addNewColumns",
            "cloudFiles.schemaHints": metadata_mod.INGEST_METADATA_SCHEMA_HINT,
        }
        assert "_bronze_ingested_at" in df.added_columns
        assert df.added_columns["_bronze_ingested_at"] == FakeColumn("fn", "current_timestamp")
        assert df.added_columns["_source_file_path"] == FakeColumn("col", "_metadata.file_path")


def test_resolve_source_file_path_handles_clean_shadowed_and_materialized_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")

    clean_df = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
    ]))
    assert metadata_mod.resolve_source_file_path(clean_df) == FakeColumn("col", "_metadata.file_path")

    legacy_shadowed_df = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
        FakeStructField("_metadata", _ingest_metadata_struct()),
    ]))
    assert metadata_mod.resolve_source_file_path(legacy_shadowed_df) == FakeColumn(
        "cast", "string", (FakeColumn("lit", None),)
    )

    materialized_file_meta_df = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
        FakeStructField("_metadata", _spark_file_metadata_struct()),
    ]))
    assert metadata_mod.resolve_source_file_path(materialized_file_meta_df) == FakeColumn(
        "col", "_metadata.file_path"
    )


def test_resolve_metadata_field_clean_schema_reads_only_ingest_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")
    F = sys.modules["pyspark.sql.functions"]

    clean_df = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
    ]))

    city_col = metadata_mod.resolve_metadata_field(clean_df, "city", lambda c: c.cast("string"))
    ts_col = metadata_mod.resolve_metadata_field(clean_df, "ingested_at_utc", F.to_timestamp)

    assert city_col == FakeColumn("cast", "string", (FakeColumn("col", "ingest_metadata.city"),))
    assert ts_col == FakeColumn("to_timestamp", None, (FakeColumn("col", "ingest_metadata.ingested_at_utc"),))
    assert "_metadata.city" not in _collect_col_refs(city_col)
    assert "_metadata.ingested_at_utc" not in _collect_col_refs(ts_col)


def test_resolve_metadata_field_mixed_schema_applies_transform_before_coalesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")
    F = sys.modules["pyspark.sql.functions"]

    mixed_df = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct(ingested_type="string")),
        FakeStructField("_metadata", _ingest_metadata_struct(ingested_type="timestamp")),
    ]))

    city_col = metadata_mod.resolve_metadata_field(mixed_df, "city", lambda c: c.cast("string"))
    ts_col = metadata_mod.resolve_metadata_field(mixed_df, "ingested_at_utc", F.to_timestamp)

    assert city_col == FakeColumn(
        "coalesce",
        None,
        (
            FakeColumn("cast", "string", (FakeColumn("col", "ingest_metadata.city"),)),
            FakeColumn("cast", "string", (FakeColumn("col", "_metadata.city"),)),
        ),
    )
    assert ts_col == FakeColumn(
        "coalesce",
        None,
        (
            FakeColumn("to_timestamp", None, (FakeColumn("col", "ingest_metadata.ingested_at_utc"),)),
            FakeColumn("to_timestamp", None, (FakeColumn("col", "_metadata.ingested_at_utc"),)),
        ),
    )


def test_resolve_metadata_field_legacy_only_schema_reads_legacy_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")
    F = sys.modules["pyspark.sql.functions"]

    legacy_only_df = FakeDataFrame(FakeStructType([
        FakeStructField("_metadata", _ingest_metadata_struct()),
    ]))

    city_col = metadata_mod.resolve_metadata_field(legacy_only_df, "city", lambda c: c.cast("string"))
    ts_col = metadata_mod.resolve_metadata_field(legacy_only_df, "ingested_at_utc", F.to_timestamp)

    assert city_col == FakeColumn("cast", "string", (FakeColumn("col", "_metadata.city"),))
    assert ts_col == FakeColumn("to_timestamp", None, (FakeColumn("col", "_metadata.ingested_at_utc"),))


def test_resolve_metadata_field_ignores_spark_file_metadata_struct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")
    F = sys.modules["pyspark.sql.functions"]

    df_with_spark_metadata = FakeDataFrame(FakeStructType([
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
        FakeStructField("_metadata", _spark_file_metadata_struct()),
    ]))

    city_col = metadata_mod.resolve_metadata_field(
        df_with_spark_metadata, "city", lambda c: c.cast("string")
    )
    ts_col = metadata_mod.resolve_metadata_field(
        df_with_spark_metadata, "ingested_at_utc", F.to_timestamp
    )

    assert city_col == FakeColumn("cast", "string", (FakeColumn("col", "ingest_metadata.city"),))
    assert ts_col == FakeColumn(
        "to_timestamp", None, (FakeColumn("col", "ingest_metadata.ingested_at_utc"),)
    )


def test_resolve_metadata_field_raises_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pipeline_stubs(monkeypatch)
    metadata_mod = importlib.import_module("utilities.metadata")

    empty_df = FakeDataFrame(FakeStructType([
        FakeStructField("hourly", FakeDataType("struct")),
    ]))

    with pytest.raises(ValueError, match="Metadata field 'city' not found"):
        metadata_mod.resolve_metadata_field(empty_df, "city")


@pytest.mark.parametrize("mixed", [False, True])
def test_silver_staged_views_select_expected_columns(
    monkeypatch: pytest.MonkeyPatch, mixed: bool
) -> None:
    fields = [
        FakeStructField("hourly", FakeDataType("struct")),
        FakeStructField("ingest_metadata", _ingest_metadata_struct()),
    ]
    if mixed:
        fields.append(FakeStructField("_metadata", _ingest_metadata_struct(ingested_type="timestamp")))

    bronze_schema = FakeStructType(fields)
    dlt_streams = {
        "weather_bronze": FakeDataFrame(bronze_schema),
        "air_quality_bronze": FakeDataFrame(bronze_schema),
    }
    _install_pipeline_stubs(monkeypatch, dlt_streams=dlt_streams)
    weather_silver_mod = _load_transformation("silver/weather_silver.py")
    aq_silver_mod = _load_transformation("silver/air_quality_silver.py")

    weather_staged = weather_silver_mod.weather_silver_staged()
    aq_staged = aq_silver_mod.air_quality_silver_staged()

    weather_names = [
        col.alias_name if isinstance(col, FakeColumn) and col.alias_name else col
        for col in weather_staged.selected_columns
    ]
    assert weather_names == [
        "city",
        "country",
        "forecast_timestamp",
        "forecast_date",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "ingested_at_utc",
    ]

    aq_names = [
        col.alias_name if isinstance(col, FakeColumn) and col.alias_name else col
        for col in aq_staged.selected_columns
    ]
    assert aq_names == [
        "city",
        "country",
        "forecast_timestamp",
        "forecast_date",
        "pm10",
        "pm2_5",
        "nitrogen_dioxide",
        "european_aqi",
        "ingested_at_utc",
    ]

    for staged in (weather_staged, aq_staged):
        city_expr = next(
            c for c in staged.selected_columns if isinstance(c, FakeColumn) and c.alias_name == "city"
        )
        ts_expr = next(
            c
            for c in staged.selected_columns
            if isinstance(c, FakeColumn) and c.alias_name == "ingested_at_utc"
        )
        if mixed:
            assert _collect_col_refs(city_expr) == ["ingest_metadata.city", "_metadata.city"]
            assert _collect_col_refs(ts_expr) == [
                "ingest_metadata.ingested_at_utc",
                "_metadata.ingested_at_utc",
            ]
        else:
            assert _collect_col_refs(city_expr) == ["ingest_metadata.city"]
            assert _collect_col_refs(ts_expr) == ["ingest_metadata.ingested_at_utc"]


def test_refactored_pipeline_layout_and_dataset_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not LEGACY_PIPELINE_FILE_PATH.exists()
    assert list(TRANSFORMATIONS_ROOT.rglob("__init__.py")) == []

    _, fake_dlt = _install_pipeline_stubs(monkeypatch)
    tx_files = sorted(TRANSFORMATIONS_ROOT.rglob("*.py"))
    assert len(tx_files) == 7

    for idx, tx_file in enumerate(tx_files):
        _load_module_from_path(f"sdp_discover_{idx}_{tx_file.stem}", tx_file)

    expected_datasets = {
        "weather_bronze",
        "air_quality_bronze",
        "weather_silver_staged",
        "weather_silver",
        "air_quality_silver_staged",
        "air_quality_silver",
        "gold_hourly_environmental",
        "gold_daily_city_summary",
        "gold_environmental_alerts",
    }
    assert fake_dlt.dataset_names == expected_datasets
    assert len(fake_dlt.registered_datasets) == len(expected_datasets)

    yaml_text = PIPELINE_RESOURCE_PATH.read_text(encoding="utf-8")
    root_match = re.search(r"^\s*root_path:\s*(\S+)\s*$", yaml_text, flags=re.MULTILINE)
    include_match = re.search(r"^\s*include:\s*(\S+)\s*$", yaml_text, flags=re.MULTILINE)
    assert root_match is not None
    assert include_match is not None

    root_rel = root_match.group(1)
    include_rel = include_match.group(1)
    assert root_rel == "../src/pipelines/weather_medallion"
    assert include_rel == "../src/pipelines/weather_medallion/transformations/**"

    resources_dir = PIPELINE_RESOURCE_PATH.parent
    assert (resources_dir / root_rel).resolve() == PIPELINE_ROOT.resolve()
    assert (resources_dir / include_rel.removesuffix("/**")).resolve() == TRANSFORMATIONS_ROOT.resolve()
