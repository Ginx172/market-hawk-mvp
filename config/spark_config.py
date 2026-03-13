"""
Spark configuration for Market Hawk MVP.
Optimized for Intel i7-9700F (8 cores), 64GB DDR4.
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("market_hawk.spark.config")


@dataclass
class SparkConfig:
    """Spark session configuration optimized for local hardware."""

    master: str = "local[8]"
    app_name: str = "MarketHawk"
    driver_memory: str = "16g"
    executor_memory: str = "8g"
    shuffle_partitions: int = 8
    sql_shuffle_partitions: int = 8
    parquet_output_dir: str = "data/parquet"
    log_level: str = "WARN"


# Default config instance
SPARK_CONFIG = SparkConfig()

_spark_session = None


def get_or_create_spark_session(config: Optional[SparkConfig] = None) -> "pyspark.sql.SparkSession":  # type: ignore[name-defined]
    """Create or reuse a SparkSession with hardware-optimised settings.

    Args:
        config: Optional SparkConfig. Uses SPARK_CONFIG default if None.

    Returns:
        Active SparkSession instance.
    """
    global _spark_session
    if _spark_session is not None:
        try:
            # Test if session is still alive
            _spark_session.sparkContext.getConf().get("spark.app.name")
            return _spark_session
        except Exception:
            _spark_session = None

    cfg = config or SPARK_CONFIG

    from pyspark.sql import SparkSession  # type: ignore[import]

    builder = (
        SparkSession.builder.master(cfg.master)
        .appName(cfg.app_name)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.sql_shuffle_partitions))
        .config("spark.default.parallelism", str(cfg.shuffle_partitions))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
    )

    _spark_session = builder.getOrCreate()
    _spark_session.sparkContext.setLogLevel(cfg.log_level)

    logger.info(
        "SparkSession created: master=%s, driver_memory=%s, partitions=%d",
        cfg.master,
        cfg.driver_memory,
        cfg.shuffle_partitions,
    )
    return _spark_session


def stop_spark() -> None:
    """Stop the active SparkSession and release resources."""
    global _spark_session
    if _spark_session is not None:
        try:
            _spark_session.stop()
            logger.info("SparkSession stopped")
        except Exception:
            pass
        _spark_session = None
