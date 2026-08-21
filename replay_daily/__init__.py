"""Canonical daily replay record extraction and storage for Kaggriculture 1.32.7.

Logical records come from `extractor.extract_replay` (D-018). Physical storage
defaults to Zstandard-compressed Parquet (`storage.write_parquet` /
`storage.read_parquet`); JSONL remains optional debug/inspection output.
"""

from .constants import ENGINE_VERSION, SCHEMA_VERSION
from .extractor import VersionMismatch, extract_replay, load_manifest
from .storage import read_parquet, records_to_table, write_parquet

__all__ = [
    "ENGINE_VERSION",
    "SCHEMA_VERSION",
    "VersionMismatch",
    "extract_replay",
    "load_manifest",
    "read_parquet",
    "records_to_table",
    "write_parquet",
]
