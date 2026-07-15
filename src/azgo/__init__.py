"""AlphaZero Go research package."""

from azgo._version import __version__
from azgo.config import AppConfig, load_config
from azgo.sgf import (
    SgfError,
    SgfGameRecord,
    load_sgf_collection,
    parse_sgf_collection,
    save_sgf_collection,
    serialize_sgf_collection,
)

__all__ = [
    "AppConfig",
    "SgfError",
    "SgfGameRecord",
    "__version__",
    "load_config",
    "load_sgf_collection",
    "parse_sgf_collection",
    "save_sgf_collection",
    "serialize_sgf_collection",
]

