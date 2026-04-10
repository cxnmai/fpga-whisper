"""FPGA-facing helpers for simulation, quantization, and kernel validation."""

from .layout import TileShape
from .quant import FixedPointConfig
from .sim import IverilogSimExecutor
from .transport import (
    DotProductRequest,
    DotProductResponse,
    GeluBlockRequest,
    GeluBlockResponse,
    GemmTileI16Request,
    GemmTileI64Response,
    GemmTileShape,
)

__all__ = [
    "DotProductRequest",
    "DotProductResponse",
    "FixedPointConfig",
    "GeluBlockRequest",
    "GeluBlockResponse",
    "GemmTileI16Request",
    "GemmTileI64Response",
    "GemmTileShape",
    "IverilogSimExecutor",
    "TileShape",
]
