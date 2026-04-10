from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


class Ct2DataType(Enum):
    FLOAT32 = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    FLOAT16 = 4
    BFLOAT16 = 5

    @classmethod
    def from_id(cls, dtype_id: int) -> "Ct2DataType":
        try:
            return cls(dtype_id)
        except ValueError as exc:
            raise ValueError(f"unsupported CTranslate2 dtype id {dtype_id}") from exc

    @property
    def label(self) -> str:
        return {
            Ct2DataType.FLOAT32: "float32",
            Ct2DataType.INT8: "int8",
            Ct2DataType.INT16: "int16",
            Ct2DataType.INT32: "int32",
            Ct2DataType.FLOAT16: "float16",
            Ct2DataType.BFLOAT16: "bfloat16",
        }[self]


@dataclass(slots=True, frozen=True)
class TensorInfo:
    name: str
    shape: list[int]
    dtype: Ct2DataType
    offset: int
    nbytes: int

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass(slots=True, frozen=True)
class TensorDataF32:
    info: TensorInfo
    values: list[float]


class Ct2ModelBin:
    def __init__(
        self,
        path: Path,
        *,
        version: int,
        spec_name: str,
        revision: int,
        tensors: dict[str, TensorInfo],
    ) -> None:
        self.path = path
        self.version = version
        self.spec_name = spec_name
        self.revision = revision
        self._tensors = tensors

    @classmethod
    def open(cls, path: str | Path) -> "Ct2ModelBin":
        path = Path(path)
        with path.open("rb") as handle:
            version = _read_u32(handle)
            spec_name = _read_string(handle)
            revision = _read_u32(handle)
            tensor_count = _read_u32(handle)
            tensors: dict[str, TensorInfo] = {}

            for _ in range(tensor_count):
                name = _read_string(handle)
                rank = _read_u8(handle)
                shape = [_read_u32(handle) for _ in range(rank)]
                dtype = Ct2DataType.from_id(_read_u8(handle))
                nbytes = _read_u32(handle)
                offset = handle.tell()
                handle.seek(nbytes, io.SEEK_CUR)

                tensors[name] = TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    offset=offset,
                    nbytes=nbytes,
                )

            alias_count = _read_u32(handle)
            for _ in range(alias_count):
                _read_string(handle)
                _read_string(handle)

        return cls(
            path=path,
            version=version,
            spec_name=spec_name,
            revision=revision,
            tensors=tensors,
        )

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in model.bin: {name}") from exc

    def read_tensor_f32(self, name: str) -> TensorDataF32:
        info = self.tensor(name)
        with self.path.open("rb") as handle:
            handle.seek(info.offset)
            raw = handle.read(info.nbytes)

        if len(raw) != info.nbytes:
            raise ValueError(
                f"failed to read tensor {name}: expected {info.nbytes} bytes, got {len(raw)}"
            )

        if info.dtype is Ct2DataType.FLOAT32:
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
            result = values.tolist()
        elif info.dtype is Ct2DataType.FLOAT16:
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            result = values.tolist()
        else:
            raise ValueError(
                f"tensor {name} has dtype {info.dtype.label}, not a float tensor"
            )

        if len(result) != info.element_count:
            raise ValueError(
                f"tensor {info.name} element count mismatch: "
                f"expected {info.element_count}, got {len(result)}"
            )

        return TensorDataF32(info=info, values=result)

    def tensor_names(self) -> Iterator[str]:
        return iter(self._tensors.keys())


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"unexpected EOF while reading {size} bytes")
    return data


def _read_u8(handle: BinaryIO) -> int:
    return struct.unpack("<B", _read_exact(handle, 1))[0]


def _read_u16(handle: BinaryIO) -> int:
    return struct.unpack("<H", _read_exact(handle, 2))[0]


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_string(handle: BinaryIO) -> str:
    byte_len = _read_u16(handle)
    if byte_len == 0:
        raise ValueError("invalid zero-length string in model.bin")

    raw = _read_exact(handle, byte_len)
    if raw[-1] != 0:
        raise ValueError("model.bin string is missing trailing NUL")

    try:
        return raw[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("model.bin contained invalid UTF-8") from exc
