"""Dependency-free Cloud-Optimized GeoTIFF (COG) window reader + UTM projection.

Reads only the internal tiles that intersect a requested window, using HTTP range
requests, so a 64x64 chip from a 110 km Sentinel-2 granule costs a few hundred kB
instead of the whole file. Supports classic TIFF and BigTIFF, little-endian,
uncompressed or DEFLATE, with or without horizontal predictor (the layout used by
the AWS ``sentinel-cogs`` archive). Compressed tiles are cached on disk so later
runs work offline.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import threading
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

USER_AGENT = "Argus-wildfire-research/1.0"
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 16: "Q", 6: "b", 8: "h", 9: "i", 17: "q", 11: "f", 12: "d"}
_DTYPES = {(1, 8): np.uint8, (1, 16): np.uint16, (1, 32): np.uint32, (2, 8): np.int8, (2, 16): np.int16,
           (2, 32): np.int32, (3, 32): np.float32, (3, 64): np.float64}


def http_range(url: str, start: int, length: int, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


@dataclass
class TiffLevel:
    width: int
    height: int
    tile_w: int
    tile_h: int
    offsets: list[int]
    counts: list[int]
    compression: int
    predictor: int
    dtype: type
    samples: int


class COGReader:
    """Window reader for one remote (or cached) COG."""

    def __init__(self, url: str, cache_dir: str | Path | None = None, header_bytes: int = 65536) -> None:
        self.url = url
        self.cache = Path(cache_dir) / hashlib.sha1(url.encode()).hexdigest()[:16] if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.bytes_downloaded = 0
        self._header = self._cached("header", lambda: http_range(url, 0, header_bytes))
        self.levels = self._parse()

    # -- byte access --------------------------------------------------------------
    def _cached(self, key: str, fetch: object) -> bytes:
        path = self.cache / f"{key}.bin" if self.cache else None
        if path is not None and path.exists():
            return path.read_bytes()
        data = fetch()  # type: ignore[operator]
        self.bytes_downloaded += len(data)
        if path is not None:  # atomic write: concurrent readers never see a partial file
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            try:
                os.replace(tmp, path)
            except OSError:
                tmp.unlink(missing_ok=True)
        return data

    def _read(self, offset: int, length: int) -> bytes:
        if offset + length <= len(self._header):
            return self._header[offset:offset + length]
        return self._cached(f"r{offset}_{length}", lambda: http_range(self.url, offset, length))

    # -- TIFF structure ----------------------------------------------------------------
    def _parse(self) -> list[TiffLevel]:
        h = self._header
        if h[:2] != b"II":
            raise ValueError("only little-endian TIFF is supported")
        magic = struct.unpack_from("<H", h, 2)[0]
        big = magic == 43
        if magic not in (42, 43):
            raise ValueError(f"not a TIFF file (magic {magic})")
        ifd = struct.unpack_from("<Q", h, 8)[0] if big else struct.unpack_from("<I", h, 4)[0]
        levels: list[TiffLevel] = []
        while ifd and len(levels) < 16:
            tags, ifd = self._read_ifd(ifd, big)
            if 322 not in tags:  # skip non-tiled IFDs (masks etc.)
                continue
            bits = int(tags.get(258, [16])[0])
            fmt = int(tags.get(339, [1])[0])
            levels.append(TiffLevel(
                width=int(tags[256][0]), height=int(tags[257][0]), tile_w=int(tags[322][0]),
                tile_h=int(tags[323][0]), offsets=[int(v) for v in tags[324]], counts=[int(v) for v in tags[325]],
                compression=int(tags.get(259, [1])[0]), predictor=int(tags.get(317, [1])[0]),
                dtype=_DTYPES[(fmt, bits)], samples=int(tags.get(277, [1])[0]),
            ))
        if not levels:
            raise ValueError("no tiled image found in TIFF")
        return levels

    def _read_ifd(self, offset: int, big: bool) -> tuple[dict[int, list[float]], int]:
        cnt_fmt, cnt_size, entry_size, off_fmt, inline = ("<Q", 8, 20, "<Q", 8) if big else ("<H", 2, 12, "<I", 4)
        n = struct.unpack(cnt_fmt, self._read(offset, cnt_size))[0]
        raw = self._read(offset + cnt_size, n * entry_size + inline)
        tags: dict[int, list[float]] = {}
        for i in range(n):
            e = raw[i * entry_size:(i + 1) * entry_size]
            tag, typ = struct.unpack_from("<HH", e, 0)
            count = struct.unpack_from("<Q" if big else "<I", e, 4)[0]
            size = _TYPE_SIZES.get(typ, 1) * count
            value_bytes = e[(12 if big else 8):] if size <= inline else \
                self._read(struct.unpack_from(off_fmt, e, 12 if big else 8)[0], size)
            if typ in _TYPE_FMT:
                tags[tag] = list(struct.unpack_from(f"<{count}{_TYPE_FMT[typ]}", value_bytes, 0))
        next_ifd = struct.unpack(off_fmt, raw[n * entry_size:n * entry_size + inline])[0]
        return tags, next_ifd

    # -- pixel access --------------------------------------------------------------
    def _tile(self, level: TiffLevel, index: int) -> np.ndarray:
        off, cnt = level.offsets[index], level.counts[index]
        shape = (level.tile_h, level.tile_w)
        if cnt == 0:
            return np.zeros(shape, dtype=level.dtype)
        data = self._cached(f"t{off}_{cnt}", lambda: http_range(self.url, off, cnt))
        if level.compression in (8, 32946):
            data = zlib.decompress(data)
        elif level.compression != 1:
            raise ValueError(f"unsupported TIFF compression {level.compression}")
        arr = np.frombuffer(data, dtype=np.dtype(level.dtype).newbyteorder("<"))[: shape[0] * shape[1]]
        arr = arr.reshape(shape).copy()
        if level.predictor == 2:
            arr = np.cumsum(arr, axis=1, dtype=arr.dtype)
        return arr

    def read_window(self, row0: int, col0: int, height: int, width: int, level_index: int = 0) -> np.ndarray:
        """Read [row0:row0+height, col0:col0+width]; pixels outside the image are 0 (nodata)."""
        lv = self.levels[level_index]
        out = np.zeros((height, width), dtype=lv.dtype)
        tiles_across = math.ceil(lv.width / lv.tile_w)
        r_lo, r_hi = max(0, row0), min(lv.height, row0 + height)
        c_lo, c_hi = max(0, col0), min(lv.width, col0 + width)
        if r_lo >= r_hi or c_lo >= c_hi:
            return out
        for ty in range(r_lo // lv.tile_h, (r_hi - 1) // lv.tile_h + 1):
            for tx in range(c_lo // lv.tile_w, (c_hi - 1) // lv.tile_w + 1):
                tile = self._tile(lv, ty * tiles_across + tx)
                y0, x0 = ty * lv.tile_h, tx * lv.tile_w
                ya, yb = max(r_lo, y0), min(r_hi, y0 + lv.tile_h)
                xa, xb = max(c_lo, x0), min(c_hi, x0 + lv.tile_w)
                out[ya - row0:yb - row0, xa - col0:xb - col0] = tile[ya - y0:yb - y0, xa - x0:xb - x0]
        return out


def latlon_to_utm(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    """WGS84 geographic -> UTM easting/northing (EPSG 326xx north, 327xx south), Snyder's series."""
    zone, south = epsg % 100, epsg // 100 == 327
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = ep2 * math.cos(phi) ** 2
    aa = math.cos(phi) * (lam - lam0)
    m = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
             + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
             - (35 * e2**3 / 3072) * math.sin(6 * phi))
    x = k0 * n * (aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * aa**5 / 120) + 500000.0
    y = k0 * (m + n * math.tan(phi) * (aa**2 / 2 + (5 - t + 9 * c + 4 * c * c) * aa**4 / 24
                                       + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * aa**6 / 720))
    return x, y + (10_000_000.0 if south else 0.0)
