"""Fixed-point arithmetic helpers for porting float ops to FPGA.

The Tier-1 evaluator's Mahalanobis cost is currently float64; for FPGA
synthesis we quantise to Q-format fixed-point (Qm.n, where m is integer
bits and n fractional bits, total width m+n+1 for signed).

This module provides:
    - `FixedPointConfig`         choose (total_bits, frac_bits, signed)
    - `quantize_scalar`          float -> fixed integer
    - `dequantize_scalar`        fixed integer -> float
    - `quantize_array`           numpy/list scalar quantisation
    - `cpp_typedef`              emit Vitis HLS `ap_fixed<W,I>` typedef
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPointConfig:
    """Q-format fixed point.

    total_bits: word width
    int_bits:   integer-part bits (including sign for signed)
    signed:     two's-complement when True
    """

    total_bits: int = 16
    int_bits: int = 4
    signed: bool = True

    @property
    def frac_bits(self) -> int:
        return self.total_bits - self.int_bits

    @property
    def cpp_type(self) -> str:
        # Vitis HLS ap_fixed<W,I> -- W total bits, I integer bits (signed
        # implied by `ap_fixed` vs `ap_ufixed`).
        prefix = "ap_fixed" if self.signed else "ap_ufixed"
        return f"{prefix}<{self.total_bits}, {self.int_bits}>"

    def value_range(self) -> tuple[float, float]:
        scale = 1.0 / (1 << self.frac_bits)
        if self.signed:
            max_int = (1 << (self.total_bits - 1)) - 1
            min_int = -(1 << (self.total_bits - 1))
        else:
            max_int = (1 << self.total_bits) - 1
            min_int = 0
        return (min_int * scale, max_int * scale)


def _saturating_clip(x: int, cfg: FixedPointConfig) -> int:
    if cfg.signed:
        max_int = (1 << (cfg.total_bits - 1)) - 1
        min_int = -(1 << (cfg.total_bits - 1))
    else:
        max_int = (1 << cfg.total_bits) - 1
        min_int = 0
    return max(min_int, min(max_int, x))


def quantize_scalar(value: float, cfg: FixedPointConfig) -> int:
    """Convert a float to its fixed-point integer representation."""
    scaled = round(value * (1 << cfg.frac_bits))
    return _saturating_clip(scaled, cfg)


def dequantize_scalar(value_int: int, cfg: FixedPointConfig) -> float:
    return value_int / (1 << cfg.frac_bits)


def quantize_array(values: list[float], cfg: FixedPointConfig) -> list[int]:
    return [quantize_scalar(v, cfg) for v in values]


def cpp_typedef(name: str, cfg: FixedPointConfig) -> str:
    return f"typedef {cfg.cpp_type} {name};"
