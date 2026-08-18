# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bind the FlyDSL FP8 rowwise-preshuffle kernels to the ``mslk::`` op names.

This is the "Replaces" half of WP-G3 Phase A.

Unlike the original WP-G3 proposal, ``csrc/gemm/gemm_ops.cpp`` **retains** its
CK ``m.impl`` for these two ops.  The Python registration below overrides it on
the architectures FlyDSL supports (gfx942, gfx950); PyTorch emits an
"overriding a previously registered kernel" warning when it does, which is
expected and already the norm for ``mslk::f8f8bf16_rowwise``.  Keeping the CK
binding means the ops are never left unimplemented if this module fails to
import, and it preserves a working backend on architectures FlyDSL does not
cover.
"""

import warnings

import torch

from mslk.gemm.flydsl.rowwise_preshuffle import (
    f8f8bf16_rowwise_preshuffle as _bf16_impl,
    f8f8f16_rowwise_preshuffle as _f16_impl,
)

_registered = False


def _register_rocm_ops() -> None:
    """Register the FlyDSL CUDA impls against the mslk preshuffle op schemas.

    Requires ``torch.ops.mslk`` to already carry the schemas (declared by the
    mslk C++ library's ``m.def``). PyTorch HIP builds surface GPU kernels under
    the "CUDA" dispatch key.
    """
    torch.library.impl(
        "mslk::f8f8bf16_rowwise_preshuffle", "CUDA"
    )(_bf16_impl)
    torch.library.impl(
        "mslk::f8f8f16_rowwise_preshuffle", "CUDA"
    )(_f16_impl)


def is_supported_arch() -> bool:
    """Whether the FlyDSL preshuffle kernel supports the current GPU.

    The kernel targets CDNA3/CDNA4 MFMA (gfx942 = MI300, gfx950 = MI350).  On
    anything else the CK ``m.impl`` in ``gemm_ops.cpp`` remains in effect.
    """
    from mslk.utils.device import is_gfx942, is_gfx950

    return is_gfx942() or is_gfx950()


def register() -> None:
    """Idempotently bind FlyDSL as the CUDA impl of the preshuffle ops.

    No-op on non-HIP builds, on architectures the kernel does not support, and
    when FlyDSL is not installed — in each of those cases the CK ``m.impl``
    stays in effect.
    """
    global _registered
    if _registered or torch.version.hip is None:
        return

    from mslk.flydsl.common import is_flydsl_available

    if not is_flydsl_available() or not is_supported_arch():
        return

    _register_rocm_ops()
    _registered = True


# Register on import — mslk/gemm/__init__.py loads the C++ schema before it
# imports this package, so the op schemas exist by the time we bind.
if torch.version.hip is not None:
    try:
        register()
    except Exception as exc:  # noqa: BLE001
        # Schema not loaded yet (e.g. a stubbed unit-test environment). Defer
        # to an explicit register() call by the caller.
        warnings.warn(
            f"Deferring FlyDSL preshuffle registration until later import: {exc}",
            stacklevel=2,
        )
