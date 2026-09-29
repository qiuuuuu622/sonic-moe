"""Hopper QuACK grouped FP8 WGMMA with SGL 1x128 / 128x128 FP32 scales."""

import _bootstrap  # noqa: F401; select local experiment dependencies

from functools import lru_cache
import os

SCALE_PRODUCT = bool(int(os.environ.get("SCALE_PRODUCT", "1")))
AB_STAGE_CAP = int(os.environ.get("AB_STAGE_CAP", "0"))
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from quack.compile_utils import make_fake_tensor
from quack.gemm_default_epi import GemmDefaultSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    make_fake_gemm_tensors,
    make_fake_varlen_args,
)
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments
from quack.cute_dsl_utils import torch2cute_dtype_map


class GemmFP8Sm90(GemmDefaultSm90):
    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        weight = epi_loop_tensors.get("mColVecBroadcast")
        if const_expr(weight is not None):
            # SGL contract: multiply FP32 down accumulator before BF16 store.
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * weight[i]
        return ()


@lru_cache(None)
def compile_fp8(gather, outdtype, tile, pingpong, weighted):
    f8 = torch2cute_dtype_map[torch.float8_e4m3fn]

    def fake(dtype, shape, align=1):
        return make_fake_tensor(
            dtype, shape, divisibility=align, leading_dim=len(shape) - 1
        )

    A, W, O, _, m, n, k, l = make_fake_gemm_tensors(
        f8,
        f8,
        torch2cute_dtype_map[outdtype],
        None,
        "k",
        "k",
        "n",
        "n",
        varlen_m=True,
        gather_A=gather,
    )
    SA = fake(Float32, (cute.sym_int(), cute.sym_int()))
    SB = fake(Float32, (cute.sym_int(), cute.sym_int(), cute.sym_int()))
    epi = GemmDefaultSm90.EpilogueArguments(
        mColVecBroadcast=fake(Float32, (m,), 4) if weighted else None
    )
    sched = TileSchedulerOptions(Int32(1))
    varlen = make_fake_varlen_args(True, False, gather, m)

    def setup(obj):
        obj.sgl_fp8_scales = True
        obj.sgl_scale_product = SCALE_PRODUCT
        obj.sgl_ab_stage_cap = AB_STAGE_CAP

    return compile_gemm_kernel(
        GemmFP8Sm90,
        f8,
        (*tile, 128),
        (1, 1, 1),
        pingpong,
        True,
        gather,
        False,
        (9, 0),
        A,
        W,
        O,
        None,
        epi,
        sched,
        varlen,
        post_init=setup,
        mSFA=SA,
        mSFB=SB,
    )


@torch.no_grad()
def grouped_fp8(
    A, W, SA, SB, cu, idx=None, out=None, tile=(64, 128), pingpong=False, weights=None
):
    assert A.dtype == W.dtype == torch.float8_e4m3fn
    assert SA.dtype == SB.dtype == torch.float32
    assert cu.dtype == torch.int32
    if out is None:
        out = torch.empty(
            (idx.numel() if idx is not None else A.shape[0], W.shape[1]),
            device=A.device,
            dtype=torch.bfloat16,
        )
    fn = compile_fp8(idx is not None, out.dtype, tile, pingpong, weights is not None)
    epi = GemmDefaultSm90.EpilogueArguments(
        mColVecBroadcast=weights, add_to_output=None, rounding_mode=None
    )
    sched = TileSchedulerOptions(
        torch.cuda.get_device_properties(A.device).multi_processor_count,
        raster_order=None,
        max_swizzle_size=8,
    )
    fn(A, W, out, None, epi, sched, VarlenArguments(mCuSeqlensM=cu, mAIdx=idx), SA, SB)
    return out


from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import unpack
from quack.activation import swiglu
from quack.rounding import RoundingMode


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def rounded_swiglu(acc):
    g, u = unpack(acc)
    g = g.to(cutlass.BFloat16).to(Float32)
    u = u.to(cutlass.BFloat16).to(Float32)
    return {"postact": swiglu(g, u)}


@lru_cache(None)
def compile_gate(tile, pingpong, gather=True):
    f8 = torch2cute_dtype_map[torch.float8_e4m3fn]
    cls = rounded_swiglu._mint((), 9, True, False, (), RoundingMode.RN, (), False)
    A, W, _, _, m, n, k, l = make_fake_gemm_tensors(
        f8, f8, None, None, "k", "k", "n", "n", varlen_m=True, gather_A=gather
    )

    def fake(dtype, shape, align=1):
        return make_fake_tensor(
            dtype, shape, divisibility=align, leading_dim=len(shape) - 1
        )

    O = fake(cutlass.BFloat16, (m, cute.sym_int()), 8)
    SA = fake(Float32, (cute.sym_int(), cute.sym_int()))
    SB = fake(Float32, (cute.sym_int(), cute.sym_int(), cute.sym_int()))
    epi = cls.EpilogueArguments(postact=O)
    sched = TileSchedulerOptions(Int32(1))
    varlen = make_fake_varlen_args(True, False, gather, m)

    def setup(obj):
        obj.sgl_fp8_scales = True
        obj.sgl_gate_layout = True
        obj.sgl_scale_product = SCALE_PRODUCT
        obj.sgl_ab_stage_cap = AB_STAGE_CAP
        obj.implicit_dtype = cutlass.BFloat16

    fn = compile_gemm_kernel(
        cls,
        f8,
        (*tile, 128),
        (1, 1, 1),
        pingpong,
        True,
        gather,
        False,
        (9, 0),
        A,
        W,
        None,
        None,
        epi,
        sched,
        varlen,
        post_init=setup,
        mSFA=SA,
        mSFB=SB,
        concat_layout=("B",),
    )
    return fn, cls


@torch.no_grad()
def grouped_gate_fp8(A, W, SA, SB, cu, idx=None, tile=(64, 128), pingpong=True):
    out = torch.empty(
        (idx.numel() if idx is not None else A.shape[0], W.shape[1] // 2),
        device=A.device,
        dtype=torch.bfloat16,
    )
    fn, cls = compile_gate(tile, pingpong, idx is not None)
    epi = cls.EpilogueArguments(postact=out)
    sched = TileSchedulerOptions(
        torch.cuda.get_device_properties(A.device).multi_processor_count,
        raster_order=None,
        max_swizzle_size=8,
    )
    fn(A, W, None, None, epi, sched, VarlenArguments(mCuSeqlensM=cu, mAIdx=idx), SA, SB)
    return out
