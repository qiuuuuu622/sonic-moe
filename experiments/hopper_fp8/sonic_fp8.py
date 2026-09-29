"""Experimental SonicMoE/QuACK Hopper FP8 forward, SGL fused_experts ABI.
Both GEMMs are QuACK WGMMA kernels. Unsupported inputs raise, never SGL fallback.
"""

from __future__ import annotations
import _bootstrap  # noqa: F401; select local experiment dependencies
from typing import Optional, List
import os
from fused_gate_quant import grouped_gate_quant

# Repeated GPU7 measurements regress at medium/large token batches.
# Keep activation quantization separate by default; opt in for experiments.
FUSED_QUANT = bool(int(os.environ.get("FUSED_QUANT", "0")))
import torch
import triton
from quack_fp8 import grouped_fp8, grouped_gate_fp8
from pointwise_quant import _act_quant
from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton
from sonicmoe.functional.forward import _router_forward
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

# Experiment controls, frozen before graph capture; independent from SGL config.
GATE_TILE = (64, 128)
DOWN_TILE = (64, 128)
PINGPONG = True
FUSED_GATE = True
AUTO_TILE = True
# Frozen candidate: only Gate threshold and metadata crossover differ from baseline.
OPT_POLICY = bool(int(os.environ.get("OPT_POLICY", "1")))
PACKED = True
FAST_METADATA = True
from optimized_ops import metadata, pack, combine
from scatter_candidate import pack_scatter_groups as pack_scatter

PACK_MODE = os.environ.get("PACK_MODE", "scatter")


@torch.no_grad()
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    a1_q: Optional[torch.Tensor] = None,
    fuse_swiglu_interleaved: bool = False,
):
    c = moe_runner_config
    x = hidden_states
    T, H = x.shape
    E, N, _ = w1.shape
    I = N // 2
    K = topk_output.topk_ids.shape[1]
    gt, dt, pp = GATE_TILE, DOWN_TILE, PINGPONG
    if AUTO_TILE and T >= 4096:
        gt = dt = (128, 128)
        pp = False
    use_policy = AUTO_TILE and OPT_POLICY and (E, H, I, K) == (256, 3072, 1024, 8)
    gpp = pp
    if use_policy and T >= 2048:
        gt, gpp = (128, 128), False
    if not (
        use_fp8_w8a8
        and block_shape == [128, 128]
        and c.activation == "silu"
        and c.is_gated
        and not c.gate_up_interleaved
        and not c.no_combine
        and not c.apply_router_weight_on_input
        and c.routed_scaling_factor in (None, 1.0)
        and not any(
            (
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                fuse_swiglu_interleaved,
            )
        )
        and all(
            v is None
            for v in (
                b1,
                b2,
                w1_zp,
                w2_zp,
                c.gemm1_alpha,
                c.gemm1_clamp_limit,
                c.swiglu_limit,
            )
        )
        and c.num_experts == c.num_local_experts == E
        and c.top_k == K
    ):
        raise NotImplementedError(
            "This port supports TP1/EP1 block-FP8 SwiGLU with standard combine only; no fallback."
        )
    assert x.dtype == torch.bfloat16 and w1.dtype == w2.dtype == torch.float8_e4m3fn
    assert w1_scale.dtype == w2_scale.dtype == torch.float32
    assert (
        H % 128 == I % 128 == 0 and w1.shape == (E, 2 * I, H) and w2.shape == (E, H, I)
    )
    assert w1_scale.shape == (E, N // 128, H // 128) and w2_scale.shape == (
        E,
        H // 128,
        I // 128,
    )
    ids = topk_output.topk_ids
    scores = topk_output.topk_weights
    assert ids.dtype == torch.int32 and scores.dtype == torch.float32
    assert ids.shape == scores.shape == (T, K)
    assert all(
        v.is_cuda and v.device == x.device and v.is_contiguous()
        for v in [x, w1, w2, w1_scale, w2_scale, ids, scores]
    )
    if T == 0:
        return x if c.inplace else torch.empty_like(x)
    TK = T * K
    device = x.device
    if FAST_METADATA and (
        E & (E - 1) or (E + 1) * triton.next_power_of_2(TK + 1) > 2**31 - 1
    ):
        raise NotImplementedError(
            "Fixed-top-k metadata requires power-of-two experts and int32 sort keys."
        )
    if FAST_METADATA:
        offsets, gather, reverse, sorted_scores = metadata(ids, scores, E, small_limit=512 if use_policy else 2048)
    else:
        scatter = torch.empty(TK, dtype=torch.int32, device=device)
        reverse = torch.empty_like(scatter)
        gather = torch.empty_like(scatter)
        freq = torch.empty(E, dtype=torch.int32, device=device)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
        token_offsets = torch.empty(T + 1, dtype=torch.int32, device=device)
        tokens = torch.arange(T, dtype=torch.int32, device=device).repeat_interleave(K)
        general_routing_router_metadata_triton(
            tokens,
            ids.flatten(),
            T,
            E,
            freq,
            offsets,
            gather,
            scatter,
            reverse,
            token_offsets,
        )
    if a1_q is None:
        if not PACKED:
            qx, sx = sglang_per_token_group_quant_fp8(x, 128)
    else:
        assert (
            a1_q.dtype == torch.float8_e4m3fn
            and a1_q.shape[0] >= T
            and a1_q.shape[1] == H
        )
        assert a1_scale is not None and a1_scale.dtype == torch.float32
        assert a1_q.device == x.device and a1_q.is_contiguous()
        assert a1_scale.device == x.device and a1_scale.ndim == 2
        assert a1_scale.shape[0] >= T and a1_scale.shape[1] == H // 128
        if not PACKED:
            qx, sx = a1_q, a1_scale.contiguous()
    if PACKED:
        if PACK_MODE == "scatter" and FAST_METADATA:
            qx, sx = pack_scatter(x, reverse, K, a1_q, a1_scale)
        else:
            qx, sx = pack(x, gather, a1_q, a1_scale)
    gate_idx = None if PACKED else gather
    if FUSED_QUANT:
        assert PACKED
        qa, sa = grouped_gate_quant(
            qx,
            w1,
            sx,
            w1_scale,
            offsets,
            tile=(int(os.environ.get("Q_TILE_M", "64")), 256),
            pingpong=bool(int(os.environ.get("Q_PINGPONG", "0"))),
        )
    elif FUSED_GATE:
        act = grouped_gate_fp8(
            qx, w1, sx, w1_scale, offsets, gate_idx, tile=gt, pingpong=gpp
        )
        qa, sa = sglang_per_token_group_quant_fp8(act, 128)
        del act
    else:
        gate_up = grouped_fp8(
            qx, w1, sx, w1_scale, offsets, gate_idx, tile=gt, pingpong=gpp
        )
        qa = torch.empty((TK, I), dtype=torch.float8_e4m3fn, device=device)
        sa = torch.empty((TK, I // 128), dtype=torch.float32, device=device)
        _act_quant[(triton.cdiv(TK, 4), I // 128)](gate_up, qa, sa, TK, I, num_warps=4)
        del gate_up
    if not FAST_METADATA:
        sorted_scores = scores.flatten()[scatter.long()]
    y = grouped_fp8(
        qa, w2, sa, w2_scale, offsets, tile=dt, pingpong=pp, weights=sorted_scores
    )
    out = torch.empty_like(x)
    # Down epilogue already applied route weights before BF16 conversion.
    # Combine in original top-k slot order using the reverse routing map.
    if FAST_METADATA:
        combine(y, reverse, out, K)
    else:
        _router_forward(
            y,
            out,
            torch.ones_like(scores).flatten(),
            reverse,
            token_offsets,
            K,
            H,
            False,
        )
    if c.inplace:
        x.copy_(out)
        return x
    return out
