"""Experimental single-kernel FP8 Gate + BF16-rounded SwiGLU + block quantization."""

import _bootstrap  # noqa: F401; select local experiment dependencies

from quack_fp8 import *
from quack.epilogue.ops import GroupedColStatsBase, GroupedColStatsOut, EpiOp


class AMax(GroupedColStatsBase):
    combine = "max"

    @cute.jit
    def stat_value(self, total, group_cols):
        return cute.arch.fmax(total, Float32(1.0e-10))


class ScaleOut(AMax):
    @cute.jit
    def stat_value(self, total, group_cols):
        return cute.arch.fmax(total, Float32(1.0e-10)) * Float32(1.0 / 448.0)


class FP8DirectStore(EpiOp):
    fn_port = "sink"

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        coords = ctx.partition_for_epilogue_fn(
            cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        )
        coords = cute.group_modes(coords, 3, cute.rank(coords))
        row0 = (
            ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx]
            + ctx.tile_coord_mnkl[0] * ctx.tile_M
        )
        limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx + 1]
        col0 = ctx.tile_coord_mnkl[1] * ctx.tile_N
        return (param, coords, row0, limit, col0)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        return (
            state[0],
            state[1][None, None, None, epi_coord],
            state[2],
            state[3],
            state[4],
        )

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        out, coords, row0, limit, col0 = state
        for i in cutlass.range_constexpr(cute.size(frag)):
            r = row0 + coords[i][0]
            col = col0 + coords[i][1]
            if r < limit and col // 2 < out.shape[1] and col % 2 == 0:
                out[r, col // 2] = frag[i].to(out.element_type)


_amax = AMax("amax")
_scale = ScaleOut("amax")


def rounded_act(acc):
    g, u = unpack(acc)
    g = g.to(cutlass.BFloat16).to(Float32)
    u = u.to(cutlass.BFloat16).to(Float32)
    return swiglu(g, u).to(cutlass.BFloat16).to(Float32)


def activation_amax(acc):
    a = rounded_act(acc)
    return {"amax": cute.arch.fmax(a, -a)}


@gemm_epilogue(
    outs={"postact": FP8DirectStore("postact")},
    mode="acc_pair",
    ops={"amax": _amax},
    prepass=activation_amax,
    prepass_outs=("amax",),
    extra_ops=(GroupedColStatsOut("scale", _scale),),
)
def quantized_swiglu(acc, amax):
    a = rounded_act(acc)
    mx, _ = unpack(amax)
    q = a * (Float32(448.0) * cute.arch.rcp_approx(mx))
    v = cute.arch.fmin(cute.arch.fmax(q, Float32(-448.0)), Float32(448.0))
    return {"postact": (v, v)}


@lru_cache(None)
def compile_quant_gate(tile, pingpong):
    assert tile[1] % 256 == 0
    f8 = torch2cute_dtype_map[torch.float8_e4m3fn]
    cls = quantized_swiglu._mint(
        (("amax", "value"),),
        9,
        True,
        False,
        (),
        RoundingMode.RN,
        (("amax", "Const"),),
        False,
    )
    cls.sgl_quant_fused = True
    A, W, _, _, m, n, k, l = make_fake_gemm_tensors(
        f8, f8, None, None, "k", "k", "n", "n", varlen_m=True, gather_A=False
    )

    def fake(dtype, shape, align=1):
        return make_fake_tensor(
            dtype, shape, divisibility=align, leading_dim=len(shape) - 1
        )

    O = fake(f8, (m, cute.sym_int()), 16)
    SO = fake(Float32, (m, cute.sym_int()))
    SA = fake(Float32, (cute.sym_int(), cute.sym_int()))
    SB = fake(Float32, (cute.sym_int(), cute.sym_int(), cute.sym_int()))
    epi = cls.EpilogueArguments(amax=256, postact=O, scale=SO)
    sched = TileSchedulerOptions(Int32(1))
    varlen = make_fake_varlen_args(True, False, False, m)

    def setup(obj):
        obj.sgl_fp8_scales = True
        obj.sgl_gate_layout = True
        obj.sgl_scale_product = True
        obj.implicit_dtype = cutlass.BFloat16

    fn = compile_gemm_kernel(
        cls,
        f8,
        (*tile, 128),
        (1, 1, 1),
        pingpong,
        True,
        False,
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
def grouped_gate_quant(A, W, SA, SB, cu, tile=(64, 256), pingpong=False):
    q = torch.empty(
        (A.shape[0], W.shape[1] // 2), device=A.device, dtype=torch.float8_e4m3fn
    )
    s = torch.empty(
        (A.shape[0], W.shape[1] // 256), device=A.device, dtype=torch.float32
    )
    fn, cls = compile_quant_gate(tile, pingpong)
    epi = cls.EpilogueArguments(amax=None, postact=q, scale=s)
    sched = TileSchedulerOptions(
        torch.cuda.get_device_properties(A.device).multi_processor_count,
        raster_order=None,
        max_swizzle_size=8,
    )
    fn(A, W, None, None, epi, sched, VarlenArguments(mCuSeqlensM=cu), SA, SB)
    return q, s
