"""Multi-group/token input quantization + top-k scatter. Experiment default."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def scatter_groups(
    X,
    SX,
    REV,
    Q,
    S,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    PRE: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.program_id(1) * G + tl.arange(0, G)
    h = tl.arange(0, 128)
    mask = (r[:, None, None] < T) & (g[None, :, None] < H // 128)
    a = tl.load(
        X + r[:, None, None] * H + g[None, :, None] * 128 + h[None, None, :], mask, 0.0
    )
    if PRE:
        q = a
        scale = tl.load(
            SX + r[:, None] * SS0 + g[None, :] * SS1,
            (r[:, None] < T) & (g[None, :] < H // 128),
            0,
        )
    else:
        a = a.to(tl.float32)
        mx = tl.maximum(tl.max(tl.abs(a), 2), 1.0e-10)
        scale = mx * (1.0 / 448.0)
        mult = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [tl.full((), 448.0, tl.float32), mx],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        q = tl.minimum(tl.maximum(a * mult[:, :, None], -448.0), 448.0).to(
            Q.dtype.element_ty
        )
    for k in tl.static_range(K):
        dest = tl.load(REV + r * K + k, r < T, 0)
        tl.store(
            Q + dest[:, None, None] * H + g[None, :, None] * 128 + h[None, None, :],
            q,
            mask,
        )
        tl.store(
            S + dest[:, None] * (H // 128) + g[None, :],
            scale,
            (r[:, None] < T) & (g[None, :] < H // 128),
        )


def pack_scatter_groups(
    x, reverse, K, q=None, scale=None, *, rows=1, groups=32, warps=8
):
    t, h = x.shape
    assert x.is_contiguous() and h % 128 == 0
    out = torch.empty((t * K, h), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((t * K, h // 128), device=x.device, dtype=torch.float32)
    scatter_groups[(tr.cdiv(t, rows), tr.cdiv(h // 128, groups))](
        x if q is None else q,
        scale,
        reverse,
        out,
        s,
        t,
        h,
        K,
        q is not None,
        *(scale.stride() if scale is not None else (0, 0)),
        rows,
        groups,
        num_warps=warps
    )
    return out, s
