import triton as tr
import triton.language as tl


@tr.jit
def _act_quant(
    G,
    Q,
    S,
    R: tl.constexpr,
    I: tl.constexpr,
    B: tl.constexpr = 4,
    DIV: tl.constexpr = 3,
):
    rows = tl.program_id(0) * B + tl.arange(0, B)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    g = tl.load(G + rows[:, None] * (2 * I) + n[None, :], rows[:, None] < R, 0).to(
        tl.float32
    )
    u = tl.load(G + rows[:, None] * (2 * I) + I + n[None, :], rows[:, None] < R, 0).to(
        tl.float32
    )
    # Preserve the materialized SGL activation's BF16 rounding.
    if DIV == 1 or DIV == 3:
        act = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[g, 1.0 + tl.exp(-g)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    elif DIV == 2:
        act = tl.div_rn(g, 1.0 + tl.exp(-g))
    else:
        act = g * tl.sigmoid(g)
    a = (act * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a), 1), 1e-10)
    s = amax * (1.0 / 448.0)
    if DIV == 3:
        multiplier = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[tl.full((), 448.0, tl.float32), amax],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        multiplier = tl.div_rn(448.0, amax)
    q = tl.minimum(tl.maximum(a * multiplier[:, None], -448.0), 448.0)
    tl.store(Q + rows[:, None] * I + n[None, :], q, rows[:, None] < R)
    tl.store(S + rows * (I // 128) + tl.program_id(1), s, rows < R)
