"""Fixed-top-k metadata and FP8 quant/permute, independent of SGL GEMM."""

import os
import torch
import triton as tr
import triton.language as tl


@tr.jit
def sort_count(
    IDS,
    SORT,
    COUNT,
    R: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    PAD: tl.constexpr,
):
    b = tl.program_id(0)
    i = b * B + tl.arange(0, B)
    e = tl.load(IDS + i, i < R, E)
    key = tl.where(i < R, e * PAD + i, E * PAD + PAD - 1)
    key = tl.sort(key, descending=False)
    tl.store(SORT + b * B + tl.arange(0, B), key)
    cnt = tl.histogram(e, E, mask=i < R)
    tl.store(COUNT + tl.arange(0, E) * C + b, cnt)


@tr.jit
def prefix(COUNT, PREFIX, TOTAL, E: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    e = tl.program_id(0)
    c = tl.arange(0, BC)
    cnt = tl.load(COUNT + e * C + c, c < C, 0)
    cs = tl.cumsum(cnt, 0)
    tl.store(PREFIX + e * C + c, cs - cnt, c < C)
    tl.store(TOTAL + e, tl.sum(cnt, 0))


@tr.jit
def emit_routes(
    SORT,
    PREFIX,
    TOTAL,
    OFF,
    GATHER,
    REVERSE,
    WEIGHT,
    SCORES,
    R: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    PAD: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.arange(0, B)
    key = tl.load(SORT + b * B + j)
    e = key // PAD
    route = key % PAD
    prev = tl.gather(e, tl.maximum(j - 1, 0), 0)
    start = tl.associative_scan(tl.where((j == 0) | (e != prev), j, 0), 0, _max)
    totals = tl.load(TOTAL + tl.arange(0, E))
    ends = tl.cumsum(totals, 0)
    begins = ends - totals
    if b == 0:
        tl.store(OFF + tl.arange(0, E), begins)
        tl.store(OFF + E, tl.sum(totals, 0))
    base = tl.gather(begins, tl.minimum(e, E - 1), 0)
    pos = base + tl.load(PREFIX + e * C + b, route < R, 0) + j - start
    mask = route < R
    tl.store(GATHER + pos, route // K, mask)
    tl.store(REVERSE + route, pos, mask)
    score = tl.load(SCORES + route, mask, 0)
    tl.store(WEIGHT + pos, score, mask)


@tr.jit
def _max(a, b):
    return tl.maximum(a, b)


def metadata(ids, scores, E, small_limit=2048):
    T, K = ids.shape
    R = T * K
    B = 256
    C = tr.cdiv(R, B)
    if R <= small_limit:
        off = torch.empty(E + 1, device=ids.device, dtype=torch.int32)
        gather = torch.empty(R, device=ids.device, dtype=torch.int32)
        rev = torch.empty_like(gather)
        weight = torch.empty(R, device=ids.device, dtype=torch.float32)
        small_metadata[(1,)](
            ids,
            scores,
            off,
            gather,
            rev,
            weight,
            R,
            K,
            E,
            tr.next_power_of_2(R),
            tr.next_power_of_2(R + 1),
            num_warps=4,
        )
        return off, gather, rev, weight
    sort = torch.empty(C * B, device=ids.device, dtype=torch.int32)
    count = torch.empty((E, C), device=ids.device, dtype=torch.int32)
    pref = torch.empty_like(count)
    total = torch.empty(E, device=ids.device, dtype=torch.int32)
    off = torch.empty(E + 1, device=ids.device, dtype=torch.int32)
    gather = torch.empty(R, device=ids.device, dtype=torch.int32)
    rev = torch.empty_like(gather)
    weight = torch.empty(R, device=ids.device, dtype=torch.float32)
    sort_count[(C,)](
        ids, sort, count, R, E, C, B, tr.next_power_of_2(R + 1), num_warps=4
    )
    prefix[(E,)](count, pref, total, E, C, tr.next_power_of_2(C), num_warps=4)
    emit_routes[(C,)](
        sort,
        pref,
        total,
        off,
        gather,
        rev,
        weight,
        scores,
        R,
        K,
        E,
        C,
        B,
        tr.next_power_of_2(R + 1),
        num_warps=4,
    )
    return off, gather, rev, weight


@tr.jit
def quant_gather(X, G, Q, S, R: tl.constexpr, H: tl.constexpr, B: tl.constexpr = 4):
    r = tl.program_id(0) * B + tl.arange(0, B)
    group = tl.program_id(1)
    src = tl.load(G + r, r < R, 0)
    c = group * 128 + tl.arange(0, 128)
    a = tl.load(X + src[:, None] * H + c[None, :], r[:, None] < R, 0.0).to(tl.float32)
    mx = tl.maximum(tl.max(tl.abs(a), 1), 1.0e-10)
    scale = mx * (1.0 / 448.0)
    mult = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [tl.full((), 448.0, tl.float32), mx],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = tl.minimum(tl.maximum(a * mult[:, None], -448.0), 448.0)
    tl.store(Q + r[:, None] * H + c[None, :], q, r[:, None] < R)
    tl.store(S + r * (H // 128) + group, scale, r < R)


@tr.jit
def prequant_gather(
    X,
    SX,
    G,
    Q,
    S,
    R: tl.constexpr,
    H: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    B: tl.constexpr = 4,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    group = tl.program_id(1)
    src = tl.load(G + r, r < R, 0)
    c = group * 128 + tl.arange(0, 128)
    a = tl.load(X + src[:, None] * H + c[None, :], r[:, None] < R, 0.0)
    s = tl.load(SX + src * SS0 + group * SS1, r < R, 0)
    tl.store(Q + r[:, None] * H + c[None, :], a, r[:, None] < R)
    tl.store(S + r * (H // 128) + group, s, r < R)


def pack(x, gather, q=None, scale=None):
    R = gather.numel()
    H = x.shape[1]
    out = torch.empty((R, H), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((R, H // 128), device=x.device, dtype=torch.float32)
    if q is None:
        quant_gather[(tr.cdiv(R, 4), H // 128)](x, gather, out, s, R, H)
    else:
        prequant_gather[(tr.cdiv(R, 4), H // 128)](
            q, scale, gather, out, s, R, H, *scale.stride()
        )
    return out, s


@tr.jit
def combine_kernel(
    Y,
    REV,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr = 512,
):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(K):
        row = tl.load(REV + t * K + k)
        val = tl.load(Y + row * H + h, h < H, 0).to(tl.float32)
        acc += val
    tl.store(OUT + t * H + h, acc, h < H)


def combine(y, rev, out, K):
    T, H = out.shape
    B = int(os.environ.get("COMBINE_B", str(4096 if T >= 4096 else 1024)))
    combine_kernel[(T, tr.cdiv(H, B))](
        y, rev, out, T, H, K, B, num_warps=4 if B <= 1024 else 8
    )


@tr.jit
def quant_scatter(
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
    B: tl.constexpr = 4,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    group = tl.program_id(1)
    c = group * 128 + tl.arange(0, 128)
    a = tl.load(X + r[:, None] * H + c[None, :], r[:, None] < T, 0.0)
    if PRE:
        q = a
        scale = tl.load(SX + r * SS0 + group * SS1, r < T, 0)
    else:
        a = a.to(tl.float32)
        mx = tl.maximum(tl.max(tl.abs(a), 1), 1.0e-10)
        scale = mx * (1.0 / 448.0)
        mult = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [tl.full((), 448.0, tl.float32), mx],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        q = tl.minimum(tl.maximum(a * mult[:, None], -448.0), 448.0).to(
            Q.dtype.element_ty
        )
    for k in tl.static_range(K):
        dest = tl.load(REV + r * K + k, r < T, 0)
        tl.store(Q + dest[:, None] * H + c[None, :], q, r[:, None] < T)
        tl.store(S + dest * (H // 128) + group, scale, r < T)


def pack_scatter(x, reverse, K, q=None, scale=None):
    T, H = x.shape
    out = torch.empty((T * K, H), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((T * K, H // 128), device=x.device, dtype=torch.float32)
    quant_scatter[(tr.cdiv(T, 4), H // 128)](
        x if q is None else q,
        scale,
        reverse,
        out,
        s,
        T,
        H,
        K,
        q is not None,
        *(scale.stride() if scale is not None else (0, 0))
    )
    return out, s


@tr.jit
def small_metadata(
    IDS,
    SCORES,
    OFF,
    GATHER,
    REV,
    WEIGHT,
    R: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    PAD: tl.constexpr,
):
    r = tl.arange(0, B)
    e = tl.load(IDS + r, r < R, 0)
    counts = tl.histogram(e, E, mask=r < R)
    ends = tl.cumsum(counts, 0)
    tl.store(OFF + tl.arange(0, E), ends - counts)
    tl.store(OFF + E, tl.sum(counts, 0))
    key = tl.sort(tl.where(r < R, e * PAD + r, E * PAD + PAD - 1), descending=False)
    route = key % PAD
    tl.store(GATHER + r, route // K, r < R)
    tl.store(REV + route, r, r < R)
    w = tl.load(SCORES + route, r < R, 0)
    tl.store(WEIGHT + r, w, r < R)
