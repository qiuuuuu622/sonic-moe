import torch


def quant_weight(w):
    q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        w.shape[0],
        w.shape[1] // 128,
        w.shape[2] // 128,
        device="cuda",
        dtype=torch.float32,
    )
    for e in range(w.shape[0]):
        z = w[e].float().reshape(w.shape[1] // 128, 128, w.shape[2] // 128, 128)
        sc = z.abs().amax(dim=(1, 3)).clamp_min(1e-10) / 448
        q[e] = (
            (z / sc[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        ).reshape(w.shape[1:])
        scales[e] = sc
    return q, scales


def dequant_weight(q, s):
    return (
        q.float().reshape(q.shape[0] // 128, 128, q.shape[1] // 128, 128)
        * s[:, None, :, None]
    ).reshape(q.shape)


def qdq_act(x):
    z = x.float().reshape(x.shape[0], -1, 128)
    sc = z.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    return ((z / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).reshape(
        x.shape
    )
