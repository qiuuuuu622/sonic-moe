import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import torch
from optimized_ops import pack_scatter as baseline
from scatter_candidate import pack_scatter_groups
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def candidate(*args):
    return pack_scatter_groups(*args, rows=1, groups=32, warps=8)


def check(a, b):
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
    assert torch.equal(a[1], b[1])


@torch.no_grad()
def main():
    assert torch.cuda.device_count() == 1
    torch.manual_seed(290929)
    cases = [
        (1, 3072),
        (3, 384),
        (63, 3072),
        (64, 3072),
        (65, 3072),
        (255, 3072),
        (256, 3072),
        (257, 3072),
    ]
    if os.environ.get("SANITIZE"):
        cases = [(1, 3072), (3, 384), (65, 3072), (257, 3072)]
    for t, h in cases:
        for magnitude in (1.0e-12, 1.0, 1.0e4):
            x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * magnitude
            x[0, :128] = 0
            rev = torch.randperm(t * 8, device="cuda", dtype=torch.int32)
            check(candidate(x, rev, 8), baseline(x, rev, 8))
            q, s = sglang_per_token_group_quant_fp8(x, 128)
            pad = ((t + 3) // 4) * 4
            qp = torch.zeros((pad, h), device="cuda", dtype=q.dtype)
            qp[:t].copy_(q)
            sp = torch.zeros((h // 128, pad), device="cuda").T
            sp[:t].copy_(s)
            check(candidate(x, rev, 8, qp, sp), baseline(x, rev, 8, qp, sp))
            for _ in range(3):
                candidate(x, rev, 8, qp, sp)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = candidate(x, rev, 8, qp, sp)
            rev.copy_(rev.flip(0))
            qp.zero_()
            g.replay()
            torch.cuda.synchronize()
            check(out, baseline(x, rev, 8, qp, sp))
            print(
                json.dumps(
                    dict(
                        event="case",
                        T=t,
                        H=h,
                        magnitude=magnitude,
                        bytes_equal=True,
                        scales_equal=True,
                    )
                ),
                flush=True,
            )
    print(json.dumps(dict(event="completed")), flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
