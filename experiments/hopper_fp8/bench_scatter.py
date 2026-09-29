import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json, statistics
from pathlib import Path
from functools import partial
import torch
from optimized_ops import metadata, pack_scatter
from scatter_candidate import pack_scatter_groups

P = Path(__file__).parent
OPTIONS = [
    ("baseline", None),
    ("r1g32w4", (1, 32, 4)),
    ("r1g32w8", (1, 32, 8)),
    ("r2g16w4", (2, 16, 4)),
    ("r4g8w4", (4, 8, 4)),
    ("r1g8w4", (1, 8, 4)),
    ("r2g32w8", (2, 32, 8)),
]


if os.environ.get("SELECTED_ONLY"):
    OPTIONS = [o for o in OPTIONS if o[0] in ("baseline", "r1g32w8")]

def emit(r):
    print(json.dumps(r), flush=True)
    with (P / os.environ.get("RESULT_FILE", "scatter-screen.jsonl")).open("a") as f:
        f.write(json.dumps(r) + "\n")


def check(a, b):
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
    assert torch.equal(a[1], b[1])


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def ms(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(50):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 50


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    cases = [("uniform", t) for t in (1, 256, 1024, 4096, 8192, 16384)] + [
        ("hot", 16384),
        ("skew", 16384),
    ]
    for mode, t in cases:
        torch.manual_seed(1000 + t)
        x = torch.randn(t, 3072, device="cuda", dtype=torch.bfloat16) * 0.2
        logits = torch.randn(t, 256, device="cuda")
        if mode == "hot":
            logits[:, :8] += 100
        if mode == "skew":
            logits[:, :16] += 2
        vals, ids = logits.topk(8)
        _, _, reverse, _ = metadata(ids.int(), vals.softmax(-1), 256, small_limit=512)
        ref = pack_scatter(x, reverse, 8)
        graphs = {}
        options = list(OPTIONS)
        if os.environ.get("CAPTURE_ORDER") == "reverse":
            options.reverse()
        for name, cfg in options:
            if cfg is None:
                fn = lambda: pack_scatter(x, reverse, 8)
            else:
                r, g, w = cfg
                fn = partial(
                    pack_scatter_groups, x, reverse, 8, rows=r, groups=g, warps=w
                )
            out = fn()
            torch.cuda.synchronize()
            check(out, ref)
            gr, out = capture(fn)
            gr.replay()
            torch.cuda.synchronize()
            check(out, ref)
            graphs[name] = (gr, out)
            emit(
                dict(
                    event="correctness",
                    T=t,
                    routing=mode,
                    name=name,
                    bytes_equal=True,
                    scales_equal=True,
                )
            )
        for gr, _ in graphs.values():
            for _ in range(20):
                gr.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        for i in range(9):
            names = list(graphs)
            names = names[i % len(names) :] + names[: i % len(names)]
            if i % 2:
                names.reverse()
            for name in names:
                samples[name].append(ms(graphs[name][0]))
        emit(
            dict(
                event="timing",
                T=t,
                routing=mode,
                ms={n: statistics.median(v) for n, v in samples.items()},
                samples=samples,
            )
        )
        x.mul_(0.7)
        reverse.copy_(reverse.flip(0))
        ref = pack_scatter(x, reverse, 8)
        for name, (gr, out) in graphs.items():
            gr.replay()
            torch.cuda.synchronize()
            check(out, ref)
            emit(
                dict(
                    event="graph_mutation",
                    T=t,
                    routing=mode,
                    name=name,
                    bytes_equal=True,
                    scales_equal=True,
                )
            )
        del graphs, gr, out, ref
    emit(dict(event="completed"))


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
