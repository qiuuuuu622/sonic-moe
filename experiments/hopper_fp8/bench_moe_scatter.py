import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json, inspect, statistics, time
from pathlib import Path
import torch
import sonic_fp8
from optimized_ops import pack_scatter as stable_pack
from scatter_candidate import pack_scatter_groups

# Validate the published default, rather than injecting a candidate only in the benchmark.
assert sonic_fp8.pack_scatter is pack_scatter_groups
from sonic_fp8 import fused_experts
from helpers import quant_weight
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts as sgl
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput

sonic_fp8.OPT_POLICY = True
P = Path(__file__).parent
for flag in ("PACKED", "FAST_METADATA", "FUSED_GATE", "AUTO_TILE"):
    if flag in os.environ:
        setattr(sonic_fp8, flag, bool(int(os.environ[flag])))


def emit(r):
    print(json.dumps(r), flush=True)
    with (P / os.environ.get("RESULT_FILE", "results.jsonl")).open("a") as f:
        f.write(json.dumps(r) + "\n")


def check(a, b):
    delta = a.float() - b.float()
    rel = (delta.norm() / b.float().norm().clamp_min(1e-20)).item()
    mx = delta.abs().max().item() if delta.numel() else 0
    assert torch.isfinite(a).all() and rel < 0.001 and mx < 0.001, (rel, mx)
    return dict(rel_l2=rel, max_abs=mx)


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
    repeats = int(os.environ.get("BENCH_REPLAYS", "30"))
    for _ in range(repeats):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / repeats


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="file:///tmp/sonic-port-" + str(os.getpid()),
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    assert str(inspect.signature(fused_experts)) == str(inspect.signature(sgl))
    if os.environ.get("TUNE"):
        sonic_fp8.AUTO_TILE = False
    if os.environ.get("PINGPONG"):
        sonic_fp8.PINGPONG = bool(int(os.environ["PINGPONG"]))
    import quack.gemm_sm90

    emit(
        dict(
            event="start",
            signature_match=True,
            fused_quant=sonic_fp8.FUSED_QUANT,
            optimized_policy=sonic_fp8.OPT_POLICY,
            capture_order=os.environ.get("CAPTURE_ORDER", "forward"),
            policy_version="scatter-r1g32w8",
            scatter_rows=1,
            scatter_groups=32,
            scatter_warps=8,
            quack_source=quack.gemm_sm90.__file__,
            pingpong=sonic_fp8.PINGPONG,
        )
    )
    torch.manual_seed(1734)
    w1 = torch.randn(256, 2048, 3072, device="cuda", dtype=torch.bfloat16) / 3072**0.5
    w2 = torch.randn(256, 3072, 1024, device="cuda", dtype=torch.bfloat16) / 1024**0.5
    q1, s1 = quant_weight(w1)
    q2, s2 = quant_weight(w2)
    del w1, w2
    cfg = MoeRunnerConfig(
        num_experts=256,
        num_local_experts=256,
        hidden_size=3072,
        intermediate_size_per_partition=1024,
        top_k=8,
        inplace=False,
        gate_up_interleaved=False,
    )
    cases = [
        ("uniform", int(t)) for t in os.environ.get("TOKENS", "32,1024").split(",")
    ]
    if os.environ.get("EDGES"):
        cases += [("hot", 1025), ("skew", 4097)]
    cases += [tuple(v) for v in json.loads(os.environ.get("EXTRA_CASES", "[]"))]
    for mode, T in cases:
        torch.manual_seed(1000 + T)
        x = torch.randn(T, 3072, device="cuda", dtype=torch.bfloat16) * 0.2
        logits = torch.randn(T, 256, device="cuda")
        if mode == "hot":
            logits[:, :8] += 100
        if mode == "skew":
            logits[:, :16] += 2
        vals, ids = logits.topk(8)
        ids = ids.int()
        scores = vals.softmax(-1)
        args = dict(
            hidden_states=x,
            w1=q1,
            w2=q2,
            topk_output=StandardTopKOutput(scores, ids, logits),
            moe_runner_config=cfg,
            use_fp8_w8a8=True,
            w1_scale=s1,
            w2_scale=s2,
            block_shape=[128, 128],
        )
        emit(dict(event="case_start", T=T, routing=mode))
        ref = sgl(**args)
        # Explicitly prove the candidate does not dispatch to SGL Triton GEMM.
        from unittest.mock import patch

        with patch(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.invoke_fused_moe_kernel",
            side_effect=AssertionError("SGL GEMM forbidden in Sonic candidate"),
        ):
            y = fused_experts(**args)
            torch.cuda.synchronize()
        emit(
            dict(
                event="correctness",
                T=T,
                routing=mode,
                no_sgl_gemm=True,
                **check(y, ref)
            )
        )
        if os.environ.get("TUNE"):
            options = [
                ((64, 128), (64, 128), False),
                ((64, 128), (64, 128), True),
                ((128, 128), (128, 128), False),
                ((128, 128), (64, 128), False),
                ((64, 256), (64, 128), False),
                ((64, 256), (64, 128), True),
                ((128, 256), (128, 128), False),
                ((128, 256), (64, 128), False),
            ]
            for gt, dt, pp in options:
                sonic_fp8.GATE_TILE = gt
                sonic_fp8.DOWN_TILE = dt
                sonic_fp8.PINGPONG = pp
                yy = fused_experts(**args)
                torch.cuda.synchronize()
                er = check(yy, ref)
                gg, oo = capture(lambda: fused_experts(**args))
                vals = [ms(gg) for _ in range(5)]
                emit(
                    dict(
                        event="tune",
                        T=T,
                        gate=gt,
                        down=dt,
                        pingpong=pp,
                        ms=statistics.median(vals),
                        samples=vals,
                        **er
                    )
                )
                del yy, gg, oo
            continue
        if os.environ.get("PROFILE_STAGES"):
            original = sonic_fp8.grouped_fp8
            stage_events = []

            def wrapped(*a, **kw):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                result = original(*a, **kw)
                end.record()
                stage_events.append(
                    ("down" if kw.get("weights") is not None else "gate", start, end)
                )
                return result

            sonic_fp8.grouped_fp8 = wrapped
            fused_experts(**args)
            torch.cuda.synchronize()
            emit(
                dict(
                    event="stages",
                    T=T,
                    ms={n: a.elapsed_time(b) for n, a, b in stage_events},
                )
            )
            sonic_fp8.grouped_fp8 = original

        def previous():
            saved = sonic_fp8.pack_scatter
            sonic_fp8.pack_scatter = stable_pack
            try:
                return fused_experts(**args)
            finally:
                sonic_fp8.pack_scatter = saved

        graphs = {}
        backends = [
            ("sgl", lambda: sgl(**args)),
            ("stable", previous),
            ("sonic_fp8", lambda: fused_experts(**args)),
        ]
        if os.environ.get("PAIR_ONLY"):
            backends = [(name, fn) for name, fn in backends if name != "sgl"]
        if os.environ.get("CAPTURE_ORDER") == "reverse":
            backends.reverse()
        for name, fn in backends:
            g, out = capture(fn)
            g.replay()
            torch.cuda.synchronize()
            check(out, ref)
            graphs[name] = (g, out)
        # Settle clocks after compilation; timing still alternates independent graphs.
        for g, out in graphs.values():
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
        samples = {n: [] for n in graphs}
        for r in range(9):
            names = list(graphs)
            names = names[r % len(names) :] + names[: r % len(names)]
            if r % 2:
                names.reverse()
            for name in names:
                samples[name].append(ms(graphs[name][0]))
        emit(
            dict(
                event="timing",
                T=T,
                routing=mode,
                ms={n: statistics.median(v) for n, v in samples.items()},
                samples=samples,
            )
        )
        x.mul_(0.7)
        ids.copy_(ids.roll(1, 0))
        scores.copy_(scores.roll(1, 1))
        ref = sgl(**args)
        graphs["sonic_fp8"][0].replay()
        torch.cuda.synchronize()
        emit(
            dict(
                event="graph_mutation",
                T=T,
                routing=mode,
                **check(graphs["sonic_fp8"][1], ref)
            )
        )
        if os.environ.get("EDGES"):
            from dataclasses import replace
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            qx, sx = sglang_per_token_group_quant_fp8(x, 128)
            padded = ((T + 3) // 4) * 4
            qp = torch.zeros((padded, 3072), device=x.device, dtype=qx.dtype)
            qp[:T].copy_(qx)
            sp = torch.zeros((24, padded), device=x.device).T
            sp[:T].copy_(sx)
            pq = dict(args, a1_q=qp, a1_scale=sp)
            emit(
                dict(
                    event="prequant",
                    T=T,
                    routing=mode,
                    **check(fused_experts(**pq), sgl(**pq))
                )
            )
            pg, po = capture(lambda: fused_experts(**pq))
            qp.zero_()
            pg.replay()
            torch.cuda.synchronize()
            emit(
                dict(
                    event="prequant_graph_mutation",
                    T=T,
                    routing=mode,
                    **check(po, sgl(**pq))
                )
            )
            ip = dict(
                args,
                hidden_states=x.clone(),
                moe_runner_config=replace(cfg, inplace=True),
            )
            ipout = fused_experts(**ip)
            assert ipout.data_ptr() == ip["hidden_states"].data_ptr()
            emit(dict(event="inplace", T=T, routing=mode, **check(ipout, ref)))
            x.zero_()
            graphs["sonic_fp8"][0].replay()
            torch.cuda.synchronize()
            emit(
                dict(
                    event="zero_graph",
                    T=T,
                    routing=mode,
                    **check(graphs["sonic_fp8"][1], sgl(**args))
                )
            )
            del pg, po, ipout
        del graphs, g, out, y, ref
    torch.distributed.destroy_process_group()
    emit(dict(event="completed"))


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
