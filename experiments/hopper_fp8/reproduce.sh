#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
unset PACK_MODE COMBINE_B SCALE_PRODUCT AB_STAGE_CAP PACKED FAST_METADATA FUSED_GATE AUTO_TILE TUNE PINGPONG Q_TILE_M Q_PINGPONG CAPTURE_ORDER EXTRA_CASES BENCH_REPLAYS PAIR_ONLY SELECTED_ONLY PROFILE_STAGES SANITIZE EDGES TOKENS
export CUDA_VISIBLE_DEVICES=7 OPT_POLICY=1 FUSED_QUANT=0
PYTHON="${PYTHON:-/opt/sglang/bin/python}"
mode="${1:-smoke}"
run_dir="runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir"
run_moe() {
    export RESULT_FILE="$run_dir/$1.jsonl"
    "$PYTHON" bench_moe_scatter.py 2>&1 | tee "$run_dir/$1.log"
}
case "$mode" in
    smoke)
        "$PYTHON" test_scatter.py 2>&1 | tee "$run_dir/unit.log"
        export TOKENS=1,64,65,2047,2048,4095,4096,16384 EDGES=1
        export EXTRA_CASES='[["hot",16384],["skew",16384]]'
        run_moe smoke
        ;;
    full)
        "$PYTHON" test_scatter.py 2>&1 | tee "$run_dir/unit.log"
        export TOKENS=1,32,64,128,256,512,1024,2048,3072,4096,8192,12288,16384 EDGES=1
        export EXTRA_CASES='[["hot",16384],["skew",16384]]'
        run_moe forward
        export CAPTURE_ORDER=reverse
        run_moe reverse
        ;;
    long)
        export TOKENS=1024,4096,8192,12288,16384 PAIR_ONLY=1 BENCH_REPLAYS=100
        export EXTRA_CASES='[["hot",16384],["skew",16384]]'
        run_moe paired-long
        ;;
    scatter)
        export SELECTED_ONLY=1 CAPTURE_ORDER=reverse RESULT_FILE="$run_dir/scatter.jsonl"
        "$PYTHON" bench_scatter.py 2>&1 | tee "$run_dir/scatter.log"
        ;;
    memcheck)
        export SANITIZE=1
        "${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}" \
            --tool memcheck --report-api-errors no --error-exitcode 99 \
            "$PYTHON" test_scatter.py 2>&1 | tee "$run_dir/memcheck.log"
        ;;
    *) echo 'Usage: reproduce.sh {smoke|full|long|scatter|memcheck}' >&2; exit 2 ;;
esac
