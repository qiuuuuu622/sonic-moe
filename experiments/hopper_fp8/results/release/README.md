# 发布目录回归（2026-09-29）

从本分支独立目录启动新 Python 进程，仅使用物理 GPU7。
与历史 scatter benchmark 的区别：本次不注入候选，直接检查并运行 `sonic_fp8.py` 的默认 scatter 入口。

- `unit.log`：24 个 scatter shape/幅度组合，普通输入、padded 预量化/列主序 scale、CUDA Graph 修改均通过 FP8 bytes/scales 精确比较。
- `default-entry.jsonl`：12 个完整 MoE case，72 条显式误差记录；relative L2 与 max absolute error 全为 0。
- `default-entry.log`：完整 stdout/stderr；SGL 缺少该 shape 调优配置的 warning 保留。
- 每个 eager case 禁止 SGL Triton GEMM 调用，候选仍通过；三个 backend 的 CUDA Graph 输出另有断言。
- `start` 记录确认 `fused_quant=false`、`optimized_policy=true`，`quack_source` 指向发布目录的 `vendor/quack/gemm_sm90.py`。
- 测试结束后 GPU7 为 0 MiB / 0% utilization，没有遗留 GPU 任务。

Uniform T：1、64、65、2047、2048、4095、4096、16384。
附加路由：hot 1025、skew 4097、hot 16384、skew 16384。
因此覆盖 metadata、Gate、Down 阈值边界及大 batch；每个 case 包括 eager、图内输入/路由变更、预量化、
预量化图内修改、inplace、零输入。计时采用 9×30 replay，此轮主要用于发布入口集成回归。
历史受控 scatter 长 A/B 的结果仍以 `../scatter/scatter-moe-paired-long.jsonl` 为准。

复现相同覆盖：在实验目录运行 `bash reproduce.sh smoke`。解释器可用 `PYTHON` 指定。
