# SonicMoE / QuACK Hopper FP8 实验实现

2026-09-29。单 H100、SGLang 0.5.20 接口的组件级 FP8 forward，包含最新输入 scatter 优化。
这是本分支的实验入口；上游 `sonicmoe` 的常规安装/API 保持原样。

**默认路径：多 group 输入 quant+scatter → QuACK Gate/Up + SwiGLU → 独立激活 FP8 量化 → QuACK Down → combine。**
两个 GEMM 都运行真正的 SM90 FP8 WGMMA；SGL 仅提供数据结构、独立量化 kernel 和测试参考。

## 文档与结果

- [实验记录和优化过程](EXPERIMENTS.md)：移植、保留/放弃的优化、逐 batch 结果及限制。
- [环境版本](environment.json)、[第三方源码来源与差异](vendor/README.md)。
- [scatter 原始结果](results/scatter/)、[冻结策略回归](results/stable/)、[融合实验](results/fusion/)、[首轮优化](results/optimized/)。
- [发布目录回归](results/release/)：直接验证本目录默认入口，不在 benchmark 中替换成候选。

本轮长 A/B 中，T=16384 均匀路由的完整 MoE 从 4.450518 ms 降至 4.354106 ms，延迟降低 **2.17%**。
独立 scatter 从 0.311385 ms 降至 0.188004 ms，降低 **39.62%**。
小 batch 无稳定整层收益；T=1 约慢 0.8–1.0 µs。不能将 scatter 的局部收益当作整层收益。

## 支持范围

验证形状：`H=3072, I=1024, E=256, top_k=8`，TP1/EP1，仅 routed experts，无 shared expert。
BF16 输入/输出；FP8 E4M3 权重/激活；FP32 scales：权重 `[128,128]` block、激活 `1x128` group。
Gate/Up 按 concat 布局，SwiGLU，标准 combine，不支持 bias、其他量化模式、EP 或训练反向。
使用合成权重和输入，未验证完整 checkpoint 精度或服务吞吐。

`sonic_fp8.fused_experts` 的 Python 签名与所测 SGL Triton `fused_experts` 相同；支持模式是上述子集。
调用者必须提供合法且与设备一致的权重、scale 和 top-k IDs。它不是自动注册到 SGLang 的生产 backend。

## 当前固定策略

| 项目 | 默认配置 |
|---|---|
| 输入 scatter | 所有 T：每 CTA 1 token × 32 groups，每 group 128 元素，8 warps |
| Gate（验证形状） | T < 2048：64×128、ping-pong；否则 128×128、非 ping-pong |
| Down | T < 4096：64×128、ping-pong；否则 128×128、非 ping-pong |
| Metadata（验证形状） | T × top_k ≤ 512：单 CTA；否则并行路径 |
| Gate/Up + SwiGLU | 融合 |
| 激活 FP8 量化 | 独立 SGL kernel；`FUSED_QUANT=0` |
| FP8 主循环 | K=128 分块缩放、复用 scale 乘积；无实验 scale-prefetch |

Gate/metadata 的新阈值仅在验证形状启用；其他形状未经系统性能验证。
`OPT_POLICY=0` 可恢复更早的 Gate/metadata 阈值，**不会关闭新 scatter**。
旧 scatter 保存在 `optimized_ops.pack_scatter`，完整 A/B runner 的 `stable` 路径用它，保持其他设置完全相同。
`FUSED_QUANT=1` 仅用于重访已实现但未采纳的融合实验；不能把它的性能算作本分支默认结果。

## 运行

建议复用已验证环境；精确版本和 SGL 源码提交见 [environment.json](environment.json)。
需要 Python 3.12、CUDA 13 系列 PyTorch、Triton、CUTLASS DSL，以及该提交的 SGLang/sgl-kernel。
本仓库包含修改后的 QuACK Python 源码，但并未打包 Torch/CUTLASS/SGL 的二进制依赖；仅执行 `pip install sonic-moe` 不足以复现。

所有附带测试/benchmark 固定使用**物理 GPU7**，先确认该卡空闲。首轮会 JIT 编译。

```bash
cd experiments/hopper_fp8
# 默认解释器 /opt/sglang/bin/python；可用 PYTHON=/path/to/python 指定兼容环境。
bash reproduce.sh smoke       # scatter 正确性 + 12 个 MoE 回归 case
bash reproduce.sh full        # 两种捕获顺序的全 batch A/B
bash reproduce.sh long        # 双路 9×100 replay 的大 batch 复测
bash reproduce.sh scatter     # 仅独立 scatter 的最终候选 A/B
bash reproduce.sh memcheck    # 定向 scatter 内存检查，需要 compute-sanitizer
```

新输出写入 `runs/<时间>-<PID>/`，历史 `results/` 不会被覆盖。
跑完整流程无需修改安装库。`_bootstrap.py` 让新进程优先使用 `vendor/quack` 和本仓库 `sonicmoe`；
如果已从别处导入同名库会明确报错，需启动新 Python 进程，避免混用实现。

在本目录的兼容环境里可直接 `from sonic_fp8 import fused_experts`。该入口默认已经选择
`scatter_candidate.pack_scatter_groups`，无需 monkey patch。脚本内部临时切换旧 scatter 只用于单线程 A/B 捕获，
不应照搬到并发服务中。

## 文件导航

| 文件 | 作用 |
|---|---|
| `sonic_fp8.py` | SGL 签名入口、范围检查、固定策略和整层调用 |
| `scatter_candidate.py` | 新输入多 group quant+scatter，默认 r1/g32/w8 |
| `optimized_ops.py` | 固定 top-k metadata、旧 scatter、gather pack、combine |
| `quack_fp8.py` | 两个真实 FP8 GEMM 的编译/调用与 BF16 舍入契约 |
| `vendor/quack/gemm_sm90.py` | K=128 FP8 分块缩放 WGMMA 主循环 |
| `fused_gate_quant.py` | 可选融合激活量化 epilogue，默认关闭 |
| `pointwise_quant.py` | 关闭 Gate 激活融合时的历史备用实现 |
| `bench_moe_scatter.py` | SGL / 冻结 scatter / 新 scatter 完整 MoE 对照 |
| `bench_scatter.py`, `test_scatter.py` | 独立阶段计时和 FP8 bytes/scales 精确检查 |
| `SHA256SUMS` | 发布实验源码、文档和结果的完整性清单 |

目录内运行 `sha256sum -c SHA256SUMS` 可校验发布快照。额外生成的 `runs/` 不参与清单。
