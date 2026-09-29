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


## 最新版 vs SGL Triton：完整 MoE 性能对比

这里的**最新版**指本分支包含新 scatter 的默认实现：`OPT_POLICY=1`、`FUSED_QUANT=0`，
Gate 从 T=2048 起使用 M=128，Down 从 T=4096 起使用 M=128，scatter 固定 rows=1/groups=32/warps=8。
对照是 **SGLang 0.5.20 的 SGL Triton FP8 `fused_experts`**，不是第三方 Triton-kernels。

两轮完整对照中，均匀路由 **T=3072–16384 的整层延迟降低约 11.4%–14.0%**；
T=2048 降低约 6.0%–6.2%，T=1024 降低约 2.1%–2.2%。
小 batch 收益较小，**T=1 慢约 18.2%–18.8%，hot T=1025 慢约 14.4%–14.5%**。
这些比例是相对 SGL 的完整 MoE 延迟变化；下文另列的 scatter 增量收益以旧 scatter 版本为基线。

### 测试条件与计时范围

| 项目 | 设置 |
|---|---|
| 硬件 | 单 H100 80GB HBM3，物理 GPU7 |
| 组件形状 | H=3072、I=1024、E=256、top-k=8；TP1/EP1，无 shared expert |
| 数据与量化 | BF16 输入/输出；FP8 E4M3 权重/激活；FP32 scales，权重 block128×128、激活 group1×128 |
| 输入 | 同进程、同一组合成权重/输入/路由；未加载完整 checkpoint |
| 对照配置 | SGL 当前环境的默认 kernel config；缺少该 shape 的调优配置，未穷尽调优 |
| 计时包含 | metadata、输入量化/重排、Gate/Up、SwiGLU、激活 FP8 量化、Down、combine |
| 计时不包含 | top-k 选择、权重量化、模型其他层、通信、JIT 编译与预热 |
| 计时方法 | CUDA Graph；3 次 eager 预热、捕获后 20 次 graph 预热；9 组×30 replay，报告各组平均延迟的中位数 |
| 对照顺序 | SGL / 旧 scatter / 最新版三路交替计时；第 2 轮反转捕获顺序；下表抽取同轮 SGL 与最新版 |

T 是传入该 MoE 层的 token 数；路由展开后为 R=T×8 行，不是请求数。
uniform 使用随机 logits 的 top-8；hot 给前 8 个 expert 的 logits 加 100；skew 给前 16 个加 2。
三种路由都对 top-8 logits 做 softmax 得到权重。原始样本、环境和种子见 [实验记录](EXPERIMENTS.md)。

### 全 batch、全路由对照（两轮）

**延迟降低 = (SGL ms − 最新版 ms) / SGL ms × 100%**，正值表示最新版更快，负值表示更慢。
所有延迟单位都是 ms。保留两轮结果，不取两轮中的最好值，也不把不同轮次的 SGL/最新版拼成一对。

| T | 路由 | SGL 第 1 轮 ms | 最新版第 1 轮 ms | 延迟降低 | SGL 第 2 轮 ms | 最新版第 2 轮 ms | 延迟降低 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | uniform | 0.042457 | 0.050455 | -18.84% | 0.042521 | 0.050265 | -18.21% |
| 32 | uniform | 0.530227 | 0.525913 | +0.81% | 0.530105 | 0.524667 | +1.03% |
| 64 | uniform | 0.712986 | 0.706205 | +0.95% | 0.712964 | 0.705577 | +1.04% |
| 128 | uniform | 0.812465 | 0.806241 | +0.77% | 0.812228 | 0.809079 | +0.39% |
| 256 | uniform | 0.852995 | 0.849728 | +0.38% | 0.852621 | 0.847466 | +0.60% |
| 512 | uniform | 0.883872 | 0.868292 | +1.76% | 0.883426 | 0.866906 | +1.87% |
| 1024 | uniform | 0.932128 | 0.913047 | +2.05% | 0.931629 | 0.911504 | +2.16% |
| 2048 | uniform | 1.116327 | 1.049241 | +6.01% | 1.116532 | 1.047564 | +6.18% |
| 3072 | uniform | 1.366711 | 1.193913 | +12.64% | 1.369023 | 1.198926 | +12.42% |
| 4096 | uniform | 1.656717 | 1.456008 | +12.11% | 1.658895 | 1.453478 | +12.38% |
| 8192 | uniform | 2.696737 | 2.369182 | +12.15% | 2.770437 | 2.453716 | +11.43% |
| 12288 | uniform | 3.798667 | 3.268210 | +13.96% | 3.849887 | 3.405460 | +11.54% |
| 16384 | uniform | 4.850947 | 4.267191 | +12.03% | 4.952907 | 4.376499 | +11.64% |
| 1025 | hot | 0.302394 | 0.346176 | -14.48% | 0.302997 | 0.346778 | -14.45% |
| 4097 | skew | 1.706003 | 1.666852 | +2.29% | 1.711611 | 1.666525 | +2.63% |
| 16384 | hot | 4.483763 | 3.845080 | +14.24% | 4.557934 | 3.906791 | +14.29% |
| 16384 | skew | 4.971587 | 4.465733 | +10.17% | 5.064814 | 4.575582 | +9.66% |

原始数据：[第 1 轮](results/scatter/scatter-moe-forward-fixed.jsonl)、
[第 2 轮（反序捕获）](results/scatter/scatter-moe-reverse.jsonl)。
这两份文件里的 `ms.sgl` 与 `ms.sonic_fp8` 对应上表两条路径；`ms.stable` 是旧 scatter 对照。
两轮共 34 个 case、204 条显式误差检查，relative L2 和 max absolute error 全为 0。
逐元素相等仅限所测输入，不表示任意模型和输入都具有相同精度。

### 发布目录默认入口回归（含阈值边界）

发布前另从独立目录直接调用默认入口，确认新 scatter 和随仓库保存的 QuACK 实际生效。
以下是该轮的全部 12 个 case，计时方法同为 9×30 replay；它补充了 T=65、2047、4095 等边界，
不能与上表其他轮次的耗时交叉计算收益。对应实现提交为 `6a4e087`，本次 README 更新不修改 kernel。

| T | 路由 | SGL Triton ms | 最新版默认入口 ms | 延迟降低 |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.042405 | 0.050410 | -18.88% |
| 64 | uniform | 0.712667 | 0.705938 | +0.94% |
| 65 | uniform | 0.680517 | 0.679900 | +0.09% |
| 2047 | uniform | 1.112910 | 1.054420 | +5.26% |
| 2048 | uniform | 1.115954 | 1.046575 | +6.22% |
| 4095 | uniform | 1.651717 | 1.467825 | +11.13% |
| 4096 | uniform | 1.658011 | 1.457053 | +12.12% |
| 16384 | uniform | 4.950990 | 4.337143 | +12.40% |
| 1025 | hot | 0.303902 | 0.349227 | -14.91% |
| 4097 | skew | 1.716277 | 1.676199 | +2.34% |
| 16384 | hot | 4.544643 | 3.877907 | +14.67% |
| 16384 | skew | 5.060807 | 4.532732 | +10.43% |

来源：[发布入口原始数据](results/release/default-entry.jsonl)、[回归说明](results/release/README.md)。
该轮 72 条显式误差检查全部为 0；每个 eager case 禁止 SGL GEMM 调用后，最新版仍通过。
独立 scatter 的 24 个正确性 case 也全部通过。完整对照可用 `bash reproduce.sh full` 重跑，
发布入口边界覆盖可用 `bash reproduce.sh smoke` 重跑。

以上是单卡合成数据的组件延迟，不是完整模型吞吐或服务端到端性能。
没有硬件计数器归因或统计显著性检验；小幅差值可能受时钟、cache、graph 内存布局影响。
本轮未重新比较 Triton-kernels、DeepGEMM、TP4 或 shared expert。

### 新 scatter 本身的增量收益（相对旧 scatter）

在其余策略完全相同的长 A/B 中，T=16384 均匀路由的完整 MoE 从 4.450518 ms 降至 4.354106 ms，
延迟降低 **2.17%**。独立 scatter 从 0.311385 ms 降至 0.188004 ms，降低 **39.62%**。
这里的基线是旧 scatter 版本，**不是 SGL Triton**；长 A/B 的 SGL 仅用于正确性参考，没有计时。
小 batch 没有明显增量收益；T=1 相对旧 scatter 约慢 0.8–1.0 µs。
来源：[整层长 A/B](results/scatter/scatter-moe-paired-long.jsonl)、
[独立 scatter](results/scatter/scatter-selected-repeat.jsonl)。

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
