# Hopper FP8 MoE：实验记录与优化过程

记录日期：2026-09-28 至 2026-09-29。所有 GPU 实验仅在同一台机器的物理 GPU7 上执行。
本记录整理从真实 FP8 主循环移植到输入 scatter 优化的过程；更早的 BF16 路径调查不混入这里的 FP8 性能表。

## 1. 当前结论与适用范围

最终候选保留：真实 QuACK FP8 WGMMA、分块 scale 乘积复用、输入量化一次后按 top-k 分发、固定 top-k metadata、
Gate/Up+SwiGLU 融合、独立激活 FP8 量化、直接 combine、简化的 Gate/metadata 阈值，以及多 group 输入 scatter。

最新 scatter 在 T=16384 均匀路由时独立阶段约减少 40% 延迟；完整 MoE 的长 A/B 减少 2.17%。
hot/skew 大 batch 的整层延迟分别减少 2.91%/1.26%。小 batch 基本持平，T=1 约慢 0.8–1.0 µs。
这是组件级实验结果，不代表所有 MoE、路由分布或完整推理服务都获益。

| 条件 | 本实验 |
|---|---|
| GPU | 单 NVIDIA H100 80GB HBM3，SM90，132 SM；物理 GPU7 |
| 模型组件 | H=3072，单 expert 中间维度 I=1024，E=256，top-k=8 |
| 并行/范围 | TP1、EP1、routed experts only；不测 shared expert |
| 数据类型 | BF16 输入/输出；FP8 E4M3 权重和激活；FP32 scales |
| 量化契约 | 权重 block128×128；激活每 token/group128 动态量化 |
| 布局/激活 | concat Gate/Up；SwiGLU；标准 combine |
| 环境 | Python 3.12，Torch 2.13.0+cu130，Triton 3.7.1，QuACK 0.6.4，CUTLASS DSL 4.6.2 |
| 对照 | SGLang 0.5.20 的 SGL Triton FP8；sgl-kernel 0.4.7 |
| 限制 | 合成权重/输入；无完整 checkpoint 精度、TP4、EP、服务吞吐或 backward 结论 |

源代码提交和其他环境信息见 [environment.json](environment.json)。SGL 对照使用测试环境的默认 kernel config，
该 shape 缺少相应调优配置；未穷尽 SGL 调优。最终 scatter 实验没有重新测第三方 Triton-kernels 或 DeepGEMM，
不能把这里的 SGL Triton 列扩展成对这些路径的结论，也不能直接套用 SonicMoE 的 Blackwell/BF16 宣传结果。

## 2. 最终实现路径

令 T 为传入该层的 token 数，R=T×8 为路由展开后的行数。T=16384 时 R=131072；这里的 T 不是请求数。

```text
BF16 X[T,H] + top-k IDs/weights[T,8]
  │
  ├─ metadata：expert offsets、gather、reverse、sorted weights
  │
  ├─ 输入 quant + multi-group scatter
  │    每个 token/group 量化一次，按 reverse 分发至 8 个 expert 行
  │    FP8 QX[R,H] + FP32 SX[R,H/128]
  │
  ├─ QuACK FP8 Gate/Up WGMMA + BF16 舍入 + SwiGLU
  │    BF16 A[R,I]
  │
  ├─ 独立 SGL group FP8 quant
  │    FP8 QA[R,I] + FP32 SA[R,I/128]
  │
  ├─ QuACK FP8 Down WGMMA
  │    FP32 累加结果先乘路由权重，再转换 BF16
  │    BF16 Y[R,H]
  │
  └─ reverse-map combine：按原始 top-k slot 顺序 FP32 求和
       BF16 OUT[T,H]
```

一般是 3 个 metadata kernel 加 5 个后续 kernel；小 metadata 路径缩成 1+5。
量化输入已提供时直接分发 FP8 bytes 和 scales，支持 padded 行与列主序 scale。
两个 GEMM 的 K 维每 128 元素计算一次 FP32 partial accumulator，然后执行
`total += partial * (activation_scale * weight_scale)`。
Gate 有 24 个 K block，Down 有 8 个。不能先做完全部 K 的一个 FP8 GEMM 再乘一次 scale，
那不满足当前 block-FP8 契约。WGMMA partial 和总和有各自的 accumulator，读取 partial 前等待相关 WGMMA 完成。

这套路径与 SGL 的数值顺序对齐：Gate/Up 在 SwiGLU 前按 BF16 舍入，Down 路由权重在转换 BF16 前应用，
combine 按固定 slot 顺序累加。相同签名仅表示可对照调用，不代表支持 SGL 的所有配置。

## 3. 优化过程与取舍

### 3.1 真实 FP8 主循环移植

最初将 SGL 的 `[128,128]` 权重尺度与 `1x128` 激活尺度接入 QuACK SM90 WGMMA，
实现每 K block 的缩放与累加，并对齐 Gate/Up concat 布局和 epilogue 舍入。
保留 SonicMoE 的 grouped GEMM/varlen expert 组织方式；没有通过转回 BF16 GEMM 或调用 SGL GEMM 冒充 FP8。

入口测试会将 SGL `invoke_fused_moe_kernel` 替换为抛异常，再执行候选；这证明所测 eager 路径没有该 GEMM 回退。

### 3.2 首轮结构优化：从可运行到有收益

| 修改 | 原因与处理 | 最终选择 |
|---|---|---|
| scale 乘积复用 | 同一 row/column 的 sA×sB 先计算并复用，减少 accumulator 上重复运算，同时匹配参考结合顺序 | 保留 |
| 预先 quant+scatter | 输入每 token/group 只量化一次；expert-packed 输入让 Gate 使用非 gather TMA 加载 | 保留 |
| 固定 top-k metadata | 替换通用路由和多次 PyTorch 索引；稳定排序、计数、并行前缀和、reverse/score 写出 | 保留 |
| 大小 batch tile | 初版以 T=4096 分界，64×128/ping-pong 与 128×128/非 ping-pong | 后续只提前 Gate 阈值 |
| 直接 combine | 用 reverse map 按原 top-k slot 累加，取消额外 ones 权重/token offsets 构造 | 保留 |
| 每条路由重复量化 | 对相同输入重复读入和量化 | 放弃 |
| 单 CTA 整个前缀矩阵 | 大 batch 工作量集中 | 放弃，改为按 expert 并行 |
| 一律降低 pipeline stage | 中小 batch 退化 | 放弃，保留 QuACK 默认 stage 计算 |
| Gate N=256 常规 tile | 所测形状明显变慢 | 不用于默认 Gate |

首轮权威数据：[optimized-final.jsonl](results/optimized/optimized-final.jsonl)、
[optimized-repeat.jsonl](results/optimized/optimized-repeat.jsonl)。当时 CUDA Graph 采用 7×10 replay，
与后续 9×30/100 的实验方法不同，跨阶段绝对耗时不能当作严格配对归因。

| T / 路由 | SGL ms | 首轮优化 ms | 同轮延迟降低 |
|---|---:|---:|---:|
| 1 / uniform | 0.042755 | 0.049882 | -16.67% |
| 1024 / uniform | 0.9314 | 0.9144 | 约 1.8% |
| 2048 / uniform | 1.1030 | 1.0329 | 约 6.4% |
| 4096 / uniform | 1.6482 | 1.4300 | 约 13.2% |
| 8192 / uniform | 2.7073 | 2.3554 | 约 13.0% |
| 16384 / uniform | 4.8411 | 4.3720 | 约 9.7% |
| 1025 / hot | 0.3000 | 0.3453 | 约 -15.1% |

### 3.3 激活 FP8 量化融合：实现了，但默认不采用

已实现 Gate/Up + BF16 舍入 + SwiGLU + group amax + FP8 转换 + scale 写出的一体 epilogue。
配对 accumulator 统计预遍历处理 256 个 Gate/Up 列，得到 128 个激活元素的 amax；
64×256 tile 由两个 N 方向 warp-group 分担，使用专用 direct-store 写出 FP8，省去 BF16 激活中间张量。

初版单 warp-group 过慢，拆 N 后曾遇到通用 gated store 布局错误，最终以专用 store 修正。
实现保留在 [fused_gate_quant.py](fused_gate_quant.py)，`FUSED_QUANT=1` 才启用。

| T / 路由 | 相对不融合方案，融合延迟变化（两轮） |
|---|---:|
| 1024 / uniform | 约持平 |
| 2048 / uniform | 慢 5.3%–6.6% |
| 8192 / uniform | 慢 3.7%–5.0% |
| 16384 / uniform | 慢 5.8%–7.0% |
| 1025 / hot | 快 8.3%–8.4% |

来源：[fusion-paired.jsonl](results/fusion/fusion-paired.jsonl)、
[fusion-repeat.jsonl](results/fusion/fusion-repeat.jsonl)。默认关闭后的回归另见
[default-unfused-final.jsonl](results/fusion/default-unfused-final.jsonl)。

新增归约/同步、更宽 tile 与 warp-group 布局变化可能抵消节省的 launch/内存流量；这是基于实现的解释，
未用硬件计数器拆分成本。最终选择独立量化，未根据单个 hot case 增加自动分支。
本次核对的 SGL 0.5.20 安装实现中，SwiGLU 和激活 FP8 量化本身也是分开的 kernel。

### 3.4 五个方向的筛选与固定候选

| 方向 | 实验与结果 | 处理 |
|---|---|---|
| FP8 主循环流水线 | 将 scale 加载放入异步 WGMMA issue 与 wait 之间，正确但复测无稳定整层收益 | 不保留 scale-prefetch |
| Gate/Down tile 分开调优 | N=64 不佳；Gate M=192 在部分 batch 回退；Down M=192 的收益依赖 batch/路由 | 不采用复杂 M=192 调度 |
| scatter/combine 简单参数 | 单纯 rows/block/warps 扫描无可靠整层收益 | 保留原方案，另做下一节结构实验 |
| metadata crossover | 单 CTA 上限从 2048 条路由减到 512，T=192/256 改善约 3% | 保留 |
| 独立激活量化替换 | 自写 row/warp tiling 未稳定优于当前 SGL quant | 保留 SGL quant |

筛选数据在 [results/tuning](results/tuning/)，其中 p1/p2/p3/p4/p5 对应上述顺序。
阶段级筛选不能替代整层 A/B；`full-metadata.jsonl`、`full-tiles.jsonl` 是整层验证。
这些日志属于历史候选，不是本分支默认设置。

根据简化配置的要求，冻结版本仅保留两个策略改动：

1. 验证形状下 Gate 使用 M=128 的阈值从 T=4096 提前到 T=2048。
2. 验证形状下单 CTA metadata 的 R 上限从 2048 改为 512。

Down 不变；不添加 M=192、scale-prefetch 或新独立量化 kernel。
[stable-regression.jsonl](results/stable/stable-regression.jsonl) 的同轮结果中，
T=3072 从 1.364959 ms 降至 1.205927 ms（11.65%），T=4095 从 1.700017 ms 降至 1.458678 ms（14.20%）。
T=192 从 0.855114 ms 降至 0.830682 ms，T=256 从 0.872928 ms 降至 0.848365 ms。
这些是该冻结回归轮次的观测；未改变执行路径的点也有小幅波动。

### 3.5 输入 scatter 的结构性实验：当前新增优化

目标是减少 scatter 的 CTA 数量和重复地址/路由索引工作，不改 FP8 数学、expert-packed 布局、metadata、GEMM 或 combine。

| 项目 | 冻结 scatter | 新 scatter |
|---|---|---|
| 每 CTA 的逻辑工作 | 4 tokens × 1 group | 1 token × 32 groups |
| Group 元素数 | 128 | 128 |
| H=3072 有效 groups | 24 | 24（其余 8 mask） |
| T=16384 CTA 数 | ceil(T/4)×24 = 98304 | T = 16384 |
| 路由/scale 访问 | group 间分开处理 | 同一 token 的 reverse 索引供多 group 使用；scale 集中写一行 |
| 每 token 的 expert 写出 | top-k 8 份 | 仍然 8 份 |

候选固定为 rows=1/groups=32/warps=8，所有 T 使用同一配置。
首轮筛了 7 个方案（包含原 scatter），随后只保留基线和该候选做反序复测。
代码见 [scatter_candidate.py](scatter_candidate.py)。

首次完整测试在预量化分支遇到 masked `tl.load` 的整数 `other=0` 无法转 FP8 的编译错误；
改成 `other=0.0` 后重新完成测试。失败运行未混入下表，权威首轮文件名带 `forward-fixed`。
没有硬件计数器证据，因此不把收益精确归因于 cache/TLB/带宽中的某一项。

## 4. 最新 scatter 结果

延迟降低 = `(baseline - candidate) / baseline × 100%`；正值更快，负值更慢。
以下基线都已经包含上一节的 Gate/metadata 冻结策略，只替换 scatter，避免把前一轮收益重复计入。

### 独立 scatter

CUDA Graph，9 组×50 replay，中位数；最终复测反转捕获顺序。
来源：[scatter-selected-repeat.jsonl](results/scatter/scatter-selected-repeat.jsonl)。

| T | Routing | 基线 ms | 新 scatter ms | 延迟降低 |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.003978 | 0.003990 | -0.31% |
| 256 | uniform | 0.004645 | 0.004012 | +13.63% |
| 1024 | uniform | 0.011943 | 0.009373 | +21.52% |
| 4096 | uniform | 0.051652 | 0.048330 | +6.43% |
| 8192 | uniform | 0.105393 | 0.095047 | +9.82% |
| 16384 | uniform | 0.311385 | 0.188004 | +39.62% |
| 16384 | hot | 0.313743 | 0.182938 | +41.69% |
| 16384 | skew | 0.311951 | 0.186518 | +40.21% |

### 完整 MoE

两轮三路图计时（SGL、冻结版、候选），9 组×30 replay，捕获顺序相反。
第三轮只计时冻结版和候选，9 组×100 replay；SGL 仍作正确性参考。

来源：[第一轮](results/scatter/scatter-moe-forward-fixed.jsonl)、
[反序第二轮](results/scatter/scatter-moe-reverse.jsonl)、
[长双路 A/B](results/scatter/scatter-moe-paired-long.jsonl)。

| T | Routing | 第一轮降低 | 第二轮降低 | 长 A/B 降低 | 长 A/B 基线 / 候选 ms |
|---:|---|---:|---:|---:|---|
| 1 | uniform | -2.05% | -1.57% | — | — |
| 32 | uniform | +0.00% | +0.47% | — | — |
| 64 | uniform | +0.15% | +0.24% | — | — |
| 128 | uniform | +0.24% | -0.16% | — | — |
| 256 | uniform | -0.15% | -0.05% | — | — |
| 512 | uniform | -0.06% | +0.07% | — | — |
| 1024 | uniform | +0.19% | +0.26% | +0.38% | 0.914976 / 0.911464 |
| 2048 | uniform | +0.25% | +0.95% | — | — |
| 3072 | uniform | +0.26% | +0.75% | — | — |
| 4096 | uniform | +0.20% | +1.15% | -0.97% | 1.468169 / 1.482346 |
| 8192 | uniform | +2.66% | -0.50% | +3.12% | 2.494188 / 2.416488 |
| 12288 | uniform | +4.02% | +1.44% | +1.53% | 3.439429 / 3.386796 |
| 16384 | uniform | +5.85% | +1.54% | +2.17% | 4.450518 / 4.354106 |
| 1025 | hot | +0.55% | +0.70% | — | — |
| 4097 | skew | +0.13% | +3.36% | — | — |
| 16384 | hot | +1.45% | +2.54% | +2.91% | 3.999611 / 3.883293 |
| 16384 | skew | +2.05% | +1.78% | +1.26% | 4.655095 / 4.596221 |

不能只报告最好一轮：T=16384 均匀路由前两轮差异为 5.85%/1.54%，较长 A/B 为 2.17%。
T=8192 第二轮小幅回退，T=4096 长 A/B 回退约 0.97%；小差值可能受时钟、cache 和 graph 内存布局影响。
T=1 的轻微回退也保留，没有为单一 batch 再增加 dispatch 分支。

## 5. 正确性与测量契约

权重 seed=1734，w1/w2 分别除以 sqrt(H)/sqrt(I)；输入 seed=1000+T、BF16 随机数×0.2。
路由从随机 logits 取 top-8 后 softmax。uniform 使用原 logits；hot 给前 8 experts 加 100；
skew 给前 16 experts 加 2。专家选择在计时之外，metadata 在计时之内。

完整 MoE 计时包括 metadata、输入量化/重排、两次 GEMM、SwiGLU、激活量化、combine；
不包括权重量化、top-k 选择、模型其他层或通信。3 次 eager 预热，捕获后再做 20 次 graph 预热，
每组轮换/反转 backend 计时顺序；报告各组 replay 平均值的中位数，原始 samples 全部保留。
没有统计显著性检验，也没有硬件计数器分析。

| 验证阶段 | 显式误差记录 | 实测 relative L2 / max absolute error |
|---|---:|---|
| 首轮优化两轮 | 174 | 全为 0 |
| 激活量化融合两轮 | 174 | 全为 0 |
| 融合关闭回归 | 48 | 全为 0 |
| 冻结 Gate/metadata 策略 | 96 | 全为 0 |
| scatter 两轮 + 长 A/B | 218 | 全为 0 |

上述计数只计 JSONL 中带 `rel_l2` 和 `max_abs` 的输出检查；各 backend 的图输出另外有断言。
这些相等结论仅适用于所测合成输入，不能保证任意输入或完整模型均 bitwise 一致。

覆盖 eager、padded 预量化及列主序 scale、inplace、图捕获后输入和路由内容变化、零输入。
`test_scatter.py` 另测 24 个 shape/幅度组合：T 的 63/64/65 与 255/256/257 边界、H=384 的非完整 group tile、
1e-12/1/1e4 幅度和零 group；直接比较 FP8 bytes 与 FP32 scales，并覆盖预量化和图内修改。

定向 scatter compute-sanitizer：12 cases，**0 device memory errors**，见
[scatter-memcheck.log](results/scatter/scatter-memcheck.log)。命令使用 `--report-api-errors no` 过滤本机已知
`cuGetProcAddress_v2` 驱动入口查询错误，设备内存检查保持启用；不应描述为未过滤 API 错误的全模型 memcheck。

## 6. 发布整理与复现

实验原目录曾由 benchmark 进程内替换 scatter 来测试候选；本分支已把候选直接接到 `sonic_fp8.py` 默认入口。
完整 runner 会断言该入口确实引用 `pack_scatter_groups`，不再靠注入才能启用；旧 scatter 仍作为 `stable` 对照。
数值 kernel 保持本轮候选，增加的代码仅负责从本仓库选择依赖并拒绝混用已导入的外部版本。

修改的 QuACK 0.6.4 源码随目录保存，完整差异和来源见 [vendor/README.md](vendor/README.md)。
原始 JSONL 保持原内容，因此里面的 `quack_source` 记录的是当时实验绝对路径；发布回归会记录新目录。
历史性能原始数据足以审计表格，但已放弃候选并未全部包装成可直接运行的独立版本。

发布入口额外回归记录在 [results/release](results/release/)，验证内容和结果见该目录说明。
运行方法见 [README.md](README.md#运行)；所有脚本固定 GPU7，并把新输出写到独立 `runs/`。

## 7. 仍有优化空间

- 大 batch scatter 已有局部收益，但 Gate/Down 仍占主要时间；不能期待继续减少 scatter launch 带来同比例整层加速。
- Down 输出布局调整可能改善 combine 的局部性，也可能损害现有 TMA store；当前尚未实施，不能预先计算收益。
- FP8 主循环进一步 overlap 需要同时考虑 accumulator、scale 缓冲与 WGMMA 同步；简单提前加载已试过，未稳定获益。
- 当前目标是少量固定策略的可复现实验实现。更多 shape、真实权重/路由分布、完整模型与 TP/EP 集成均需独立验证。
