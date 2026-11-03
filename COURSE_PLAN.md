# AI Infra 自学课程计划

更新日期：2026-10-17。当前学习入口为本文件。

助手负责学习计划、每课目标、参考资料、前置知识与资源要求；阅读、消化、练习、实验、代码和总结由学习者自行完成。学习进度由学习者自行判断和记录。

## 时间与使用方式

- 24 个学习单元，每个参考两周，按每周 6～8 小时估算；另留四周复习和整理。时间仅为参考，可按实际情况调整。
- 顺序主线：Transformer 基础 → 单机推理 → GPU 与引擎 → 优化 → 分布式 → 平台 → 综合设计。
- 每课目标表示希望最终达到的能力，不是现成答案或操作教程。
- 参考资料按“优先阅读”与“深入阅读”区分，围绕本课主题查阅，无需通读整站。
- 当前实验环境为 Colab。多 GPU、多节点、RDMA 与集群课程的实测取决于后续资源；暂时缺少条件时保留理论学习与待验证事项。

## 阶段一：认识现代推理系统（第 1～8 周）

前置：能阅读基础 Python，了解向量、矩阵形状、矩阵乘和 Softmax 的基本含义。缺少的部分按需补充，张量资料见 R1；本阶段不要求先完成整套机器学习、微积分或 CUDA 课程。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 01 | Transformer 架构与前向计算基础 | 能识别模型主要模块及数据流；理解 Q/K/V、因果 Attention 与下一 Token 预测；区分原始 Encoder–Decoder 与常见 Decoder-only 架构 | 优先 R22、R23、R3；深入 R6 第 3 节 Model Architecture |
| 02 | 生成过程、KV Cache 与请求链路 | 能将第 01 课的计算过程对应到 Prefill/Decode；解释 KV 复用及代价；区分首 Token、后续生成与端到端时间 | 优先 R5、R2 的 Understanding LLM inference；补充 R4 的时间指标 |
| 03 | 推理调度、并发与缓存 | 能区分延迟和吞吐目标；解释批处理、排队、Prefix Cache、Chunked Prefill 的作用与限制 | 优先 R7 的相关 Features；补充 R4 的队列、吞吐和缓存指标 |
| 04 | 推理性能分析基础 | 能定义计时边界、控制变量和重复测量；初步区分计算、显存、带宽与外围服务问题 | 优先 R8；补充 R9 的 GPU 执行概念、R10 的时间线阅读 |

阶段目标：建立从用户请求到模型执行的系统地图，并能合理描述一个性能问题。

## 阶段二：GPU、算子与引擎内部（第 9～16 周）

前置：阶段一；张量形状、数据类型、矩阵乘基础。CUDA 深入部分需要基本 C/C++，可在第 06 课前按需补充。实验条件：可用 CUDA GPU；具体 Profiling 能力以运行环境权限为准。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 05 | GPU 架构与内存层次 | 能说明 SM、Warp、Tensor Core、HBM 与缓存的分工；将访存、计算强度和矩阵规模联系起来 | 优先 R9 的硬件与编程模型；补充 R1 的 Tensor |
| 06 | CUDA 编程基础 | 理解线程组织、同步和内存访问；具备独立实现并验证一个简单 Kernel 的能力 | 优先 R9 的入门、线程层次与内存模型 |
| 07 | Triton 与算子优化 | 能分析小算子的正确性与瓶颈；比较访存布局、分块、融合前后的表现 | 优先 R11 的 Vector Addition、Fused Softmax；Matrix Multiplication 为深入内容 |
| 08 | Attention 优化与推理引擎源码 | 能解释 Attention 的数据搬运问题；在固定版本引擎中定位调度、缓存与 Worker 的调用关系 | 优先 R12；深入 R13；结合 R10 观察执行 |

阶段目标：把模型执行与 GPU 行为联系起来，具备阅读关键源码和分析小算子的能力。

## 阶段三：评估推理优化（第 17～24 周）

前置：阶段一、二；可使用一个固定模型和可重复工作负载。优化方式按硬件和模型支持选择。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 09 | 量化与数值精度 | 能区分权重、激活和 KV 量化；从性能、显存、质量和硬件支持评价一种量化方案 | 优先 R14，选择一种受支持方案深入 |
| 10 | 推测解码 | 能说明草稿、验证、接受率与净收益的关系；辨别适合与不适合的负载 | 优先 R7 的 Speculative Decoding，选择一个方法深入 |
| 11 | MoE 与 Attention 架构差异 | 能区分总参数与激活参数；解释专家路由及 GQA/MLA 对计算、缓存和通信需求的影响 | 优先 R15 的架构部分；选择 MoE、GQA 或 MLA 中一个深入 |
| 12 | 引擎与工作负载比较 | 能为两种引擎设计公平对照；认识长上下文、多模态、Embedding、Agent 多轮或离线任务的不同要求 | 优先 R7、R16；新工作负载选一类深入 |

阶段目标：能够评价优化的适用条件、收益与代价，不以单一速度数字判断方案。

## 阶段四：分布式计算与训练基础设施（第 25～32 周）

前置：GPU 基础与一般网络知识。第 13、14 课的 GPU 通信实测通常需要至少两张 GPU；第 16 课跨节点实测需要相应节点和网络。CPU 分布式练习不作为 GPU 通信性能证据。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 13 | GPU 拓扑与集合通信 | 能区分通信延迟和带宽；解释 PCIe/NVLink、NUMA 与 AllReduce/AllGather 等操作对执行的影响 | 优先 R17 的集合通信与排错；补充 R18 的分布式通信 |
| 14 | TP、PP、DP 与 EP | 能比较不同并行方式如何划分模型、请求或专家；根据显存、通信与服务目标选择策略 | 优先 R7 的 Parallelism and Scaling、Expert Parallel；补充 R18 |
| 15 | 训练、分片与 Checkpoint | 理解前向、反向、梯度与优化器状态；区分 DDP/FSDP；知道恢复训练应保存和验证哪些状态 | 优先 R1 的训练与保存加载、R18；FSDP 为深入内容 |
| 16 | 多节点、RDMA 与扩展性 | 能分析跨节点数据路径、消息规模、拓扑与通信计算重叠；辨别理论推演和实测结论 | 优先 R17 的网络、环境配置与排错；补充 R18 |

阶段目标：能解释多 GPU 的收益、通信成本与扩展限制，并建立训练基础设施认知。

## 阶段五：GPU 集群与模型服务平台（第 33～40 周）

前置：Linux、容器、Kubernetes 基础；已有的基础设施经验可以直接复用。集群实测需要实验集群及 GPU 权限。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 17 | GPU 资源管理与调度 | 理解设备暴露、GPU 资源申请、放置、共享和隔离；认识队列、配额与拓扑约束 | 优先 R19；共享、MIG、GPU Operator、Kueue/Slurm 按实际平台选一项深入 |
| 18 | 网关、路由与流量控制 | 能比较请求路由、缓存亲和与负载均衡；明确流式响应、超时、取消、准入与队列的设计要求 | 优先 R20 或 R21 的服务架构和路由资料，选一个主线 |
| 19 | 模型存储、加载与生命周期 | 能拆解权重获取、加载、冷启动和就绪过程；分析扩缩容、版本切换、排空与回滚 | 优先 R7 的部署资料；补充所选 R20/R21 的部署与扩缩容 |
| 20 | 可观测性、可靠性与多租户 | 能组织服务与 GPU 指标；建立故障定位和恢复思路；说明鉴权、配额、隔离与日志的要求 | 优先 R4、R19；补充所选平台的运行与排错资料 |

阶段目标：能从可运行的模型服务进一步考虑持续运营、可靠性与资源管理。

## 阶段六：综合设计与验证（第 41～48 周）

前置：前五阶段。可以先针对单 GPU 或单后端限定设计范围；高级分布式部分按实际资源展开。

| 课次 | 主题 | 学习目标 | 参考资料与范围 |
|---|---|---|---|
| 21 | 工作负载、SLO、容量与成本 | 能定义输入输出分布、流量模式和服务目标；建立容量估算与有边界的成本模型 | 优先 R4、R8，结合实际资源计价资料 |
| 22 | 推理平台整体设计 | 能组织网关、模型后端、监控和生命周期管理；说明组件职责、接口、取舍及资源边界 | 优先 R12；R20/R21 选一个作架构参考 |
| 23 | PD 分离与 KV 感知系统 | 能评价 Prefill/Decode 分离、KV 传输、缓存路由与分层存储的收益条件和额外成本 | 优先 R20 或 R21；选择一种机制深入 |
| 24 | 平台验收与架构复盘 | 能用固定负载评估 SLO、故障恢复、成本和复现性；说明当前方案的限制及扩大规模后的变化 | 综合 R4、R8 与所选平台官方资料 |

阶段目标：形成有证据支持的小型推理平台方案和清晰的能力边界。第 49～52 周用于复习、补充验证与整理记录。

## 资料索引

以下为官方文档或原始论文，核对日期：2026-10-17。框架资料会更新，实际学习和实验时以选定版本为准。这里只规定阅读方向，不要求一次读完。

| 编号 | 资料 | 用途 |
|---|---|---|
| R1 | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensor、模型、训练、保存与加载；按需补基础 |
| R2 | [NVIDIA：LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | 首课只读 Understanding LLM inference，其余部分后续查阅 |
| R3 | [Hugging Face Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary) | Tokenizer 与文本编码 |
| R4 | [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) | 指标定义与测量范围 |
| R5 | [Hugging Face How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation) | Attention 与 KV Cache |
| R6 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer 原始论文，优先模型架构部分 |
| R7 | [vLLM 文档](https://docs.vllm.ai/en/latest/) | 调度、缓存、推理优化、并行和部署等专题入口 |
| R8 | [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) | 性能测量维度与配置参考 |
| R9 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | GPU 编程、线程、内存、同步 |
| R10 | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | CPU/GPU 时间线与执行分析 |
| R11 | [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | 从简单算子到矩阵乘 |
| R12 | [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | 引擎模块与执行关系 |
| R13 | [FlashAttention](https://arxiv.org/abs/2205.14135) | Attention 的 IO 优化原始论文 |
| R14 | [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/) | 量化方式、兼容性与限制 |
| R15 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | MoE、MLA 与并行设计案例；不要求部署完整模型 |
| R16 | [SGLang 文档](https://docs.sglang.io/) | 第二个推理引擎的对照资料 |
| R17 | [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) | GPU 集合通信与网络排错 |
| R18 | [PyTorch 分布式教程](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)及 [教程目录](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)中的 DDP/FSDP 专题 | 进程通信与分布式训练 |
| R19 | [Kubernetes GPU 调度](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) | GPU 设备与集群资源调度入口 |
| R20 | [NVIDIA Dynamo 文档](https://docs.nvidia.com/dynamo/latest/) | 分布式推理架构候选主线 |
| R21 | [llm-d 官方仓库与文档入口](https://github.com/llm-d/llm-d) | Kubernetes 推理服务架构候选主线 |
| R22 | [Hugging Face：How do Transformers work?](https://huggingface.co/learn/llm-course/chapter1/4) | 架构入门，优先 General Transformer architecture、Attention layers、The original architecture |
| R23 | [Hugging Face：Transformer Architectures](https://huggingface.co/learn/llm-course/chapter1/6) | 比较模型架构，重点 Decoder-only 与生成任务 |

## 当前课程：第 01 课

**主题：Transformer 架构与前向计算基础。**

目标：

1. 区分文本、Token、Token ID 与 Embedding，知道位置信息在模型中的作用。
2. 理解 Self-Attention 中 Q、K、V、注意力分数与加权结果的关系；能说明因果 Mask 的约束。
3. 认识多头 Attention、前馈网络、残差连接和归一化的职责，能沿架构图追踪数据流。
4. 能描述一个 Decoder-only 模型从输入到输出 logits、再到选出下一 Token 的前向路径。
5. 能区分原始 Transformer 的 Encoder–Decoder 架构与常见生成式 LLM 的 Decoder-only 架构。
6. 区分训练与推理；知道普通生成中存在步骤间的顺序依赖。

优先阅读顺序：R22 的架构部分 → R23 的架构比较 → R3 的基本分词说明 → R6 第 3 节。原始论文用于理解组件与 Attention，不将其中的 Encoder–Decoder 结构直接等同于所有现代 LLM。

本课深度：能看懂架构图、Attention 公式里的主要量，以及一轮前向计算的数据流。完整反向传播推导、从零训练、具体 Kernel 优化不作为本课目标；KV Cache 的详细执行与性能指标进入第 02 课。

前置要求：向量与矩阵形状、矩阵乘、Softmax 的基本含义；不熟悉时先补对应概念。当前阅读不需要 GPU；后续实验环境已选择 Colab。

进度记录由学习者自行决定。本计划不要求逐题提交答案、等待助手验收或运行已准备的工具。
