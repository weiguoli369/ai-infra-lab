# 从 Token 到显存

用浏览器打开 `index.html`，即可离线使用，无需安装依赖或启动服务。

四个看板展示嵌入查表、dense padding、packed prefill / 分页 KV、连续批处理与维度速查。可调节 A / B 请求长度、数据精度和教学块大小，点击逻辑块映射定位物理缓存，并逐轮观察请求完成与替换。

模型参数参考 Qwen2.5-0.5B：vocab=151936、hidden=896、layers=24、attention heads=14、KV heads=2、head dim=64。蓝图中的两个 token ID 为固定示例，未执行真实分词；向量数值、块号与调度均为模拟，不构成性能基准。隐藏状态与 KV 采用相同的选定精度，KiB 按 1024 B 计算。

默认 A=10、B=2、BF16：有效隐藏状态 21 KiB，dense 35 KiB，PAD 14 KiB，全 Batch 浪费率 40%。KV 每 token 12 KiB；教学块容量 4 时，4 个块束共 16 槽，192 KiB，其中 48 KiB 为尾块空位。内存池可预留更多空间，因此这些数值不等于引擎总显存。

参考：[模型配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)、[vLLM 原理说明](https://vllm.ai/blog/2023-06-20-vllm)、[PagedAttention 论文](https://arxiv.org/abs/2309.06180)。
