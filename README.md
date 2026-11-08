# Strata Qwen Flash：256K 上下文测试 / 256K context benchmarks

[中文](#中文) | [English](#english)

## 中文

本仓库记录 2026-11-08 一台三显卡服务器运行 Qwen3.8-Flash-Next 的环境、部署方法和 MTP 测试。数据针对该机器和合成工作负载，不能代表所有场景下的模型质量或长期稳定性。

### 实测环境

| 项目 | 配置 |
|---|---|
| CPU / 内存 | AMD EPYC 7K62，48 核 96 线程；系统可见内存 91 GiB、交换空间 14 GiB |
| 显卡 | 2 × RTX 3090（各 24,576 MiB）+ 1 × RTX 4090 D（49,140 MiB） |
| 软件 | Linux 7.0.0-38-generic；NVIDIA 驱动 595.99.02；Strata 容器 0.2.12 CUDA 13；引擎 0.1.58 |
| 模型 | Qwen3.8-Flash-Next，Unsloth UD-Q4_K_XL；启用 MTP 和图片编码器 |
| 上下文 / 并发 | 上限 262,144 token；配置 2 个请求槽 |
| GPU 与分层 | GPU 0 → 2 → 1（3090 → 3090 → 4090 D）；层切分点 12、24 |
| 内存和缓存 | INT8 KV；`--kv-resident 0`；专家缓存及预填充 auto；常驻内存预算 64 GiB；PLE 文件从 RAM 读取 |
| 其他 | 提示缓存 2，每 16,384 token 保存一次；会话缓存 0 MiB；CPU worker 32 |
| 保留的 MTP 参数 | `--spec 5 --spec-min-p 0.5` |

配置两个槽不保证两个 256K 请求均能同时装入，也不保证达到下述单请求速度。仓库不包含权重、密钥或机器标识。

### 部署步骤（Linux + NVIDIA）

下面是上游 **Terminal 安装路径**，便于在新机器上从零准备模型并启动；它不是这台服务器的 Docker 快照。原服务器使用自定义监督进程、预先准备的模型和专用网络。下面的命令参数已在 0.2.12 镜像中通过 `--plan` 检查，但没有在新机器上重新完整下载模型或做端到端部署。实际版本、显卡拓扑和可用内存不同，结果也会不同。

1. **准备资源。** 安装 NVIDIA 驱动并运行 `nvidia-smi`，确认目标显卡可见。实测机的驱动是 595.99.02；CUDA 13 镜像的上游要求为 RTX 显卡和驱动 580 或更新。0.2.12 计划模式估算需下载约 117.4 GB（权重、MTP、视觉编码器），还需额外磁盘空间存放准备后的 pack、缓存和临时文件。确保主机内存、磁盘和网络可用，参考 [上游模型指南](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/MODELS.md)。

2. **安装 Strata Terminal。** 从 [0.2.12 发布页](https://github.com/Niko1221/Strata/releases/tag/v0.2.12) 获取对应 Linux 组件并参照 [Terminal 指南](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/TERMINAL_LAUNCH.md) 安装。上游也提供以下快捷安装方式；它使用最新发布版本，**不保证复现 0.2.12 / 引擎 0.1.58**，需要严格对比时请固定版本并核对实际引擎。

```bash
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o strata-install.sh
sh strata-install.sh
strata launch --help
```

3. **先检查安装计划。** 使用 `unsloth` 家族选择 UD-Q4_K_XL；显式指定 `--drafting mtp`，避免由其他版本或默认选择改用其他草稿器。先根据 `nvidia-smi` 调整 GPU 编号；本例的 `0,2,1` 只适用于相同设备顺序。

```bash
strata launch --plan --family unsloth --model UD-Q4_K_XL \
  --context 256k --kv int8 --vision gpu --parallel 2 \
  --gpu 0,2,1 --drafting mtp --tune skip \
  --set spec=5 --set spec-min-p=0.5
```

确认计划中的模型、下载内容、显卡、上下文和 MTP 参数。256K 指输入与输出共享的上下文预算；空间不足时按上游提示降低并发、上下文或选择更小模型。已有 GGUF 分片可参考 `strata launch --help` 的 `--model-path`，传入包含同一模型所有分片的目录；准备后的 pack、MTP 和视觉文件仍可能需要额外准备或下载。

4. **生成密钥并启动。** 以下命令仅示例：密钥保存在当前用户目录，避免写进命令历史。首次启动会下载、准备模型并加载，等待服务就绪；数据目录可按磁盘情况用 `--data-dir` 指定。密钥及数据目录不要提交到 Git。

```bash
umask 077
mkdir -p "$HOME/.config/strata-bench"
openssl rand -hex 24 > "$HOME/.config/strata-bench/api-key"
export STRATA_API_KEY="$(cat "$HOME/.config/strata-bench/api-key")"
strata launch --family unsloth --model UD-Q4_K_XL \
  --context 256k --kv int8 --vision gpu --parallel 2 \
  --gpu 0,2,1 --drafting mtp --tune skip \
  --set spec=5 --set spec-min-p=0.5 \
  --access local --port 8080 --api-key-env STRATA_API_KEY \
  --save-preset qwen-mtp-256k
```

此命令使用上游自动选择的分层与缓存，**不会自动复刻**实测机的 `[12,24]` 层切分、64 GiB 预算和自定义监督器；请检查启动日志和生成配置。若上游版本改变参数或默认值，以该版本 `--help` 和文档为准。上游 Docker 的标准流程另见 [Docker 指南](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/DOCKER.md)；不要把本仓库测试配置直接当成可直接启动的 Compose 文件。

5. **访问网页并验证 API。** 在服务器打开 `http://127.0.0.1:8080/`；若页面要求密钥，在 About → Settings 填入上一步生成的密钥。客户端从 `/v1/models` 读取实际模型名；把下例 `<model-id>` 换成返回的 `id`。需要远程访问时，优先使用经认证的隧道或安全代理，不要把无认证接口暴露到公网。

```bash
curl -fsS http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $STRATA_API_KEY"
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

6. **复测与维护。** 克隆本仓库，按下方命令运行长上下文测试；观察 `cached_tokens`、标记检查、显存和温度。首次加载时 502 可能只表示服务尚未就绪。用 `strata launch --status` 查看状态、`strata launch --stop` 停止；使用相同 preset 重启前，参照 `strata launch --help` 核对保留的参数。变更设置前保存配置或 preset，以便回退。

```bash
git clone https://github.com/blakeccb/strata-qwen-flash-benchmarks.git
cd strata-qwen-flash-benchmarks
export STRATA_BASE_URL='http://127.0.0.1:8080'
export STRATA_MODEL='<model-id>'
python3 scripts/long_test.py unique-run-01
```

### 测试结果

每次成功请求输入约 259,293 token，`temperature=0`、关闭 thinking、`max_tokens=512`，缓存输入为 0。输入由 16,200 次重复记录组成，在约三分之一、三分之二及结尾放置三个标记；检查仅验证字符串是否出现。所有成功请求均因输出上限结束。速度为引擎报告的**生成速度**，不是整次请求速度。

| 窗口 | 阈值 | 每次生成速度 token/s | 平均 token/s | 标记 |
|---|---:|---|---:|---:|
| 4 | 默认 | 90.7、82.3、95.4 | 89.5 | 3/3 |
| 4 | 0.3 | 95.4、105.6、96.7 | 99.2 | 3/3 |
| 5 | 0.3 | 96.6、107.9、101.3 | 101.9 | 3/3 |
| 5 | 0.5 | 97.1、106.2、110.6、112.0 | 106.5 | 4/4 |
| 6 | 默认 | 86.4 | 86.4 | 1/1 |

窗口 5 的 0.5 比 0.3 平均快约 4.4%，两组平均总耗时分别是 42.73 秒与 42.89 秒；预填充占据大部分时间。测试按组顺序执行，没有交替运行，样本少；默认阈值组并非与 0.3 的严格对照。另一次窗口 5 / 0.3 请求在服务启动期间得到 502，已保留为错误记录，未计入速度平均值。

[逐次长上下文数据](data/mtp-long.jsonl) 包含预填充、耗时、token、草稿及接受数；[短请求并发记录](data/short-concurrency.jsonl) 包含早期探索性测试，但各历史运行的完整配置未全部保存。短请求与合成标记测试不能证明复杂推理、工具调用、图像理解、两个 256K 请求并发、自然结束或长期稳定性。草稿接受率不等于答案正确率。

---

## English

This repository records a three-GPU Qwen3.8-Flash-Next setup and MTP tests from 2026-11-08. The observations apply to this machine and synthetic workload; they are not a general quality or long-term reliability evaluation.

### Test system

| Component | Configuration |
|---|---|
| CPU / RAM | AMD EPYC 7K62, 48 cores / 96 threads; 91 GiB visible RAM and 14 GiB swap |
| GPUs | 2 × RTX 3090 (24,576 MiB each), 1 × RTX 4090 D (49,140 MiB) |
| Software | Linux 7.0.0-38-generic; NVIDIA driver 595.99.02; Strata container 0.2.12 CUDA 13; engine 0.1.58 |
| Model | Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL; MTP and vision enabled |
| Context / slots | 262,144-token limit; two configured request slots |
| GPU order / split | GPU 0 → 2 → 1 (3090 → 3090 → 4090 D), layer boundaries 12 and 24 |
| Cache | INT8 KV, `--kv-resident 0`, automatic expert cache and prefill, 64 GiB resident budget, PLE I/O from RAM |
| Other | two prompt caches, 16,384-token interval, conversation cache 0 MiB, 32 CPU workers |
| Retained MTP | `--spec 5 --spec-min-p 0.5` |

Two slots do not guarantee that two 256K requests fit simultaneously or attain the single-request speed below. Model weights, keys and machine identifiers are not included.

### Deployment (Linux + NVIDIA)

This is an upstream **Terminal installation path** for a new machine, not an exact copy of the benchmark server's Docker deployment. The latter uses a custom supervisor, prepared model files and dedicated networks. The parameters below were checked with `--plan` in the 0.2.12 image; a fresh end-to-end installation was not performed. Hardware and software differences can change the outcome.

1. **Prepare the host.** Install the NVIDIA driver and check `nvidia-smi`. The measured host used driver 595.99.02; upstream specifies driver 580 or newer for RTX CUDA 13 images. The 0.2.12 plan estimates a 117.4 GB download for weights, MTP and vision; allow additional disk space for the prepared pack, cache and temporary files. Check available RAM, storage and networking. See [upstream models](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/MODELS.md).

2. **Install Strata Terminal.** Follow the [0.2.12 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.12) and [Terminal guide](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/TERMINAL_LAUNCH.md) for the Linux components. The convenient installer below uses the latest release, so it **does not pin 0.2.12 / engine 0.1.58**; pin and verify the actual versions for a strict comparison.

```bash
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o strata-install.sh
sh strata-install.sh
strata launch --help
```

3. **Inspect the plan.** Choose family `unsloth` for UD-Q4_K_XL and explicitly select `--drafting mtp`; defaults may select another drafter. Adjust GPU indices to match `nvidia-smi`; `0,2,1` applies only to the measured ordering.

```bash
strata launch --plan --family unsloth --model UD-Q4_K_XL \
  --context 256k --kv int8 --vision gpu --parallel 2 \
  --gpu 0,2,1 --drafting mtp --tune skip \
  --set spec=5 --set spec-min-p=0.5
```

Check the reported model, downloads, GPUs, context and MTP settings. The context budget includes both input and output. Reduce slots, context or model size if it does not fit. For existing GGUF shards, see `--model-path` in `strata launch --help`; use the directory containing all shards for the same model. A prepared pack, MTP and vision files may still need preparation or download.

4. **Create a key and start.** The example keeps the key in the current user's home directory rather than command history. The first run downloads, prepares and loads the model; wait until it is ready. Use `--data-dir` to choose a suitable disk if needed. Do not commit keys or data directories.

```bash
umask 077
mkdir -p "$HOME/.config/strata-bench"
openssl rand -hex 24 > "$HOME/.config/strata-bench/api-key"
export STRATA_API_KEY="$(cat "$HOME/.config/strata-bench/api-key")"
strata launch --family unsloth --model UD-Q4_K_XL \
  --context 256k --kv int8 --vision gpu --parallel 2 \
  --gpu 0,2,1 --drafting mtp --tune skip \
  --set spec=5 --set spec-min-p=0.5 \
  --access local --port 8080 --api-key-env STRATA_API_KEY \
  --save-preset qwen-mtp-256k
```

This uses upstream automatic splitting and cache choices: it **does not automatically recreate** the measured `[12,24]` boundaries, 64 GiB budget or custom supervisor. Inspect the startup log and generated configuration. If options or defaults change, consult that release's `--help` and docs. For standard Docker installation see the [Docker guide](https://github.com/Niko1221/Strata/blob/v0.2.12/docs/DOCKER.md); this repository does not ship a ready-to-run copy of the custom Compose stack.

5. **Open the web UI and test the API.** Open `http://127.0.0.1:8080/` on the server. If prompted, enter the key under About → Settings. Read the actual model `id` from `/v1/models` and replace `<model-id>` below. For remote access, prefer an authenticated tunnel or secure proxy; do not expose an unauthenticated endpoint publicly.

```bash
curl -fsS http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $STRATA_API_KEY"
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

6. **Benchmark and maintain.** Clone this repo and run the long-context script below; inspect `cached_tokens`, markers, GPU memory and temperature. A 502 during loading can mean the server is not ready yet. Use `strata launch --status` and `strata launch --stop`; check `strata launch --help` for retained preset settings before a restart. Save the configuration or preset before changes so you can roll back.

```bash
git clone https://github.com/blakeccb/strata-qwen-flash-benchmarks.git
cd strata-qwen-flash-benchmarks
export STRATA_BASE_URL='http://127.0.0.1:8080'
export STRATA_MODEL='<model-id>'
python3 scripts/long_test.py unique-run-01
```

### Benchmark results

Each successful run used approximately 259,293 input tokens, `temperature=0`, thinking disabled, `max_tokens=512` and zero cached input tokens. The prompt repeats one record 16,200 times, with markers near one-third, two-thirds and the end. The check is substring presence only. All successful runs ended at the output limit. Speeds are server-reported **decode** throughput, not overall request throughput.

| Window | Threshold | Individual decode token/s | Mean token/s | Markers |
|---|---:|---|---:|---:|
| 4 | default | 90.7, 82.3, 95.4 | 89.5 | 3/3 |
| 4 | 0.3 | 95.4, 105.6, 96.7 | 99.2 | 3/3 |
| 5 | 0.3 | 96.6, 107.9, 101.3 | 101.9 | 3/3 |
| 5 | 0.5 | 97.1, 106.2, 110.6, 112.0 | 106.5 | 4/4 |
| 6 | default | 86.4 | 86.4 | 1/1 |

At window 5, the 0.5 mean was about 4.4% higher than 0.3; mean wall times were 42.73 s and 42.89 s respectively, dominated by prefill. Runs were grouped rather than interleaved, with few samples. Default-threshold runs are not controlled 0.3 comparisons. One additional window-5/0.3 request got HTTP 502 while the server was starting; it is retained as an error record and excluded from throughput averages.

[Per-run long-context records](data/mtp-long.jsonl) include prefill, elapsed time, token counts and draft acceptance. [Short-request concurrency records](data/short-concurrency.jsonl) contain earlier exploratory tests; complete settings for every historical run were not captured. Neither synthetic markers nor short requests prove complex reasoning, tool calling, vision quality, concurrent 256K capacity, natural completion or sustained stability. Draft acceptance is not answer accuracy.

This repository is independent of the [upstream Strata project](https://github.com/Niko1221/Strata).
