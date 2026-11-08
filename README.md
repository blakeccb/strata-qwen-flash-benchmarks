# Strata Qwen Flash: 256K context benchmarks

A reproducibility-oriented snapshot of one three-GPU local inference setup and its MTP experiments, recorded on 2026-11-08. Results describe this particular machine and synthetic workload; they are not general model-quality or long-term reliability claims.

## Hardware and software

| Component | Configuration |
|---|---|
| CPU | AMD EPYC 7K62, 48 cores / 96 threads, one socket |
| Host memory | 91 GiB available to OS; 14 GiB swap |
| GPUs | 2 × RTX 3090 (24,576 MiB each), 1 × RTX 4090 D (49,140 MiB) |
| OS / driver | Linux 7.0.0-38-generic; NVIDIA driver 595.99.02 |
| Runtime | Strata container 0.2.12 CUDA 13; engine 0.1.58 |
| Model | Qwen3.8-Flash-Next, UD-Q4_K_XL, with MTP draft and vision encoder |
| Context / concurrent slots | 262,144 tokens / 2 configured slots |
| Layer order | 3090 → 3090 → 4090 D; split boundaries 12, 24 |
| Cache and prefill | INT8 KV, KV resident 0, auto prefill and expert cache; resident budget 64 GiB; PLE I/O RAM |
| Other | prompt cache 2, interval 16,384; conversation cache 0 MiB; pool workers 32; vision enabled |
| Retained MTP settings | `--spec 5 --spec-min-p 0.5` |

The configured two slots do not imply two independent 256K requests are guaranteed to fit or achieve the single-request throughput below. The snapshot excludes model weights, keys, machine identifiers, IP addresses and private paths.

## Long-context MTP results

Each successful run used an approximately 259,293-token input, `temperature=0`, thinking disabled, `max_tokens=512`, and no cached input tokens. The synthetic input repeats one short record 16,200 times and places three markers at roughly one-third, two-thirds and the end. The check tests substring presence, not a semantic evaluation. All successful runs ended at the output limit (`finish_reason=length`). Speeds are server-reported decoding throughput, not overall request throughput.

| Window | Threshold | Decode tokens/s (individual runs) | Mean tokens/s | Marker checks |
|---|---:|---|---:|---:|
| 4 | default | 90.7, 82.3, 95.4 | 89.5 | 3/3 |
| 4 | 0.3 | 95.4, 105.6, 96.7 | 99.2 | 3/3 |
| 5 | 0.3 | 96.6, 107.9, 101.3 | 101.9 | 3/3 |
| 5 | 0.5 | 97.1, 106.2, 110.6, 112.0 | 106.5 | 4/4 |
| 6 | default | 86.4 | 86.4 | 1/1 |

The default-threshold runs did not explicitly set `--spec-min-p`; do not interpret them as controlled 0.3 comparisons. The 0.5 and 0.3 window-5 runs were sequential, not interleaved, and have only 4 and 3 samples respectively. The 0.5 average was about 4.4% higher, while mean wall time was 42.73 s versus 42.89 s: prefill dominates this workload. One additional window-5/0.3 request received HTTP 502 during startup before the model became ready; it is retained as an error record, excluded from throughput averages. It should not be interpreted as a healthy-server inference failure.

The full per-run numeric records, including prefill time, request duration, token counts, draft/accepted counts and failure summary, are in [data/mtp-long.jsonl](data/mtp-long.jsonl). The prompt construction and request settings are in [scripts/long_test.py](scripts/long_test.py). The script is adapted to read its URL and key from environment variables rather than machine-specific paths.

## Reproducing

Run against a compatible OpenAI-style endpoint with a matching model and context configuration. The test consumes substantial GPU time and memory. The script prints metrics and marker checks, not the model response.

```bash
export STRATA_BASE_URL='http://127.0.0.1:8095'
export STRATA_API_KEY='<your key>'
export STRATA_MODEL='qwen3.8-flash-next-strata-ud-q4_k_xl'
python3 scripts/long_test.py my-unique-run
```

Use a fresh tag for each run, check `cached_tokens` is zero, repeat and preferably interleave settings, and record temperature, output cap, model/engine versions and system load. Changing MTP settings can require an engine restart; allow it to become healthy before testing.

## Interpretation and limitations

These are cold, single-request, short-output synthetic retrieval and throughput tests. The marker check does not establish long-context reasoning, tool calling, image quality, output equivalence, natural task completion, multi-user performance or sustained stability. The data is not a controlled model-quality evaluation. A threshold changes when the draft is extended; it is not a measure of answer accuracy. Results can vary with workload, model, sampling, GPU temperature and software version. Draft acceptance is `draft_n_accepted / draft_n`, not answer correctness.

See the [Strata project](https://github.com/Niko1221/Strata) for engine documentation and current supported options. This repository shares observations and test code; it is not affiliated with the upstream project.
