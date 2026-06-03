"""
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85

这次我固定了模型、固定了 prompt 类型、固定了并发数，只改 max_num_batched_tokens。
我观察的是：长 prompt 和短 prompt 混在一起时，短请求的 TTFT 有没有变化。
这不是论文级 benchmark，而是为了训练我看系统行为的能力。
"""

import argparse
import concurrent.futures
import json
import statistics
import time

import requests


def make_prompt(i: int, long: bool) -> str:
    if long:
        body = """
# AlphaServe 长文档

AlphaServe 是一个高吞吐 LLM 推理系统。
它使用 API server 接收请求。
它使用 input processing 处理输入。
它使用 engine core 推进请求。
它使用 scheduler 组织 batch。
它使用 KV cache 保存历史 token 的状态。
它使用 GPU worker 执行模型 forward。
""" * 90

        return body + f"\n问题 {i}：请用两句话总结这个系统的请求路径。"

    return f"问题 {i}：请用一句话解释 vLLM 为什么适合高并发推理。"


def request_once(base_url: str, model: str, prompt: str, max_tokens: int):
    url = f"{base_url.rstrip('/')}/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    chars = 0

    response = requests.post(url, json=payload, stream=True, timeout=None)
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        line = line.strip()

        if not line.startswith("data:"):
            continue

        data = line[len("data:"):].strip()

        if data == "[DONE]":
            break

        obj = json.loads(data)
        piece = obj["choices"][0].get("text", "")

        if piece and first_token_time is None:
            first_token_time = time.perf_counter()

        chars += len(piece)

    end = time.perf_counter()

    return {
        "ttft": None if first_token_time is None else first_token_time - start,
        "total": end - start,
        "chars": chars,
    }


def summarize(name, results):
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    totals = [r["total"] for r in results]

    print(f"\n=== {name} ===")
    print(f"requests: {len(results)}")

    if ttfts:
        print(f"TTFT avg: {statistics.mean(ttfts):.3f}s")
        print(f"TTFT p50: {statistics.median(ttfts):.3f}s")
        print(f"TTFT max: {max(ttfts):.3f}s")

    print(f"TOTAL avg: {statistics.mean(totals):.3f}s")
    print(f"TOTAL max: {max(totals):.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-short", type=int, default=8)
    parser.add_argument("--num-long", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    short_prompts = [make_prompt(i, long=False) for i in range(args.num_short)]
    long_prompts = [make_prompt(i, long=True) for i in range(args.num_long)]

    mixed_prompts = []
    for i in range(max(len(short_prompts), len(long_prompts))):
        if i < len(long_prompts):
            mixed_prompts.append(("LONG", long_prompts[i]))
        if i < len(short_prompts):
            mixed_prompts.append(("SHORT", short_prompts[i]))

    print("开始并发请求")
    print(f"short={args.num_short}, long={args.num_long}, concurrency={args.concurrency}")

    results_by_type = {
        "SHORT": [],
        "LONG": [],
    }

    def task(item):
        label, prompt = item
        result = request_once(args.base_url, args.model, prompt, args.max_tokens)
        return label, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(task, item) for item in mixed_prompts]

        for future in concurrent.futures.as_completed(futures):
            label, result = future.result()
            results_by_type[label].append(result)
            ttft = result["ttft"]
            ttft_text = "None" if ttft is None else f"{ttft:.3f}s"
            print(f"{label:<5} TTFT={ttft_text:<8} TOTAL={result['total']:.3f}s")

    summarize("SHORT", results_by_type["SHORT"])
    summarize("LONG", results_by_type["LONG"])

    print("\n观察重点：")
    print("1. 看 SHORT 的 TTFT avg / p50 / max。")
    print("2. 换 max_num_batched_tokens 后，只比较 SHORT 的体感有没有变化。")
    print("3. 不要同时改模型、prompt、并发数，否则你不知道是谁造成变化。")


if __name__ == "__main__":
    main()