"""
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85
"""

"""
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85
"""



import json
import threading
import time

import requests

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_URL = "http://127.0.0.1:8000"


def make_long_prompt():
    text = """
# AlphaServe 系统手册

AlphaServe 是一个高吞吐推理系统。
它使用 scheduler 管理请求。
它使用 KV cache 保存历史 token 的中间状态。
它支持 continuous batching。
它支持 prefix caching。
它支持 OpenAI-compatible server。
它通过 GPU worker 执行模型计算。
""" * 120
    return text + "\n问题：请用三句话总结这个系统的请求处理路径。"

def make_short_prmopt():
    return "请用一句话解释 vLLM 为什么适合高并发推理。"



def request_once(prompt, label):
    url = f"{BASE_URL}/v1/chat/completions"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    output = []

    response = requests.post(url, json=payload, stream=True)
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        line = line.strip()

        if not line.startswith("data: "):
            continue

        data = line[len("data: "):].strip()
        if data == "[DONE]":
            break

        obj = json.loads(data)
        piece = obj["choices"][0].get("text", "")


        if piece and first_token_time is None:
            first_token_time = time.perf_counter()
        
        output.append(piece)
    
    end = time.perf_counter()

    ttft = None
    if first_token_time is not None:
        ttft = first_token_time - start

    return {
        "label": label,
        "ttft": ttft,
        "total_time": end - start,
        "output_chars": len("".join(output)),
    }

def print_result(result):
    ttft = result["ttft"]

    if ttft is None:
        ttft_text = "None"
    else:
        ttft_text = f"{ttft:.3f} seconds"

    print(
        f"{result['label']:<20} "
        f"TTFT={ttft_text:<10} "
        f"TOTAL={result['total']:.3f}s"
    )


def main():
    long_prompt = make_long_prompt()
    short_prompt = make_short_prompt()

    print("\n=== 1. 短请求单独跑 ===")
    short_alone = request_once(short_prompt, "SHORT alone")
    print_result(short_alone)

    print("\n=== 2. 长请求单独跑 ===")
    long_alone = request_once(long_prompt, "LONG alone")
    print_result(long_alone)

    print("\n=== 3. 长请求先进来，短请求随后插入 ===")

    results = {}

    def run_long():
        results["long"] = request_once(long_prompt, "LONG during pair")

    thread = threading.Thread(target=run_long)
    thread.start()

    time.sleep(0.15)

    results["short"] = request_once(short_prompt, "SHORT while long")

    thread.join()

    print_result(results["long"])
    print_result(results["short"])

    print("\n=== 核心对比 ===")
    print_result(short_alone)
    print_result(results["short"])

    if short_alone["ttft"] and results["short"]["ttft"]:
        ratio = results["short"]["ttft"] / short_alone["ttft"]
        print(f"\n短请求 TTFT 放大倍数: {ratio:.2f}x")


if __name__ == "__main__":
    main()











