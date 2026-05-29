"""
运行
python model_compare.py

如果显存紧张：

python model_compare.py --max-model-len 1024 --gpu-memory-utilization 0.75

如果 TinyLlama 首次下载很慢，第一次结果会包含下载时间。下载完后再跑一次：

python model_compare.py

第二次结果更适合作为冷启动体感比较。
"""

import argparse
import json
import subprocess
import sys
import time


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def gpu_memory_mb():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        values = [int(x.strip()) for x in output.splitlines() if x.strip()]
        return values
    except Exception:
        return []


def run_child(args):
    from vllm import LLM, SamplingParams

    before_mem = gpu_memory_mb()

    print(f"加载模型: {args.model}")
    load_start = time.perf_counter()

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    load_end = time.perf_counter()
    after_load_mem = gpu_memory_mb()

    prompt = """
请用三句话解释 vLLM 的请求路径：
Client、API server、engine core、scheduler、KV cache、GPU worker 分别做什么？
"""

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    first_start = time.perf_counter()
    llm.generate([prompt], sampling_params)
    first_end = time.perf_counter()

    second_start = time.perf_counter()
    llm.generate([prompt], sampling_params)
    second_end = time.perf_counter()

    result = {
        "model": args.model,
        "load_seconds": load_end - load_start,
        "first_request_seconds": first_end - first_start,
        "second_request_seconds": second_end - second_start,
        "gpu_mem_before_mb": before_mem,
        "gpu_mem_after_load_mb": after_load_mem,
    }

    print("RESULT_JSON " + json.dumps(result, ensure_ascii=False))


def run_parent(args):
    results = []

    for model in args.models:
        cmd = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            model,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-tokens",
            str(args.max_tokens),
        ]

        print("\n" + "=" * 80)
        print(f"测试模型：{model}")
        print("=" * 80)

        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(proc.stdout)

        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                results.append(json.loads(line[len("RESULT_JSON "):]))

    print("\n=== 模型体感对比 ===")
    for r in results:
        print(f"\n模型: {r['model']}")
        print(f"加载耗时:       {r['load_seconds']:.3f}s")
        print(f"首次请求耗时:   {r['first_request_seconds']:.3f}s")
        print(f"第二次请求耗时: {r['second_request_seconds']:.3f}s")
        print(f"加载前显存:     {r['gpu_mem_before_mb']}")
        print(f"加载后显存:     {r['gpu_mem_after_load_mb']}")

    print("\n写用户体验笔记，不要写模型智商评测。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()