"""
if not obvious:
python bench_apc.py --repeat-prefix 100 --max-tokens 32
"""

import argparse
import json
import subprocess
import sys
import time


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run_child(args):
    from vllm import LLM, SamplingParams

    shared_prefix = """
# AlphaServe 公司知识库

AlphaServe 是一个高吞吐 LLM 推理服务系统。
它支持 OpenAI-compatible server。
它支持 continuous batching。
它支持 automatic prefix caching。
它支持 chunked prefill。
它使用 scheduler 管理请求。
它使用 KV cache 存储历史 token 的中间状态。
""" * args.repeat_prefix

    questions = [
        "这个系统的核心能力是什么？",
        "它为什么适合高并发？",
        "它和 KV cache 有什么关系？",
        "它和 continuous batching 有什么关系？",
        "它为什么适合做在线服务？",
        "如果 prompt 很长，它有什么优化机会？",
    ]

    prompts = [
        shared_prefix + f"\n问题：{q}\n请用两句话回答。"
        for q in questions
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    print(f"加载模型：{args.model}")
    print(f"enable_prefix_caching={args.enable_prefix_caching}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    llm.generate(["请只回答 OK。"], SamplingParams(temperature=0.0, max_tokens=8))

    per_request = []

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        llm.generate([prompt], sampling_params)
        end = time.perf_counter()

        elapsed = end - start
        per_request.append(elapsed)

        print(f"request {i}: {elapsed:.3f}s")

    result = {
        "enable_prefix_caching": args.enable_prefix_caching,
        "per_request": per_request,
        "first": per_request[0],
        "avg_after_first": sum(per_request[1:]) / max(1, len(per_request[1:])),
        "total": sum(per_request),
    }

    print("RESULT_JSON " + json.dumps(result, ensure_ascii=False))


def run_parent(args):
    results = []

    for enable in [False, True]:
        cmd = [
            sys.executable,
            __file__,
            "--child",
            "--model",
            args.model,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--repeat-prefix",
            str(args.repeat_prefix),
            "--max-tokens",
            str(args.max_tokens),
        ]

        if enable:
            cmd.append("--enable-prefix-caching")

        print("\n" + "=" * 70)
        print(f"运行 APC benchmark: enable_prefix_caching={enable}")
        print("=" * 70)

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

    print("\n=== APC 对比汇总 ===")
    for r in results:
        print(
            f"APC={r['enable_prefix_caching']:<5} "
            f"first={r['first']:.3f}s "
            f"avg_after_first={r['avg_after_first']:.3f}s "
            f"total={r['total']:.3f}s"
        )

    print("\n观察重点：")
    print("1. 不要只看第一个请求。")
    print("2. 看 avg_after_first，因为 APC 的收益主要体现在后续共享前缀请求。")
    print("3. 如果差异不明显，把 --repeat-prefix 调大，或把 --max-tokens 调小。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--repeat-prefix", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    args = parser.parse_args()

    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()