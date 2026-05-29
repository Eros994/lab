下面给你一版**明天一天完成版**。
结构保持干净：**5 个模块，4 个 Python 文件，1 份复盘笔记**。

---

# vLLM 第 8–12 天压缩版

明天只做这 5 件事：

| 模块 | 主题                                | 你要得到的感觉                    |
| -- | --------------------------------- | -------------------------- |
| 1  | LLMEngine 思维                      | server 只是外壳，engine 才是在推进请求 |
| 2  | continuous batching / KV cache 背景 | batching 和 KV cache 不再是抽象词 |
| 3  | 小 benchmark                       | 学会固定条件、一次只改一个参数            |
| 4  | 0.5B vs 1.1B 模型体感                 | 模型变大后，系统味道会变               |
| 5  | V1 guide 纠偏                       | 清理旧文章、旧视频留下的过时印象           |

官方 LLMEngine example 里核心就是 `engine.add_request(...)` 然后不断 `engine.step()`；这正好用来建立“engine 在推进请求”的直觉。([vLLM][1])

---

# 0. 准备目录

```bash
mkdir -p vllm-day8-12-lab
cd vllm-day8-12-lab
```

检查环境：

```bash
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
```

需要的话安装：

```bash
pip install -U vllm requests
```

如果你想顺手试官方 benchmark CLI，再装：

```bash
pip install -U "vllm[bench]"
```

vLLM 官方 CLI 里 benchmark 入口包括 `latency`、`serve`、`throughput` 三类；你明天主要用自己的小脚本，但知道官方入口存在就够了。([vLLM][2])

---

# 1. Day 8：LLMEngine 思维

## 目标

你之前看到的是：

```text
curl -> API server -> engine -> worker
```

今天你要第一次绕过 server，直接看 engine 思维：

```text
add_request()
    ↓
step()
    ↓
step()
    ↓
step()
    ↓
request finished
```

你要形成的理解是：

> server 是外面那层 HTTP 皮，内部真正推动请求前进的是 engine。

---

## 创建代码

新建文件：

```bash
nano engine_loop_lab.py
```

粘贴：

```python
import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_prompts() -> List[Tuple[str, SamplingParams]]:
    return [
        (
            "请用一句话解释 vLLM 的 engine 是什么。",
            SamplingParams(temperature=0.0, max_tokens=48),
        ),
        (
            "请用一句话解释 scheduler 在推理服务里做什么。",
            SamplingParams(temperature=0.0, max_tokens=48),
        ),
        (
            "请用一句话解释 KV cache 为什么重要。",
            SamplingParams(temperature=0.0, max_tokens=48),
        ),
    ]


def process_requests(engine: LLMEngine, prompts: List[Tuple[str, SamplingParams]]) -> None:
    request_id = 0
    step_id = 0

    print("\n开始 engine loop")
    print("=" * 60)

    while prompts or engine.has_unfinished_requests():
        print(f"\n[step {step_id}]")

        if prompts:
            prompt, sampling_params = prompts.pop(0)
            print(f"add_request: request_id={request_id}, prompt={prompt[:30]}...")
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        outputs: List[RequestOutput] = engine.step()

        if not outputs:
            print("本 step 暂时没有 finished output")
        else:
            for output in outputs:
                print(f"收到 output: request_id={output.request_id}, finished={output.finished}")

                if output.finished:
                    text = output.outputs[0].text
                    print("-" * 40)
                    print(text.strip())
                    print("-" * 40)

        step_id += 1

    print("\n所有请求完成")


def main() -> None:
    parser = FlexibleArgumentParser(description="LLMEngine add_request / step demo")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    prompts = create_prompts()
    process_requests(engine, prompts)


if __name__ == "__main__":
    main()
```

---

## 运行

```bash
python engine_loop_lab.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

---

## 观察什么

你要看打印里的这几个东西：

```text
add_request
step
finished=False
finished=True
```

你要体会的是：

| 现象                | 说明                                |
| ----------------- | --------------------------------- |
| `add_request()`   | 把请求放进 engine                      |
| `step()`          | engine 向前推进一步                     |
| 一个请求可能多次 step 才完成 | 推理不是一次函数调用就结束，而是被调度器一步步推进         |
| server 没出现        | 说明 HTTP server 只是外层入口，不是 vLLM 的全部 |

---

## 今天要写下来的话

写到笔记里：

```text
我第一次看到 LLMEngine 的 add_request / step 模式。
这让我意识到：server 只是对外接口，内部真正推进请求的是 engine。
engine 每 step 一次，scheduler 就有机会安排请求向前走一步。
```

---

# 2. Day 9：补 continuous batching 和 KV cache 背景

## 目标

今天不跑复杂实验。

只做一件事：

> 把 continuous batching 和 KV cache 跟你前几天的实验绑起来。

Anyscale 的文章强调，LLM 是逐 token 生成的，因此可以做 iteration-level scheduling，也就是 continuous batching；这类系统级 batching 优化能显著影响吞吐和延迟。([Anyscale][3])
Hugging Face 的 KV cache 解释里也说，KV cache 的本质是记住之前步骤的计算，避免重复算；它能加速生成，但会占额外内存。([Hugging Face][4])

---

## 你要写的 10 句话

新建笔记：

```bash
nano background_notes.md
```

粘贴模板：

```markdown
# Day 9 背景笔记：continuous batching 和 KV cache

## A. continuous batching：5 句话

1. LLM 不是一次性生成完整答案，而是一个 token 一个 token 地生成。
2. 因为不同请求的输出长度不同，传统“等一整个 batch 都结束”的方式会浪费 GPU。
3. continuous batching 的直觉是：每一步 decode 都可以重新组织 batch，让新请求插进来。
4. 这和我前面看到的 scheduler 现象对应：请求不是静态排队，而是在 engine step 里不断被调度。
5. 这也解释了为什么长 prompt、短 prompt 混在一起时，调度策略会影响短请求的体感延迟。

## B. KV cache：5 句话

1. KV cache 保存的是模型对历史 token 计算出来的 Key / Value。
2. 有了 KV cache，模型生成下一个 token 时不需要从头重算整个上下文。
3. KV cache 让 decode 更快，但会占用显存。
4. prefix caching 进一步把“同一个请求内部的 KV 复用”扩展成“不同请求之间共享前缀的 KV 复用”。
5. 我前面 APC 实验里第二次请求更快，就是因为共享前缀对应的 KV cache 被复用了。

## 我自己的补充

今天我终于把这几个词连起来了：

Client 请求
-> engine step
-> continuous batching
-> scheduler
-> KV cache
-> GPU worker
```

---

## 你要体会什么

这一天的关键不是记术语，而是把术语和现象对上。

| 术语                  | 对应你已经看到的现象                |
| ------------------- | ------------------------- |
| continuous batching | 多个请求不是死板排队，而是在 step 中动态组织 |
| KV cache            | 生成时不用反复重算历史               |
| APC                 | 不同请求之间复用共享前缀的 KV cache    |
| token budget        | 每轮调度能塞多少 token            |
| chunked prefill     | 长 prompt 被拆块进入调度          |

---

# 3. Day 10：第一次做像样一点的 benchmark

## 目标

今天你只练三个原则：

```text
固定模型
固定 prompt 模式
一次只改一个参数
```

你要做两组对比：

| 对比                  | 只改什么                        |
| ------------------- | --------------------------- |
| APC off vs on       | 只改 `enable_prefix_caching`  |
| token budget 小 vs 大 | 只改 `max_num_batched_tokens` |

---

# 3.1 Benchmark A：APC off vs on

## 创建代码

```bash
nano bench_apc.py
```

粘贴：

```python
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
```

---

## 运行

```bash
python bench_apc.py
```

如果现象不明显：

```bash
python bench_apc.py --repeat-prefix 100 --max-tokens 32
```

---

## 记录结果

```text
APC=False
first:            ____ 秒
avg_after_first:  ____ 秒
total:            ____ 秒

APC=True
first:            ____ 秒
avg_after_first:  ____ 秒
total:            ____ 秒
```

---

## 正确理解

你要写下：

```text
这次 benchmark 比前几天更像样，因为我固定了模型、固定了 prompt 模式，只改 enable_prefix_caching。
我不再只看单个请求，而是看多个共享前缀请求的后续平均耗时。
APC 的核心收益应该主要体现在第一个请求之后。
```

---

# 3.2 Benchmark B：token budget 小 vs 大

这一组用 server 跑。

你只改：

```text
--max-num-batched-tokens
```

其他都不动。

---

## 创建 client 代码

```bash
nano bench_server_budget.py
```

粘贴：

```python
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
```

---

## 启动 server：小 budget

终端 A：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85
```

终端 B：

```bash
python bench_server_budget.py
```

记录：

```text
max_num_batched_tokens = 1024

SHORT TTFT avg: ____ 秒
SHORT TTFT p50: ____ 秒
SHORT TTFT max: ____ 秒

LONG TTFT avg:  ____ 秒
```

---

## 启动 server：大 budget

终端 A 先 `Ctrl-C`，然后：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85
```

终端 B 再跑：

```bash
python bench_server_budget.py
```

记录：

```text
max_num_batched_tokens = 4096

SHORT TTFT avg: ____ 秒
SHORT TTFT p50: ____ 秒
SHORT TTFT max: ____ 秒

LONG TTFT avg:  ____ 秒
```

---

## 正确理解

写下来：

```text
这次我固定了模型、固定了 prompt 类型、固定了并发数，只改 max_num_batched_tokens。
我观察的是：长 prompt 和短 prompt 混在一起时，短请求的 TTFT 有没有变化。
这不是论文级 benchmark，而是为了训练我看系统行为的能力。
```

---

# 3.3 可选：跑一次官方 vLLM bench

这一步可做可不做。

先启动 server：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

再跑：

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 64 \
  --num-prompts 20 \
  --max-concurrency 4
```

官方 `serve` benchmark 是在线 serving benchmark，文档也说明 server 侧用 `vllm serve`，client 侧用 `vllm bench serve`。([vLLM][5])

你今天只需要知道：

```text
官方 benchmark 工具是存在的。
以后真做性能测试，不一定都要自己手写计时代码。
```

---

# 4. Day 11：0.5B vs 1.1B 模型体感

## 目标

今天不是比较模型聪不聪明。

你要比较的是：

```text
启动速度
首次请求速度
显存占用
是否更容易想压小 max_model_len
哪个更适合 smoke test
```

Qwen2.5-0.5B-Instruct 的模型卡写明参数量约 0.49B；TinyLlama-1.1B-Chat-v1.0 是 1.1B 级别模型。([Hugging Face][6])

---

## 创建代码

```bash
nano model_compare.py
```

粘贴：

```python
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
```

---

## 运行

```bash
python model_compare.py
```

如果显存紧张：

```bash
python model_compare.py --max-model-len 1024 --gpu-memory-utilization 0.75
```

如果 TinyLlama 首次下载很慢，第一次结果会包含下载时间。下载完后再跑一次：

```bash
python model_compare.py
```

第二次结果更适合作为冷启动体感比较。

---

## 记录结果

```text
Qwen/Qwen2.5-0.5B-Instruct

加载耗时：       ____ 秒
首次请求耗时：   ____ 秒
第二次请求耗时： ____ 秒
加载后显存：     ____ MB

TinyLlama/TinyLlama-1.1B-Chat-v1.0

加载耗时：       ____ 秒
首次请求耗时：   ____ 秒
第二次请求耗时： ____ 秒
加载后显存：     ____ MB
```

---

## 你要写的不是 benchmark 报表

你要写这种土但有用的观察：

```markdown
# Day 11 模型体感笔记

## 哪个模型冷启动更慢？

我的观察：

____

## 哪个模型首次请求更慢？

我的观察：

____

## 哪个模型让我更想压小 max_model_len？

我的观察：

____

## 哪个模型更适合第一阶段 smoke test？

我的观察：

____

## 这次我学到的系统直觉

模型变大后，我明显感觉到：

____
```

---

# 5. Day 12：回看 V1 guide，纠正过时印象

## 目标

今天你不深入源码。

你只做一件事：

> 把自己脑子里可能过时的 vLLM 印象清理一下。

官方 V1 guide 里，prefix caching 和 chunked prefill 都列为 functional；同一页还写明 V1 已移除 GPU <> CPU KV Cache Swapping，并说明 V1 不再需要 KV cache swapping 来处理 request preemption。([vLLM][7])

另外，官方 Paged Attention 文档现在明确标了 warning：它是基于原始 vLLM paper 的历史文档，不再描述今天 vLLM 使用的代码。([vLLM][8])

---

## 新建纠偏笔记

```bash
nano v1_cleanup_notes.md
```

粘贴：

```markdown
# Day 12：vLLM V1 纠偏笔记

## 1. 我确认 V1 里这些能力是当前重点

- Prefix Caching：Functional
- Chunked Prefill：Functional
- FP8 KV Cache：Functional
- Spec Decode：Functional

我的理解：

我前面做 APC 和 chunked prefill 实验，不是在学边缘功能，而是在学 V1 里真实重要的 serving 行为。

---

## 2. 我要清理的旧印象：GPU <> CPU KV cache swapping

旧印象：

有些旧文章或旧视频会讲 request preemption 时把 KV cache swap 到 CPU。

今天我更新后的理解：

V1 guide 明确说 GPU <> CPU KV Cache Swapping 已经移除。
在 V1 里，不能再把旧版本的 swapping 机制当成当前主线理解。

---

## 3. 我要清理的旧印象：PagedAttention 文档要放到最前面读

旧印象：

一学 vLLM 就必须先深挖 PagedAttention kernel。

今天我更新后的理解：

官方 Paged Attention 文档已经提示它是历史文档，不代表今天的代码。
所以我应该先建立全局 serving 感觉，再回来看 attention kernel。
现在还不是拆 kernel 的时候。

---

## 4. 我现在的学习顺序

我的顺序应该是：

1. 先看请求路径
2. 再看 engine / scheduler / KV cache
3. 再通过实验理解 APC、chunked prefill、token budget
4. 再做 benchmark
5. 最后再深入源码和 attention kernel

---

## 5. 我今天纠正掉的一个误区

误区：

____

新的理解：

____
```

---

# 6. 明天最终复盘

最后新建总复盘：

```bash
nano day8_12_review.md
```

粘贴并填写：

````markdown
# vLLM 第 8–12 天压缩复盘

## 1. LLMEngine 给我的新感觉

我看到的调用模式是：

```text
add_request()
step()
step()
step()
finished
````

我的理解：

---

---

## 2. continuous batching 给我的新感觉

我现在理解 continuous batching 是：

---

它和前面实验的关系是：

---

---

## 3. KV cache 给我的新感觉

我现在理解 KV cache 是：

---

它和 APC 的关系是：

---

---

## 4. 第一次 benchmark 我学到了什么？

我这次固定了：

* 模型：____
* prompt 模式：____
* 并发 / 输出长度：____

我只改了：

* APC 实验：____
* token budget 实验：____

我的理解：

---

---

## 5. 模型从 0.5B 到 1.1B 后，系统行为有没有变味？

我的观察：

---

---

## 6. V1 guide 帮我纠正了什么？

我纠正掉的旧印象：

---

新的理解：

---

---

## 7. 我现在脑中的 vLLM 第一层地图

```text
Client / curl
  -> API server
  -> input processing
  -> engine core
  -> scheduler
  -> KV cache
  -> GPU worker
  -> output
```

现在我能解释：

---

````

---

# 明天结束标准

明天结束时，你只要能说顺这 5 句话，就算完成。

## 1. LLMEngine

```text
server 只是外层 HTTP 入口，内部真正推进请求的是 LLMEngine。
````

## 2. continuous batching

```text
continuous batching 的核心不是把请求静态拼成 batch，而是在生成过程中不断重新组织 batch。
```

## 3. KV cache

```text
KV cache 是为了避免重复计算历史 token；APC 是跨请求复用共享前缀的 KV cache。
```

## 4. benchmark

```text
我开始知道怎么做最基本的性能对比：固定模型、固定 prompt、一次只改一个参数。
```

## 5. V1 纠偏

```text
我不能拿旧 vLLM 文章里的机制直接套到 V1；尤其是 GPU <> CPU KV swapping 和 PagedAttention 历史文档，要放在正确位置理解。
```

---

# 明天不要做什么

明天不要：

```text
不要读主干源码
不要深挖 attention kernel
不要追求论文级 benchmark
不要同时改多个参数
不要比较模型谁更聪明
不要被旧文章牵着走
```

明天只抓这些感觉：

```text
engine 怎么推进请求
continuous batching 为什么有用
KV cache 为什么是 serving 的核心状态
benchmark 怎么控制变量
模型变大后系统体感怎么变
V1 和旧资料哪里不能混着理解
```

[1]: https://docs.vllm.ai/en/latest/examples/deployment/llm_engine_example/ "LLM Engine Example - vLLM"
[2]: https://docs.vllm.ai/en/v0.9.0/api/vllm/benchmarks/ "vllm.benchmarks - vLLM"
[3]: https://www.anyscale.com/blog/continuous-batching-llm-inference "Achieve 23x LLM Inference Throughput & Reduce p50 Latency"
[4]: https://huggingface.co/blog/not-lain/kv-caching "KV Caching Explained: Optimizing Transformer Inference Efficiency"
[5]: https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/ "serve - vLLM"
[6]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?utm_source=chatgpt.com "Qwen/Qwen2.5-0.5B-Instruct"
[7]: https://docs.vllm.ai/en/stable/usage/v1_guide/ "vLLM V1 - vLLM"
[8]: https://docs.vllm.ai/en/latest/design/paged_attention/?utm_source=chatgpt.com "Paged Attention - vLLM"
