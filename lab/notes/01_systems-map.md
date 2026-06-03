确实，上版太碎了。下面我重新排成一个**清爽版今日任务单**，你照着从上到下做就行。

---

# vLLM 第 4–7 天压缩版：今天一天做完

今天只做 4 件事：

1. 建立一张 vLLM 请求路径图
2. 跑一次 Automatic Prefix Caching 实验
3. 跑一次 chunked prefill / token budget 实验
4. 写一页复盘笔记

---

# 0. 准备目录

```bash
mkdir -p vllm-today-lab
cd vllm-today-lab
```

检查环境：

```bash
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
```

如果缺依赖：

```bash
pip install -U vllm requests
```

---

# 1. 建立系统地图

## 你今天只需要记住这张图

```text
Client / curl
    ↓
API server
    ↓
input processing
    ↓
engine core
    ↓
scheduler / KV cache
    ↓
GPU worker
    ↓
output
```

## 你要形成的理解

你的 `curl` 请求不是直接打到 GPU。

它大致会经历：

| 模块                   | 作用                    |
| -------------------- | --------------------- |
| Client / curl        | 发请求                   |
| API server           | 接收 HTTP 请求            |
| input processing     | 解析输入、tokenize         |
| engine core          | 管理推理流程                |
| scheduler / KV cache | 决定请求怎么排队、KV cache 怎么用 |
| GPU worker           | 真正在 GPU 上跑模型          |
| output               | 返回生成结果                |

今天不要看源码。

你只要能说出这句话就够了：

> 我发出的请求不是直接碰 GPU，而是先经过 API server，再进入 engine core，由 scheduler 管理 KV cache，最后由 GPU worker 执行模型计算。

---

# 2. 实验一：Automatic Prefix Caching

## 实验目标

你要亲眼看到：

> 两个请求有相同的长前缀时，第二个请求可能更快。
> 快的原因不是“答案被缓存了”，而是共享前缀的 KV cache 被复用了。

---

## 创建代码文件

新建文件：

```bash
nano apc_lab.py
```

粘贴下面代码：

```python
import time
from vllm import LLM, SamplingParams


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


shared_prefix = """
# 公司知识库

产品名称: AlphaServe

产品特点:
1. 支持高吞吐推理
2. 支持 prefix caching
3. 支持 continuous batching
4. 支持长上下文
5. 支持 OpenAI-compatible server

请严格根据上面的公司知识库回答问题。
""" * 60


prompt1 = shared_prefix + "\n问题：这个系统最重要的能力是什么？"
prompt2 = shared_prefix + "\n问题：这个系统为什么适合高并发？"


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)


def run_experiment(enable_apc: bool):
    print("\n" + "=" * 60)
    print(f"开始实验：enable_prefix_caching = {enable_apc}")
    print("=" * 60)

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=enable_apc,
    )

    # 预热，减少首次启动噪声
    llm.generate(["请只回答 OK。"], SamplingParams(max_tokens=8))

    t1 = time.perf_counter()
    llm.generate([prompt1], sampling_params)
    t2 = time.perf_counter()

    llm.generate([prompt2], sampling_params)
    t3 = time.perf_counter()

    first_time = t2 - t1
    second_time = t3 - t2

    print(f"APC = {enable_apc}")
    print(f"第一次请求耗时: {first_time:.3f} 秒")
    print(f"第二次请求耗时: {second_time:.3f} 秒")


if __name__ == "__main__":
    run_experiment(enable_apc=False)
    run_experiment(enable_apc=True)
```

---

## 运行

```bash
python apc_lab.py
```

---

## 你要观察什么

记录结果：

```text
APC=False
第一次请求耗时：____ 秒
第二次请求耗时：____ 秒

APC=True
第一次请求耗时：____ 秒
第二次请求耗时：____ 秒
```

重点不是数字本身，而是这个现象：

| 情况     | 你应该怎么理解                      |
| ------ | ---------------------------- |
| 第一次请求  | 必须处理完整长前缀                    |
| 第二次请求  | 如果 APC 开启，可以复用相同前缀的 KV cache |
| 变快的部分  | 主要是共享前缀的 prefill             |
| 没变快的部分 | 生成新 token 的 decode 阶段        |

---

## 这个实验要体会什么

你要体会的是：

> APC 不是缓存答案。
> APC 缓存的是共享前缀经过模型计算后的 KV cache。

比如这两个请求：

```text
共享前缀 + 问题 1
共享前缀 + 问题 2
```

APC 复用的是：

```text
共享前缀
```

不是复用：

```text
问题 1 的答案
```

---

# 3. 实验二：chunked prefill / token budget

## 实验目标

你要体会：

> 长 prompt 的 prefill 会占用计算资源。
> 短请求如果刚好排在长请求后面，可能会感觉被“顶住”。
> `max_num_batched_tokens` 会改变这种调度感觉。

---

# 3.1 启动 server：小 token budget

开一个终端，运行：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85
```

这个版本的特点是：

```text
max_num_batched_tokens = 1024
```

也就是每轮 token budget 比较小。

---

# 3.2 创建测试脚本

另开一个终端，仍然进入刚才的目录：

```bash
cd vllm-today-lab
nano chunked_prefill_lab.py
```

粘贴代码：

```python
import json
import threading
import time

import requests


BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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


def make_short_prompt():
    return "请用一句话解释 vLLM 为什么适合高并发推理。"


def request_once(prompt, label):
    url = f"{BASE_URL}/v1/completions"

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

        output.append(piece)

    end = time.perf_counter()

    ttft = None
    if first_token_time is not None:
        ttft = first_token_time - start

    return {
        "label": label,
        "ttft": ttft,
        "total": end - start,
        "output_chars": len("".join(output)),
    }


def print_result(result):
    ttft = result["ttft"]

    if ttft is None:
        ttft_text = "None"
    else:
        ttft_text = f"{ttft:.3f}s"

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
```

---

## 运行测试

```bash
python chunked_prefill_lab.py
```

记录结果：

```text
max_num_batched_tokens = 1024

SHORT alone TTFT:       ____ 秒
LONG alone TTFT:        ____ 秒
SHORT while long TTFT:  ____ 秒
短请求 TTFT 放大倍数:    ____ x
```

---

# 3.3 启动 server：大 token budget

回到 server 那个终端，按：

```bash
Ctrl-C
```

然后重新启动：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85
```

这个版本的特点是：

```text
max_num_batched_tokens = 4096
```

也就是每轮 token budget 更大。

---

## 再跑一次测试

```bash
python chunked_prefill_lab.py
```

记录结果：

```text
max_num_batched_tokens = 4096

SHORT alone TTFT:       ____ 秒
LONG alone TTFT:        ____ 秒
SHORT while long TTFT:  ____ 秒
短请求 TTFT 放大倍数:    ____ x
```

---

# 3.4 这个实验要体会什么

你要观察这两个现象。

## 现象一：长 prompt 会占资源

长请求的 prompt 很长。

所以模型在真正生成答案之前，要先处理大量输入 token。

这一步叫：

```text
prefill
```

你的直觉应该变成：

> 长 prompt 不是“只是输入长一点”。
> 它会实实在在占用推理计算资源。

---

## 现象二：短请求可能被长请求顶住

你会比较：

```text
SHORT alone
SHORT while long
```

如果：

```text
SHORT while long 的 TTFT 明显更长
```

说明短请求虽然自己很短，但它到来的时候，系统正在处理长请求，所以短请求体感上被顶住了。

---

## 现象三：token budget 会改变调度形态

你比较：

```text
max_num_batched_tokens = 1024
max_num_batched_tokens = 4096
```

你不需要得出“哪个一定更好”。

你只需要理解：

| token budget | 可能的感觉                         |
| ------------ | ----------------------------- |
| 小            | 长 prefill 更容易被切成小块，短请求可能更容易插入 |
| 大            | 长 prefill 可能更快完成，但也可能一次占用更多计算 |

今天你要记住这句话：

> chunked prefill 不是魔法。
> 它本质上是在 token budget 约束下，改变长 prefill 进入 scheduler 的方式。

---

# 4. 最后一页复盘

新建文件：

```bash
nano first_week_review.md
```

粘贴这个模板，然后把空白填上。

````markdown
# 我已经亲手看到的 vLLM 现象有哪些？

## 1. 离线 API 是什么样？

我用过：

```python
from vllm import LLM, SamplingParams
````

我的理解：

vLLM 的离线 API 可以直接在 Python 里加载模型，然后调用 `generate` 生成结果。
这种方式没有 HTTP server，适合实验、脚本和批处理。

---

## 2. server 是什么样？

我用过：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct
```

我的理解：

server 模式下，请求先进入 API server。
API server 解析请求后，把任务交给 engine core。
engine core 再通过 scheduler 和 GPU worker 完成推理。

---

## 3. prefix caching 的现象是什么？

我的实验结果：

APC=False：

* 第一次请求：____ 秒
* 第二次请求：____ 秒

APC=True：

* 第一次请求：____ 秒
* 第二次请求：____ 秒

我的理解：

APC 复用的是共享前缀的 KV cache。
它加速的是 prefill 阶段。
它不是缓存答案。

---

## 4. token budget / chunked prefill 的现象是什么？

我的实验结果：

max_num_batched_tokens = 1024：

* SHORT alone TTFT：____ 秒
* LONG alone TTFT：____ 秒
* SHORT while long TTFT：____ 秒

max_num_batched_tokens = 4096：

* SHORT alone TTFT：____ 秒
* LONG alone TTFT：____ 秒
* SHORT while long TTFT：____ 秒

我的理解：

长 prompt 会占用 prefill 计算。
短请求如果排在长请求后面，可能会感觉被顶住。
`max_num_batched_tokens` 会影响长 prefill 怎么进入调度。

---

## 5. API server / engine core / worker 在我脑中的分工是什么？

Client / curl：

负责发请求。

API server：

负责接收 HTTP 请求，解析输入。

input processing：

负责处理输入，例如 tokenize。

engine core：

负责管理推理流程。

scheduler / KV cache：

负责调度请求，管理 KV cache。

GPU worker：

负责在 GPU 上真正执行模型计算。

output：

负责把结果返回给用户。

````

---

# 今天结束标准

今天结束时，你只要能说出这三句话，就算完成。

## 第一句

> 我的 curl 请求不是直接打到 GPU，而是经过 API server、input processing、engine core、scheduler / KV cache，最后由 GPU worker 执行。

## 第二句

> APC 加速的是共享前缀的 prefill。它复用 KV cache，不是复用答案。

## 第三句

> chunked prefill 和 token budget 影响的是长 prompt 怎么进入调度，因此会影响短请求在长请求旁边的体感延迟。

---

# 今天不要做什么

今天不要：

```text
看源码
追求严谨 benchmark
纠结具体快了多少秒
深挖 block hash
深挖 scheduler 源码
````

今天只要抓住现象：

```text
请求经过哪些模块
APC 为什么让第二次共享前缀请求更快
长 prompt 为什么会顶住短请求
token budget 为什么会改变调度体感
```
