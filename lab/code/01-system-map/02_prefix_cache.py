import time
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

shared_prefix = """
# 公司知识库
产品名称: AlphaServe
特点:
1. 支持高吞吐推理
2. 支持 prefix caching
3. 支持 continuous batching
4. 支持长上下文
5. 支持 OpenAI-compatible server
""" * 60

prompt1 = shared_prefix + "please introduce the difference between vllm and sglang。"
prompt2 = shared_prefix + "please explain the benefits of using prefix caching in vllm。"

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

def run_experiment(enable_apc:bool):
    print("="*60)
    print(f"Running experiment with enable_apc={enable_apc}")
    print("="*60)

    llm = LLM(
        model = MODEL,
        max_model_len = 4096,
        gpu_memory_utilization = 0.85,
        enable_prefix_caching = enable_apc,
    )

    llm.generate(["只回答ok"],  SamplingParams(max_tokens=8))

    t1 = time.perf_counter()
    llm.generate([prompt1], sampling_params)
    t2 = time.perf_counter()

    llm.generate([prompt2], sampling_params)
    t3 = time.perf_counter()

    first_time = t2 - t1
    second_time = t3 - t2

    print(f"APC Enabled: {enable_apc}, First Generation Time: {first_time:.2f} seconds, Second Generation Time: {second_time:.2f} seconds")

if __name__ == "__main__":
    run_experiment(enable_apc=False)
    run_experiment(enable_apc=True) 


# python 02_prefix_cache.py






















