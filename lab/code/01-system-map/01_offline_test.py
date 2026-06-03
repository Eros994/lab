from vllm import LLM, SamplingParams

prompts = [
    "one sentence for daodejing",
    "one sentence for zhuangzi",
    "one sentence for mozi"
]

sampling_params = SamplingParams(
    temperature=0.0, 
    max_token=64,
)

llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct", 
    max_model_len=2048, 
    gpu_memory_utilization=0.85,
)

outputs = llm.generate(prompts, sampling_params)

for i, out in enumerate(outputs):
    print("=" * 40)
    print(f"Prompt {i}:{out.prompt}")
    print(out.outputs[0].text)