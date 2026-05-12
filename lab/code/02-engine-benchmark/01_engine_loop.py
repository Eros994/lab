import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser

def create_prompts() -> List[str]:
    return [
        (
            "please one sentence for daodejing",
            SamplingParams(temperature=0.0, max_tokens=48),
        ), 
        (
            "please one sentence for zhuangzi",
            SamplingParams(temperature=0.0, max_tokens=48),
        ), 
        (
            "please one sentence for mozi",
            SamplingParams(temperature=0.0, max_tokens=48),
        )
    ]

def process_requests(engine: LLMEngine, prompts: List[Tuple[str, SamplingParams]])->None:
    request_id = 0
    step_id = 0

    print("\n begine engine loop, press Ctrl+C to stop \n")
    print("="*60)

    while prompts or engine.has_unfinished_requests():
        print(f"Step {step_id}:")

        if prompts:
            prompt, sampling_params = prompts.pop(0)
            print(f"add")










