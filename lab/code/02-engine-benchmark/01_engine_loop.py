"""
python engine_loop_lab.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
"""

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
            print(f"add_request: request_id={request_id}, prompt={prompt[:30]}...")
            engine.add_request(
                str(request_id),
                prompt,
                sampling_params,
            )
            request_id += 1

        outputs: List[RequestOutput]  = engine.step()

        if not outputs:
            print("no output yet")
        else:
            for output in outputs:
                print(f"收到 output: request_id={output.request_id}, output={output.text[:30]}...")
                
                if output.is_finished:
                    text = output.outputs[0].text
                    print("-"*40)
                    print(text.strip())
                    print("-"*40)
        step_id += 1

    print("\n engine loop finished \n")   

def main() -> None:
    parser = FlexibleArgumentParser(description="LLM Engine add_request / step demo")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    prompts = create_prompts()
    process_requests(engine, prompts)


if __name__ == "__main__":
    main()
