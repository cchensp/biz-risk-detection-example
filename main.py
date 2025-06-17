import json
import time

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from typing import Union

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

import asyncio
import uuid
import math

app = FastAPI()
model_name = "meta-llama/Llama-3.1-8B-Instruct"

async def gen(engine, prompt, params):
    """
    Generate the logprobs of the first token
    """
    results_generator = engine.generate(
        prompt,
        params,
        uuid.uuid4(),
    )

    final_output = None

    async for request_output in results_generator:
        final_output = request_output

    logit_output = [output.logprobs[0] for output in final_output.outputs]
    return logit_output[0]


class Chat(BaseModel): # input keys
    llm_answer: str
    threshold: float
    intent_policy_matching: list[dict]

@app.get('/test')
async def hello():
    return "hello"


@app.post("/risk-detection")
async def judge(request: Chat):
    """
    request:
    {
        "llm_answer": "...", # the answer from LLM
        "threshold": 0.5,
        "intent_policy_matching": [{"policy_id": "...", "policy": "..."}] # the intent-policy matching from policy KB
    }
    """
    try:
        df_intent_policy = pd.DataFrame(request.intent_policy_matching)
        if len(df_intent_policy) == 0:
            is_risky = False
            against_policy = []
        else:
            df_intent_policy = df_intent_policy.drop_duplicates(subset=['policy_id'])
            policy_id_list = []
            policy_list = []
            scores = []
            tasks = []
            # set the sampling params to output the logprobs
            sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=15)
            for _, row in df_intent_policy.iterrows():
                prompt = """Given a passage from customer service chatbot to user and a company policy, determine whether the passage violates the policy based on the hint below.
**Hint:** 
1. The passage violates the policy if the policy say there is no such service but the passage still tells user to use this service.
2. The passage violates the policy if the number or fact in the policy is not consistent with the passage.

<passage>
{}

<policy>
{}

**Instructions:**
Does the passage violate the policy? Your answer must be 'Yes' or 'No'."""
                answer, policy = request.llm_answer, row["policy"]
                answer = answer.strip().strip('\n')
                policy = policy.strip().strip('\n')
                input_text = prompt.format(answer, policy)
                input_json = [{"role": "user", "content": input_text}]
                formatted_input = tokenizer.apply_chat_template(input_json, tokenize=False, add_generation_prompt=True)
                tasks.append(asyncio.create_task(gen(model, formatted_input, sampling_params)))

                policy_id_list.append(row['policy_id'])
                policy_list.append(row['policy'])

            outputs = [await task for task in tasks]
            
            for logprobs in outputs:
                if yes_loc in logprobs:
                    scores.append(logprobs[yes_loc].logprob)
                else:
                    scores.append(-100)

            scores = [math.exp(s) for s in scores]
            thres = float(request.threshold)

            # if the max score is greater than the threshold, the answer is risky
            if max(scores) > thres:
                is_risky = True
                against_policy = [{'policy': policy_list[i], 'policy_id': policy_id_list[i], 'score': score} for i, score in enumerate(scores) if score > thres]
            else:
                is_risky = False
                against_policy = []

        content = {
            'is_risky': is_risky,
            'against_policy': against_policy
        }

        return content

    except Exception as error:
        result = {'code': 500}

        return result



try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the model in bf16, change the dtype to fp16 and disable prefix caching if GPU can't support
    model = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_name, 
            tensor_parallel_size=1, 
            dtype=torch.bfloat16, 
            max_model_len=4096, 
            enable_prefix_caching=True, 
            gpu_memory_utilization=0.95,
            max_num_seqs=128,
        )
    )
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
except Exception as e:
    print('exception in main task: exception=%s', e)
