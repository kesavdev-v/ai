import asyncio
import os
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from env import SupplyChainEnv, Action
from grader import grade

ACTIONS = ["order_low", "order_medium", "order_high", "negotiate", "diversify"]

TASK_NAME = os.getenv("TASK", "easy")
MODEL_PATH = "./trained_model"
ENV_NAME = "supplychain-openenv"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def llm_decide(obs):
    prompt = f"""
State:
{obs.dict()}

Transcript:
{obs.transcript}

Choose best action from:
{ACTIONS}
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.4,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    for a in ACTIONS:
        if a in text:
            return a

    return "order_medium"


async def main():
    env = SupplyChainEnv(TASK_NAME)
    obs = env.reset()

    rewards: List[float] = []
    total_reward = 0
    steps_taken = 0

    log_start(TASK_NAME, ENV_NAME, MODEL_PATH)

    done = False

    while not done:
        steps_taken += 1

        action = llm_decide(obs)

        obs, reward_obj, done, info = env.step(Action(action=action))
        reward = reward_obj.value

        rewards.append(reward)
        total_reward += reward

        log_step(steps_taken, action, reward, done)

    score = grade(info, total_reward, steps_taken)
    success = score > 0.3

    env.close()
    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())