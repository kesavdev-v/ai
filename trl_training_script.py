import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from env import SupplyChainEnv, Action

ACTIONS = ["order_low", "order_medium", "order_high", "negotiate", "diversify"]


def generate_data(episodes=200):
    data = []

    for _ in range(episodes):
        env = SupplyChainEnv("medium")
        obs = env.reset()

        done = False
        while not done:
            prompt = f"""
State:
{obs.dict()}

Transcript:
{obs.transcript}

Choose best action:
"""

            # stronger heuristic (important)
            if obs.risk > 0.18:
                action = "diversify"
            elif obs.cost > 11:
                action = "negotiate"
            elif obs.inventory < 30:
                action = "order_high"
            else:
                action = "order_medium"

            data.append({"text": prompt + action})

            obs, _, done, _ = env.step(Action(action))

    return data


dataset = Dataset.from_list(generate_data())

model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./trained_model",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=10,
    ),
)

trainer.train()
trainer.save_model("./trained_model")

print("✅ Training done")