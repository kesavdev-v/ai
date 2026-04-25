from env import SupplyChainEnv, Action
import random

ACTIONS = ["order_low", "order_medium", "order_high", "negotiate", "diversify"]


def random_agent(obs):
    return random.choice(ACTIONS)


def run(agent):
    env = SupplyChainEnv("medium")
    obs = env.reset()
    total = 0

    done = False
    while not done:
        action = agent(obs)
        obs, r, done, _ = env.step(Action(action))
        total += r.value

    return total


print("Before training (random):")
print(run(random_agent))

print("\nAfter training (expected improved decisions):")
print("Use inference.py with trained model")