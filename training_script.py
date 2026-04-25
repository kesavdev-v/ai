import random
import pickle
import matplotlib.pyplot as plt
from env import SupplyChainEnv, Action

ACTIONS = ["order_low", "order_medium", "order_high", "negotiate", "diversify"]

# Q-learning params
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2


def get_state(obs):
    # richer state representation
    return f"{round(obs.inventory)}_{round(obs.cost,1)}_{round(obs.risk,2)}_{obs.step}"


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    q_vals = [Q.get((state, a), 0.0) for a in ACTIONS]
    return ACTIONS[q_vals.index(max(q_vals))]


def run_episode(env, agent_fn):
    obs = env.reset()
    total = 0
    done = False

    while not done:
        action = agent_fn(obs)
        obs, reward_obj, done, _ = env.step(Action(action))
        total += reward_obj.value

    return total


def train(task="medium", episodes=1000):
    rewards = []

    for ep in range(episodes):
        env = SupplyChainEnv(task)
        obs = env.reset()

        total = 0
        done = False

        while not done:
            state = get_state(obs)
            action = choose_action(state)

            obs2, reward_obj, done, _ = env.step(Action(action))
            reward = reward_obj.value

            next_state = get_state(obs2)

            old_q = Q.get((state, action), 0.0)
            next_max = max([Q.get((next_state, a), 0.0) for a in ACTIONS])

            # Q-learning update
            Q[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

            obs = obs2
            total += reward

        rewards.append(total)

        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {total:.2f}")

    return rewards


def random_agent(obs):
    return random.choice(ACTIONS)


def trained_agent(obs):
    state = get_state(obs)
    q_vals = [Q.get((state, a), 0.0) for a in ACTIONS]

    if all(v == 0 for v in q_vals):
        return random.choice(ACTIONS)

    return ACTIONS[q_vals.index(max(q_vals))]


def evaluate(agent_fn, episodes=50):
    scores = []
    for _ in range(episodes):
        env = SupplyChainEnv("medium")
        scores.append(run_episode(env, agent_fn))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    print("🚀 Training RL agent...")
    rewards = train(episodes=1500)

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)

    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.savefig("training_curve.png")

    # Evaluate baseline vs trained
    print("\n📊 Evaluating agents...")
    random_score = evaluate(random_agent)
    trained_score = evaluate(trained_agent)

    print(f"Random Agent Avg Reward:   {random_score:.2f}")
    print(f"Trained Agent Avg Reward: {trained_score:.2f}")
    print(f"Improvement:              {trained_score - random_score:.2f}")

    # Save comparison plot
    plt.figure()
    plt.bar(["Random", "Trained"], [random_score, trained_score])
    plt.ylabel("Average Reward")
    plt.title("Baseline vs Trained")
    plt.savefig("comparison.png")

    print("✅ Training complete. Files saved:")
    print("- q_table.pkl")
    print("- training_curve.png")
    print("- comparison.png")