import random
import os
from dataclasses import dataclass, asdict
from openai import OpenAI

ACTIONS = ["order_low", "order_medium", "order_high", "negotiate", "diversify"]
AGENTS = ["CEO", "CFO", "CMO", "INVESTOR"]


@dataclass
class Observation:
    step: int
    demand: float
    inventory: float
    cost: float
    risk: float
    transcript: str
    consensus: float
    disagreement: float

    def dict(self):
        return asdict(self)


@dataclass
class Action:
    action: str


@dataclass
class Reward:
    value: float


class SupplyChainEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.max_steps = {"easy": 4, "medium": 5, "hard": 6}[task]
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
        )
        self.reset()

    def reset(self):
        self.step_count = 0
        self.inventory = 50
        self.cost = 10
        self.risk = random.uniform(0.05, 0.25)

        self.transcript = ""
        self.consensus = 0.0
        self.disagreement = 0.0

        return self._obs()

    def _obs(self):
        return Observation(
            step=self.step_count,
            demand=random.uniform(40, 60),
            inventory=self.inventory,
            cost=self.cost,
            risk=self.risk,
            transcript=self.transcript,
            consensus=self.consensus,
            disagreement=self.disagreement,
        )

    def state(self):
        return self._obs().dict()

    def agent_llm(self, agent, obs, transcript):
        personalities = {
            "CEO": "maximize growth and expansion",
            "CFO": "minimize cost and financial risk",
            "CMO": "maximize demand and market reach",
            "INVESTOR": "minimize risk and ensure stability",
        }

        try:
            resp = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {agent}. Your goal is to {personalities[agent]}."
                    },
                    {
                        "role": "user",
                        "content": f"""
State: {obs.dict()}

Discussion so far:
{transcript}

Give ONE decisive recommendation.
"""
                    },
                ],
                temperature=0.3,
                max_tokens=40,
            )
            return resp.choices[0].message.content.strip()
        except:
            return f"{agent}: maintain balance"

    def generate_transcript(self, obs):
        rounds = 3 if self.risk < 0.1 else 4 if self.risk < 0.2 else 5
        transcript = ""

        for r in range(rounds):
            transcript += f"\nRound {r+1}:\n"
            for agent in AGENTS:
                speech = self.agent_llm(agent, obs, transcript)
                transcript += f"{agent}: {speech}\n"

        return transcript

    def infer_action(self, text):
        text = text.lower()
        if "increase" in text or "scale" in text:
            return "order_high"
        if "cost" in text or "reduce" in text:
            return "negotiate"
        if "risk" in text:
            return "diversify"
        return "order_medium"

    def step(self, action: Action):
        self.step_count += 1
        obs = self._obs()

        self.transcript = self.generate_transcript(obs)

        votes = [
            self.infer_action(line)
            for line in self.transcript.split("\n")
            if any(line.startswith(a) for a in AGENTS)
        ]

        agreement = votes.count(action.action) / len(votes) if votes else 0
        self.consensus = agreement
        self.disagreement = 1 - agreement

        demand = random.uniform(40, 60)
        if random.random() < self.risk:
            demand *= 0.5

        order_map = {"order_low": 20, "order_medium": 40, "order_high": 60}
        order = order_map.get(action.action, 0)

        if action.action == "negotiate":
            self.cost *= 0.9
        if action.action == "diversify":
            self.risk *= 0.75
            self.cost *= 1.05

        self.inventory += order

        fulfilled = min(self.inventory, demand)
        self.inventory -= fulfilled
        shortage = max(0, demand - fulfilled)

        revenue = fulfilled * 15
        cost = order * self.cost
        penalty = shortage * 3 + self.risk * 5

        reward = (revenue - cost - penalty) / 100

        # 🔥 negotiation importance (strong)
        reward += self.consensus * 0.5
        reward -= self.disagreement * 0.3

        done = self.step_count >= self.max_steps

        return self._obs(), Reward(reward), done, {
            "efficiency": fulfilled / (demand + 1e-6),
            "shortage": shortage,
            "consensus": self.consensus,
        }

    def close(self):
        pass