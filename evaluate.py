import random, pickle
from env import SupplyChainEnv, Action

ACTIONS = ["order_low","order_medium","order_high","negotiate","diversify"]
Q = pickle.load(open("q.pkl","rb"))

def state(o): return f"{o.inventory:.0f}_{o.cost:.1f}_{o.risk:.2f}_{o.step}"

def trained(o):
    q=[Q.get((state(o),a),0) for a in ACTIONS]
    return ACTIONS[q.index(max(q))]

def random_agent(o): return random.choice(ACTIONS)

def run(agent):
    env=SupplyChainEnv("medium")
    o=env.reset()
    total=0
    done=False
    while not done:
        o,r,done,_=env.step(Action(agent(o)))
        total+=r.value
    return total

for _ in range(20):
    print("Random:", run(random_agent), "Trained:", run(trained))