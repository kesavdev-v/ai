from fastapi import FastAPI
from pydantic import BaseModel
from env import SupplyChainEnv, Action

app = FastAPI()
env = None

class Act(BaseModel):
    action: str

@app.post("/reset")
def reset():
    global env
    env = SupplyChainEnv("easy")
    o = env.reset()
    return {"observation": o.dict(), "done": False}

@app.post("/step")
def step(a: Act):
    o,r,d,i = env.step(Action(a.action))
    return {"observation": o.dict(),"reward":r.value,"done":d,"info":i}

@app.get("/state")
def state():
    return env.state()
