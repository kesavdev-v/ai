from fastapi import FastAPI
from pydantic import BaseModel
from env import BoardroomEnv

app = FastAPI()
env_instance = None


class ActionInput(BaseModel):
    action: str


@app.post("/reset")
def reset():
    global env_instance
    env_instance = BoardroomEnv("easy")
    obs = env_instance.reset()

    return {"observation": obs.dict(), "done": False}


@app.post("/step")
def step(action: ActionInput):
    obs, reward, done, info = env_instance.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env_instance.state()


@app.get("/")
def home():
    return {"message": "Boardroom AI running"}