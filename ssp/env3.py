import numpy as np
import gym
from tqdm import tqdm


class CustomEnv(gym.Env):
    def __init__(self):
        self.nS=4

    def reset(self):
        self.state=0
        return self.state

    def step(self,action):
        if self.state==0:
            self.state=action+1
            reward=0
        else:
            reward=-2 if self.state==1 else -1
            self.state=3
        done=self.state==3
        return self.state,reward,done,None


