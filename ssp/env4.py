import numpy as np
import gym
from tqdm import tqdm


class CustomEnv(gym.Env):
    def __init__(self):
        self.nS=3
        self.nA=2
        self.R=np.array([[[0,0,-2],[0,0,-2],[0,0,0]],[[0,0,-1],[0,0,-1],[0,0,0]]])

        self.P=np.array([[[0,0.9,0.1],[0,0.9,0.1],[0,0,1]],[[0.9,0,0.1],[0.9,0,0.1],[0,0,1]]])


    def reset(self):
        self.state=np.random.randint(2)
        return self.state

    def step(self,action):
        next_state=np.random.choice(self.nS,p=self.P[action,self.state])
        reward=self.R[action,self.state,next_state]
        self.state=next_state
        done=self.state==2
        return self.state,reward,done,None


