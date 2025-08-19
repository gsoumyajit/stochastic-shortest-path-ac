import numpy as np
import gym
from tqdm import tqdm

if __name__=="__main__":
    nS=20
    nA=4
    i0=nS-1
    R=np.random.randint(1,20,size=(nA,nS,nS))
    Pr=np.random.randint(1,20,size=(nA,nS,nS)).astype(float)
    Pr[:,i0,:]=0
    Pr[:,i0,i0]=1
    R[:,i0,i0]=0
    for i in range(nA):
        for j in range(nS):
            Pr[i,j]=Pr[i,j]/np.sum(Pr[i,j])
    
    iters=1000
    state0=0
    values=np.zeros(nS)
    policy=np.zeros(nS)
    for it in range(iters):
        value=np.zeros(nS)
        for i in range(nS-1):
            policy[i]=np.argmin([np.dot(Pr[a,i],R[a,i]+values) for a in range(nA)])
            value[i]=np.min([np.dot(Pr[a,i],R[a,i]+values) for a in range(nA)])
        values=np.copy(value)
        print(it,":",values)

    np.save("mdp/R1.npy",R)
    np.save("mdp/Pr1.npy",Pr)
    np.savetxt("mdp/value1",values)
    np.savetxt("mdp/policy1",policy)

class CustomEnv(gym.Env):
    def __init__(self):
        self.R=np.load("mdp/R1.npy")
        self.Pr=np.load("mdp/Pr1.npy")
        self.V=np.loadtxt("mdp/value1")
        self.pol=np.loadtxt("mdp/policy1").astype(int)
        self.nS=20
        self.nA=4
        self.i0=self.nS-1
        print(self.V)


    def reset(self):
        self.state=0
        return self.state

    def step(self,action):
        next_state=np.random.choice(self.nS,p=self.Pr[action,self.state])
        reward=self.R[action,self.state,next_state]
        done=next_state==self.i0
        self.state=next_state
        return self.state,reward,done,None

    def sample(self,state,action):
        next_state=np.random.choice(self.nS,p=self.Pr[action,state])
        reward=self.R[action,state,next_state]
        done=next_state==self.i0
        return next_state,reward,done,None

