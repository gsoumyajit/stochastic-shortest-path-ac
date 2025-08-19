import numpy as np
from scipy.special import softmax
import gymnasium as gym
from numpy.random import choice,randint
from collections import deque
import sys,os,time

nS=16
nA=4

logrd="data/ac2/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

a=lambda n:0.1*np.log(n+2)/(n+2)
b=lambda n:0.1/(n+1)

K=10000
value=np.zeros(nS)
theta=np.zeros((nS,nA))

N=100000000
start_time=time.time()
env=gym.make("FrozenLake-v1",map_name="4x4")

returns=deque(maxlen=10000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\n")
t=0
while t<=N:
    state,_=env.reset()
    ret=0
    while True:
        t+=1
        probs=softmax(theta[state])
        action=choice(nA,p=probs/np.sum(probs))
        next_state,reward,done,_,_=env.step(action)

        value[state]+=a(vstep[state]//K+1)*(reward+value[next_state]-value[state])
        theta[state,action]+=b(pstep[state,action]//K+1)*(reward+value[next_state]-value[state])
        vstep[state]+=1
        pstep[state,action]+=1
        ret+=reward
        
        if t%100000==0 and t<=N:
            mean=np.mean(returns) if len(returns)>0 else 0
            fr.write(str(t)+"\t"+str(mean)+"\n")
            fr.flush()
            print(t,":",mean)

        state=next_state
        if done: break
    returns.append(ret)

end_time=time.time()
time_elapsed=end_time-start_time
fr.write(str(t)+"\t"+str(time_elapsed)+"\n")
fr.flush()
print("Time elapsed:",time_elapsed)

fr.close()




