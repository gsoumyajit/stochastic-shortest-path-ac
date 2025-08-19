import numpy as np
from scipy.special import softmax
from env2 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os,time

nS=20
nA=4

logrd="data/ca/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

b=lambda n:0.001*np.log(n+2)/(n+2)
a=lambda n:0.001/(n+1)

K=10000
value=np.zeros(nS)
theta=np.zeros((nS,nA))

N=100000000
start_time=time.time()
env=CustomEnv()

returns=deque(maxlen=10000)
vstep=np.ones(nS)
pstep=np.ones((nS,nA))
fr.write("timestep\treturn\tverror\n")
t=0
while t<=N:
    state=env.reset()
    ret=0
    while True:
        t+=1
        probs=softmax(theta[state])
        action=choice(nA,p=probs/np.sum(probs))
        next_state,reward,done,_=env.step(action)
        
        state1=randint(nS-1)
        probs=softmax(theta[state1])
        action1=choice(nA,p=probs/np.sum(probs))
        next_state1,reward1,_,_=env.sample(state1,action1)

        state2=randint(nS-1)
        probs=softmax(theta[state2])
        action2=choice(nA,p=probs/np.sum(probs))
        next_state2,reward2,_,_=env.sample(state2,action2)

        value[state1]+=a(vstep[state1]//K+1)*(reward1+value[next_state1]-value[state1])
        theta[state2,action2]-=b(pstep[state2,action2]//K+1)*(reward2+value[next_state2]-value[state2])
        vstep[state1]+=1
        pstep[state2,action2]+=1
        ret+=reward
        
        if t%100000==0 and t<=N:
            mean=np.mean(returns) if len(returns)>0 else 0
            error=np.linalg.norm(value-env.V)
            fr.write(str(t)+"\t"+str(mean)+"\t"+str(error)+"\n")
            fr.flush()
            print(t,":",mean,error)

        state=next_state
        if done: break
    returns.append(ret)

end_time=time.time()
time_elapsed=end_time-start_time
fr.write(str(t)+"\t"+str(time_elapsed)+"\n")
fr.flush()
print("Time elapsed:",time_elapsed)

fr.close()




