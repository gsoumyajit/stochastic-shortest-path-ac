import numpy as np
from scipy.special import softmax
from env1 import CustomEnv
from numpy.random import choice,randint
from collections import deque
import sys,os,time

nS=3
nA=2

logrd="data/acfa1/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

a=lambda n:0.1*np.log(n+2)/(n+2)
b=lambda n:0.1/(n+1)

K=1
value=np.array([-2.0])
theta=np.array([-2.0,-1.0])

def feat(state):
    res=np.zeros(1)
    if state==2: return res
    res[0]=state+1
    return res

def phi(state,action):
    res=np.zeros(2)
    if state==2: return res
    res[action]=state+1
    return res
N=100000000
start_time=time.time()
env=CustomEnv()
returns=deque(maxlen=10000)

fr.write("timestep\tv0\n")
t=0
n=0
fr.write(str(t)+"\t"+str(value[0])+"\n")
while t<=N:
    state=env.reset()
    ret=0
    n+=1
    value1=value[:]
    theta1=theta[:]
    while True:
        t+=1
        probs=softmax([np.dot(theta,phi(state,k)) for k in range(nA)])
        action=choice(nA,p=probs/np.sum(probs))

        next_state,reward,done,_=env.step(action)

        dk=reward+np.dot(value,feat(next_state))-np.dot(value,feat(state))
        value1+=a(n//K+1)*dk*feat(state)
        psi=phi(state,action)-np.sum([phi(state,k)*probs[k] for k in range(nA)])
        theta1-=b(n//K+1)*dk*psi

        ret+=reward
        
        if t%100000==0 and t<=N:
            fr.write(str(t)+"\t"+str(value[0])+"\n")
            fr.flush()
            print(t,":",value)

        state=next_state
        if done: break
    value=value1[:]
    theta=theta1[:]
    returns.append(ret)

end_time=time.time()
time_elapsed=end_time-start_time
fr.write(str(t)+"\t"+str(time_elapsed)+"\n")
fr.flush()
print("Time elapsed:",time_elapsed)

fr.close()




