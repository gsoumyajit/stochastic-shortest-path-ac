import numpy as np
from scipy.special import softmax
from env3 import CustomEnv

from numpy.random import choice,randint
from collections import deque
import sys,os,time

nS=4
nA=2

logrd="data/acfa2/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

a=lambda n:0.01*np.log(n+2)/(n+2)
b=lambda n:0.01/(n+1)

K=10000
value=np.array([0.0,0.0])
theta=np.array([0.0,0.0,0.0])

def feat(state):
    res=np.zeros(2)
    if state==3: return res
    if state==0: res[0]=1
    else: res[1]=1
    return res

def phi(state,action):
    res=np.zeros(3)
    if state==3: return res
    if state==0: res[action]=1
    else: res[2]=1
    return res

N=100000000
start_time=time.time()
env=CustomEnv()
returns=deque(maxlen=10000)
epsilon=0.1
fr.write("timestep\tv0\tv1\treturn\n")
t=0
n=0
res="\t".join(str(item) for item in value)
fr.write(str(t)+"\t"+res+"\t"+str(0)+"\n")
while t<=N:
    state=env.reset()
    ret=0
    n+=1
    value1=value[:]
    theta1=theta[:]
    while True:
        t+=1
        probs=softmax([np.dot(theta,phi(state,k)) for k in range(nA)])
        probs1=(epsilon/nA)*np.ones(nA)+(1-epsilon)*softmax([np.dot(theta,phi(state,k)) for k in range(nA)])
        action=choice(nA,p=probs1/np.sum(probs1))
        next_state,reward,done,_=env.step(action)

        dk=reward+np.dot(value,feat(next_state))-np.dot(value,feat(state))
        value1+=a(n//K+1)*dk*feat(state)
        psi=(1-epsilon)*(probs[action]/probs1[action])*(phi(state,action)-np.sum([phi(state,k)*probs[k] for k in range(nA)]))
        theta1+=b(n//K+1)*dk*psi

        ret+=reward
        
        if t%100000==0 and t<=N:
            mean=np.mean(returns)
            res="\t".join(str(item) for item in value)
            fr.write(str(t)+"\t"+res+"\t"+str(mean)+"\n")
            fr.flush()
            print(t,":",value,mean)

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




