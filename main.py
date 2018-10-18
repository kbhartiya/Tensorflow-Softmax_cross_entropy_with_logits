#Implementation of softmax_cross_entropy_with_logits()
import numpy as np 
#Logits as last layer outputs
logits = [0.1,0.4,0.2,0.3,0.001,0.2]
sum=0
for i in range(len(a)):
    sum =sum + float(np.exp(logits[i]))
ce = 0
    
print("Sum\n")    
print(sum)
soft_p = []
for i in range(len(a)):
    soft_p.append(np.exp(logits[i])/sum)
print("\nProbability list:") 
print("\n {}".format(soft_p))
print("\nprediction")
pr = np.argmax(soft_p)
pred = np.zeros((1,6))
for i in range(len(a)):
    if(i==pr):
      pred[0,i]=1
print("\n {}".format(pred))      
  
for i in range(len(a)):
    ce= ce - np.log(soft_p[i])*soft_p[i]  
print("\ncross entropy loss")
print("\n {}".format(ce))
