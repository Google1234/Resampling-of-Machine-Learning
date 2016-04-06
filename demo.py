
from Resample import *
import pylab as pl
import numpy as np
'''
import numpy as np


(dataMat,Classlabel,point)=LoadSimpData()
class1_x=[]
class1_y=[]
class2_x=[]
class2_y=[]
for i in range(len(Classlabel)):
    if Classlabel[i]==1.0:
        class1_x.append(point[i][0])
        class1_y.append(point[i][1])
    else:
        class2_x.append(point[i][0])
        class2_y.append(point[i][1])

pl.xlim(0.0, 3.0)# set axis limits
pl.ylim(0.0, 3.0)
pl.plot(class1_x, class1_y,'or')# use pylab to plot x and y
pl.plot(class2_x, class2_y,'og')# use pylab to plot x and y

weakClassArr=adaBoostTrainDS(dataMat,Classlabel,9)
print(weakClassArr[1])

if weakClassArr[0]['dim']==0:
    x=[weakClassArr[0]['thresh'],weakClassArr[0]['thresh']]
    y=[0,3]
    pl.plot(x,y,label="$第1次$",color="blue",linewidth=2)
else :
    y=[weakClassArr[0]['thresh'],weakClassArr[0]['thresh']]
    x=[0,3]
    pl.plot(x,y,label="$第1次$",color="blue",linewidth=2)
if weakClassArr[1]['dim']==0:
    x=[weakClassArr[1]['thresh'],weakClassArr[1]['thresh']]
    y=[0,3]
    pl.plot(x,y,label="$第2次$",color="green",linewidth=2)
else :
    y=[weakClassArr[1]['thresh'],weakClassArr[1]['thresh']]
    x=[0,3]
    pl.plot(x,y,label="$第2次$",color="green",linewidth=2)
if weakClassArr[2]['dim']==0:
    x=[weakClassArr[2]['thresh'],weakClassArr[2]['thresh']]
    y=[0,3]
    pl.plot(x,y,label="$第3次$",color="red",linewidth=2)
else :
    y=[weakClassArr[2]['thresh'],weakClassArr[2]['thresh']]
    x=[0,3]
    pl.plot(x,y,label="$第3次$",color="red",linewidth=2)
#pl.plot(class1_x,class1_y,label="$N(1,1)$",color="red",linewidth=2)

pl.show()# show the plot on the screen
'''

'''
bagging 回归测试
'''
numbers=500
x=[random.uniform(0,3) for i in range(numbers)]
y=[x[i]*1+random.random() for i in range(numbers)]
point=[[x[i],y[i]] for i in range(numbers)]
pl.xlim(0.0, 3.0)# set axis limits
pl.ylim(0.0, 3.0)
pl.plot(x, y,'or')# use pylab to plot x and y
point_x=np.linspace(0.0,3,2)

def test(point,numIt):
    bagClass=bagging_regression(point,numIt)
    k=b=0
    for i in range(len(bagClass)):
        k+=bagClass[i][0]
        b+=bagClass[i][1]
    k=k/len(bagClass)
    b=b/len(bagClass)
    point_y=k*point_x+b
    return point_y

pl.plot(point_x,test(point,100),label="$10$",color="red",linewidth=2)
pl.plot(point_x,test(point,200),label="$20$",color="blue",linewidth=2)
pl.plot(point_x,test(point,500),label="$50$",color="green",linewidth=2)
pl.plot(point_x,test(point,1000),label="$50$",color="yellow",linewidth=2)
pl.show()


