from Resample import *
import numpy as np
import pylab as pl

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