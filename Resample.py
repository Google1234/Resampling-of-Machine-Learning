from numpy import *

def LoadSimpData():
    datMat=matrix([[1.,2.1],
                  [ 2., 1.1],
                  [1.3,1.],
                  [1.,1.],
                  [2.,1.]]
                  )
    dat=[[1.,2.1],[ 2., 1.1],[1.3,1.],[1.,1.],[2.,1.]]
    classLbabel=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLbabel,dat
'''
单层决策树
'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq =='lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else :
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat( classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst =mat( zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin +float(j) * stepSize)
                predictedVals = \
                    stumpClassify(dataMatrix,i ,threshVal,inequal)
                errArr=mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D .T*errArr
                #print("split:dim %d, thresh %.2f, thresh i n e q a l： \ %s , the weighted error is %.3f "%(i, threshVal, inequal , weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim' ] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

'''
adaboost
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40 ):
    weakClassArr =[]
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst =mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print ("D:" ,D.T)
        alpha = float(0.5 * log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr .append(bestStump)
        print ("classEst :", classEst .T )
        expon = multiply(- 1*alpha * mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D /D .sum()
        aggClassEst += alpha*classEst
        print("aggClassEst : ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate =aggErrors.sum()/m
        print ("total error：",errorRate,"\n")
        if errorRate == 0.0:
            break
    return weakClassArr

'''
bagging：bootstrap aggregating的缩写。让该学习算法训练多轮，每轮的训练集由从初始的训练集中随机取出的n个训练样本组成，
某个初始训练样本在某轮训练集中可以出现多次或根本不出现，训练之后可得到一个预测函数序列h_1，⋯ ⋯h_n ，
最终的预测函数H对分类问题采用投票方式，对回归问题采用简单平均方法对新示例进行判别

以下为回归
'''
import random
def bagging_regression(point,numIt=10):
    bagClass=[]
    for i in range(numIt):
        sample=random.sample(point,2)
        if (sample[0][0]-sample[1][0])!=0:
            k=(sample[0][1]-sample[1][1])*1.0/(sample[0][0]-sample[1][0])
            b=sample[0][1]-k*sample[0][0]
            bagClass.append([k,b])
    return bagClass



