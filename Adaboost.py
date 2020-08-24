import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('2.06_glass0.csv',header = None)
d = data.values
indx = np.where(d[:,-1] == 0)
d[:,-1][indx]= -1

# m = np.array([[ 1. ,  2.1],
#         [ 2. ,  1.1],
#         [ 1.3,  1. ],
#         [ 1. ,  1. ],
#         [ 2. ,  1. ]])
# Labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
x_train,x_test, y_train, y_test = train_test_split(d[:,0:-2],d[:,-1],test_size=0.3, random_state=0)


#单层决策树作为弱分类器
def StumpClassify(threshold, sample_weights, features, label, inequ):
    error_sum = 0
    pred = np.zeros(label.shape[0])
    if inequ == 'gt':
        for i in range(label.shape[0]):
            if features[i] > threshold:
                pred[i] = 1
            else:
                pred[i] = -1
            if pred[i] != label[i]:
                    error_sum = error_sum + sample_weights[i]
    else:
        for i in range(label.shape[0]):
            if features[i] < threshold:
                pred[i] = 1
            else:
                pred[i] = -1
            if pred[i] != label[i]:
                    error_sum = error_sum + sample_weights[i]
                    
    return error_sum, pred

def WeakClassify(threshold, inequ, features):
    pred = np.zeros(features.shape[0])
    if inequ == 'gt':
        for i in range(features.shape[0]):
            if features[i] > threshold:
                pred[i] = 1
            else:
                pred[i] = -1

    else:
        for i in range(features.shape[0]):
            if features[i] < threshold:
                pred[i] = 1
            else:
                pred[i] = -1
              
    return pred
            
        
def Decesion_Stump(data,label,sample_weights):
#     遍历每个特征
    minError = math.inf
    for i in range(data.shape[1]):
        step = 12
        step_size = (data[:,i].max() - data[:,i].min())/step
#         遍历每个阈值
        for j in range(step -1):
            T = data[:,i].min() + (j+1) * step_size
            for Inequ in ['gt','lt']:
                er, pred = StumpClassify(T, sample_weights, data[:,i], label,Inequ)
                if er < minError:
                    minError = er
                    recInequ = Inequ
                    recThres = T
                    recFea = i
                    yPred = pred
                    
    return recThres, recInequ, recFea, minError, yPred

def adaBoost(Data, Label):
    Sample_Weights = 1/Data.shape[0] * np.ones(Data.shape[0])
    M = 13  #迭代M步，有M个弱分类器
    Alpha= np.zeros(M)
    Inequ_Set = []
    Thres_Set = []
    featureIndx = []
    for m in range(M):
        T, I, F, error, prediction = Decesion_Stump(Data,Label, Sample_Weights)
        Inequ_Set.append(I)
        Thres_Set.append(T)
        featureIndx.append(F)
        if error == 0:
            Alpha[m] = 10  #设置较大权重，防止error为0，一般不会有这种情况
        else:
            Alpha[m] = 1/2*math.log((1 - error)/error)
            
        Z = np.dot(Sample_Weights,np.exp(-1*Alpha[m] * Label * prediction))
        Sample_Weights = (Sample_Weights * np.exp(-1*Alpha[m] * Label * prediction))/Z
        
    #Synth Classifier:
        
    return Alpha, Inequ_Set, Thres_Set, featureIndx
    

#Classify
alpha, Inequ, T, Fea = adaBoost(x_train, y_train)
# alpha, Inequ, T, Fea = adaBoost(m, Labels)
Pred = np.zeros(y_test.shape[0])
# Pred = np.zeros(m.shape[0])
for n in range(alpha.shape[0]):
    Pred = Pred + alpha[n] * WeakClassify(T[n], Inequ[n], x_test[:,Fea[n]])
    # Pred = Pred + alpha[n] * WeakClassify(T[n], Inequ[n], m[:, Fea[n]])

Pred = np.sign(Pred)

acc = accuracy_score(y_test, Pred)
# acc = accuracy_score(Labels, Pred)
print(acc)