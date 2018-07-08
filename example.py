# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 12:25:33 2018

@author: a
"""
from AdalineSGD import AdalineSGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_excel(io = 'lris.xlsx',header = None)    #读取数据为Dataframe结构，没有表头行
y = df.iloc[0:100,4].values         #取前100列数据，4列为标识
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100,[0,2]].values  #iloc为选取表格区域，此处取二维特征进行分类,values为返回不含索引的表
X_std = np.copy(X)                #将样本特征归一化、标准化
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineSGD(eta = 0.01, n_iter = 15,random_state = None)
ada.fit(X_std,y)
def plot_decision_region(X,y,classifier,resolution = 0.02):
    markers = ('s','x','o','~','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #画出界面
    x1_min, x1max = X[:,0].min() - 1, X[:,0].max() + 1   
    x2_min, x2max = X[:,1].min() - 1, X[:,1].max() + 1  
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1max,resolution),  
                          np.arange(x2_min,x2max,resolution))   #生成均匀网格点，
    '''meshgrid的作用是根据传入的两个一维数组参数生成两个数组元素的列表。如果第一个参数是xarray，
    维度是xdimesion，第二个参数是yarray，维度是ydimesion。那么生成的第一个二维数组是以xarray为行，
    ydimesion行的向量；而第二个二维数组是以yarray的转置为列，xdimesion列的向量。'''

    Z = classifier.predict(X = np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    #在全图上每一个点（间隔0.2）计算预测值，并返回1或-1
    
    plt.contourf(xx1,xx2,Z,alpha = 0.5,cmap = cmap) #画出等高线并填充颜色
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #画上分类后的样本
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],alpha=0.8,
                    c=cmap(idx),marker=markers[idx],label=cl)
 
   
plot_decision_region(X_std, y, classifier = ada)    #展示分类结果
plt.xlabel('sepal lenth [nondimensional]')
plt.ylabel('petal lenth [nondimensional]')    
plt.legend(loc = 2)
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker = 'o')   #展示损失函数误差收敛过程
plt.xlabel('Epoches')
plt.ylabel('Average cost_')
plt.show()