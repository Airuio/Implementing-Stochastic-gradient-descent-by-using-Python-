# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:01:05 2018

@author: a
"""
import numpy as np
from numpy.random import seed

class AdalineSGD(object):
    def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)
            
    def fit(self,X,y):
        self._initialize_weights(X.shape[1])    #初始化权重并标记已经初始化
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:                    #每轮开始时重新给样本排序
                X,y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X,y):         
                cost.append(self._update_weights(xi, target))  #更新权重值
            agv_cost = sum(cost)/len(y)
            self.cost_.append(agv_cost)   #记录本轮的损失函数均值
        return self
    
    def partial_fit(self,X,y):       #实现在线学习，实时更新权重时调用此函数
        if not self.w_initialized:   #检查是否存在权重w，不存在则初始化。
            self._initialize_weights(X.shape[1])
        if y.reval().shape[0] > 1:
            for xi, target in zip(X,y):     #调用新数据进行更新
                self._updata_weights(xi,target)
        else:
            self._update_weights(x,y)
        return self
        
    def _initialize_weights(self,m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
        
    def _shuffle(self,X,y):
        r = np.random.permutation(len(y))
        return X[r],y[r]
    
    def _update_weights(self,xi,target):
        output = self.net_input(xi)    #计算预测值
        errors = (target - output)   #统计误差
        self.w_[1:] += self.eta*xi.dot(errors)
        self.w_[0] += self.eta * errors
        cost = (errors**2).sum()/2.0  #损失函数
        return cost
    
    def net_input(self,X):    #计算预测值
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):    #激活函数~~此处等于预测值
        return self.net_input(X) 

    def predict(self, X):    #返回预测输出
        return np.where(self.activation(X) >= 0.0, 1, -1)    
    
    
    
    
    
    
    
    