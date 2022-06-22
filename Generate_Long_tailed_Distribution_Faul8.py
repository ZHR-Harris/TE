# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:09:06 2021

@author: ppyt
"""


from sklearn import preprocessing
import numpy as np
def get_data():
    train_dataset = ['./d01.txt','./d02.txt','./d04.txt','./d06.txt','./d07.txt']  
    test_dataset = ['./d01_te.txt','./d02_te.txt','./d04_te.txt','./d06_te.txt','./d07_te.txt','./d08_te.txt']   
    unknown_dataset = ['./d01.txt','./d02.txt','./d04.txt','./d06.txt','./d07.txt','./d08.txt']

    positive_data = np.loadtxt('./d00.txt')
    positive_data = positive_data.T
    negative_data = np.empty(shape=[0, 52])
    
    for i in range(5):
        tmp = np.loadtxt(train_dataset[i])
        tmp = tmp[range(0,480,32),:]
        negative_data = np.concatenate((negative_data, tmp))
    
    unknown_data = np.empty(shape=[0, 52])
    for i in range(6):
        tmp = np.loadtxt(unknown_dataset[i])
        unknown_data = np.concatenate((unknown_data, tmp))
    unknown_data = np.concatenate((positive_data, unknown_data)) 
 
        
    data = np.concatenate((positive_data, negative_data))
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)
    
    X_train = []
    X_train.append(data)
    y_train = []
    for index in range(500):
        y_train.append(0)
    for i in range(5):
        for index in range(15):
             y_train.append(i+1)

    
    
    unknown_data = scaler.transform(unknown_data)
    X_unknown = []
    X_unknown.append(unknown_data)   
    y_unknown = [] 
    for index in range(500):
        y_unknown.append(0)
    for i in range(6):
        for index in range(480):
             y_unknown.append(i+1)
    
    
    X_train = np.array(X_train)

    y_train = np.array(y_train)
    X_unknown = np.array(X_unknown)
    y_unknown = np.array(y_unknown)
    
    
    positive_data = np.loadtxt('./d00_te.txt')

    negative_data = np.empty(shape=[0, 52])
    for i in range(6):
        tmp = np.loadtxt(test_dataset[i])
        negative_data = np.concatenate((negative_data, tmp[160:,:]))
    
    data = np.concatenate((positive_data, negative_data))
    data = scaler.transform(data)
    X_test = []
   
    X_test.append(data)
    y_test = []
    
    for index in range(960):
        y_test.append(0)
    for i in range(6):
        for index in range(960+i*800,960+(i+1)*800):
            y_test.append(i+1)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train = np.reshape(X_train,(-1,52))
    X_unknown = np.reshape(X_unknown,(-1,52))
    X_test = np.reshape(X_test,(-1,52))
    X_train_normal = X_train[y_train==0]
    y_train_normal = y_train[y_train==0]
    X_train_abnormal = X_train_normal + 0.7*np.random.randn(500,52)
#    X_train_abnormal = np.load('X_ood_Long_tail.npy')
    y_train_abnormal = (np.ones(500)*6).astype(int)
    X_train_abnormal = np.concatenate((X_train,X_train_abnormal))
    y_train_abnormal = np.concatenate((y_train,y_train_abnormal))
       

    np.save('X_train_multiclass_Long_tail.npy', X_train)
    np.save('y_train_multiclass_Long_tail.npy', y_train)
    np.save('X_test_multiclass_Long_tail.npy', X_test)
    np.save('y_test_multiclass_Long_tail.npy', y_test)
    np.save('X_train_abnormal_Long_tail.npy', X_train_abnormal)
    np.save('y_train_abnormal_Long_tail.npy', y_train_abnormal)
    
    return [X_train, y_train, X_test, y_test]

if __name__ == '__main__':
    get_data()
