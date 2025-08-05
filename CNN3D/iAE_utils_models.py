# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:22:57 2022

@author: nikic
"""

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import mat73
import numpy.random as rnd
import numpy.linalg as lin
import os
plt.rcParams['figure.dpi'] = 200
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from sklearn.decomposition import PCA as PCA
pca=PCA(n_components=2)
from statsmodels.stats.multitest import fdrcorrection as fdr
import statsmodels.api as sm
import numpy.random as rnd
import scipy as scipy
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### UTILS SECTION


# get the data 
def get_data(filename,num_classes=7):
    data_dict = mat73.loadmat(filename)
    data_imagined = data_dict.get('condn_data')
    condn_data_imagined = np.zeros((0,data_imagined[0].shape[1]))
    Y = np.zeros(0)
    for i in np.arange(num_classes):
        tmp = np.array(data_imagined[i])
        condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
        idx = i*np.ones((tmp.shape[0],1))[:,0]
        Y = np.concatenate((Y,idx),axis=0)

    Y_mult = np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        tmp = round(Y[i])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return condn_data_imagined, Y

# get the data 
def get_data_hand_B3(filename,num_classes=7):
    data_dict = mat73.loadmat(filename)
    data_imagined = data_dict.get('condn_data_bins')
    condn_data_imagined = np.zeros((0,data_imagined[0].shape[1]))
    Y = np.zeros(0)
    for i in np.arange(num_classes):
        tmp = np.array(data_imagined[i])
        condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
        idx = i*np.ones((tmp.shape[0],1))[:,0]
        Y = np.concatenate((Y,idx),axis=0)

    Y_mult = np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        tmp = round(Y[i])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return condn_data_imagined, Y

# get the data 
def get_data_B2(filename):
    data_dict = mat73.loadmat(filename)
    data_imagined = data_dict.get('condn_data')
    condn_data_imagined = np.zeros((0,96))
    Y = np.zeros(0)
    for i in np.arange(4):
        tmp = np.array(data_imagined[i])
        condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
        idx = i*np.ones((tmp.shape[0],1))[:,0]
        Y = np.concatenate((Y,idx),axis=0)

    Y_mult = np.zeros((Y.shape[0],4))
    for i in range(Y.shape[0]):
        tmp = round(Y[i])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return condn_data_imagined, Y


# convert to one hot vectors
def one_hot_convert(indata):
    n = len(np.unique(indata))
    Y_mult = np.zeros((indata.shape[0],n))
    for i in range(indata.shape[0]):
        tmp = round(indata[i])
        #tmp = round(indata[i][0])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return Y

# median bootstrap
def median_bootstrap(indata,iters):
    out_boot=  np.zeros([iters,indata.shape[1]])  
    for cols in np.arange(indata.shape[1]):
        xx = indata[:,cols]
        for i in np.arange(iters):
            idx = rnd.choice(indata.shape[0],indata.shape[0])
            xx_tmp = np.median(xx[idx])
            out_boot[i,cols] = xx_tmp
    out_boot = np.sort(out_boot,axis=0)
    out_boot_std = np.std(out_boot,axis=0)
    return out_boot, out_boot_std

# median bootstrap
def mean_bootstrap(indata,iters):
    out_boot=  np.zeros([iters,indata.shape[1]])  
    for cols in np.arange(indata.shape[1]):
        xx = indata[:,cols]
        for i in np.arange(iters):
            idx = rnd.choice(indata.shape[0],indata.shape[0])
            xx_tmp = np.mean(xx[idx])
            out_boot[i,cols] = xx_tmp
    out_boot = np.sort(out_boot,axis=0)
    out_boot_std = np.std(out_boot,axis=0)
    return out_boot, out_boot_std

# scale each data sample to be within 0 and 1
def scale_zero_one(indata):
    for i in range(indata.shape[0]):
        a = indata[i,:]
        a = (a-a.min())/(a.max()-a.min())
        indata[i,:] = a
    
    return(indata)


def get_distance_means(z,idx,num_classes=7):        
    dist_means=np.array([])
    for i in np.arange(num_classes):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]        
        for j in np.arange(i+1,num_classes):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            d = np.mean(A,axis=0)-np.mean(B,axis=0)
            d = (d @ d.T) ** (0.5)
            dist_means = np.append(dist_means,d)
    return dist_means

def get_distance_means_B2(z,idx):        
    dist_means=np.array([])
    for i in np.arange(4):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]        
        for j in np.arange(i+1,4):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            d = np.mean(A,axis=0)-np.mean(B,axis=0)
            d = (d @ d.T) ** (0.5)
            dist_means = np.append(dist_means,d)
    return dist_means

def get_variances(z,idx,num_classes=7):
    dist_var = np.empty((num_classes,))
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
       # A = stats.zscore(A,axis=0)
        C = np.cov(A,rowvar=False)
        if len(C.shape) > 0:
            C = C + 1e-12*np.identity(C.shape[0])
            A = lin.det(C)
        elif len(C.shape) == 0:
            A = C
        dist_var[i] = A
    return dist_var

def get_variances_B2(z,idx):
    dist_var = np.empty((4,))
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
       # A = stats.zscore(A,axis=0)
        C = np.cov(A,rowvar=False)
        if len(C.shape) > 0:
            C = C + 1e-5*np.identity(C.shape[0])
            A = lin.det(C)
        elif len(C.shape) == 0:
            A = C
        dist_var[i] = A
    return dist_var

def get_variance_overall(z):
    C = np.cov(z,rowvar=False)
    if lin.matrix_rank(C) == C.shape[0]:
        A = lin.det(C)
    else:
        C = C + 1e-12*np.identity(C.shape[0])
        A = lin.det(C)
    return A


# function to get mahalanobis distance
def get_mahab_distance(x,y):
    C1 = np.cov(x,rowvar=False) +  1e-9*np.eye(x.shape[1])
    C2 = np.cov(y,rowvar=False) +  1e-9*np.eye(y.shape[1])
    C = (C1+C2)/2
    m1 = np.mean(x,0)
    m2 = np.mean(y,0)
    D = (m2-m1) @ lin.inv(C) @ np.transpose(m2-m1)     
    return D

# function to get mahalanobis distance
def get_mahab_distance_latent(z,idx,num_classes=7):
    mdist =  np.zeros([num_classes,num_classes])
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,len(np.unique(idx))):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            mdist[i,j] = get_mahab_distance(A, B)    
            mdist[j,i] = mdist[i,j]
    return mdist

# function to get mahalanobis distance
def get_mahab_distance_latent_B2(z,idx):
    mdist =  np.zeros([4,4])
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,len(np.unique(idx))):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            mdist[i,j] = get_mahab_distance(A, B)    
            mdist[j,i] = mdist[i,j]
    return mdist


# get monte carlo estimate of the mahab distance pairwise in data in latent space
def monte_carlo_mahab(data,labels,model,num_samples):
    D=np.zeros((labels.shape[1],labels.shape[1]))
    labels = np.argmax(labels,axis=1)
    data = torch.from_numpy(data).to(device).float()
    z = model.encoder(data)
    for i in np.arange(np.max(labels)+1):
        idxA = (labels==i).nonzero()[0]
        A = z[idxA,:].detach().cpu().numpy()        
        for j in np.arange(i+1,np.max(labels)+1):
            idxB = (labels==j).nonzero()[0]
            B = z[idxB,:].detach().cpu().numpy()      
            D[i,j] = get_mahab_distance(A,B)
            D[j,i] = D[i,j]
    
    return(D)

# get monte carlo estimate of the mahab distance pairwise in data in full space
def monte_carlo_mahab_full(data,labels,num_samples):
    D=np.zeros((labels.shape[1],labels.shape[1]))
    labels = np.argmax(labels,axis=1)    
    z = data
    for i in np.arange(np.max(labels)+1):
        idxA = (labels==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,np.max(labels)+1):
            idxB = (labels==j).nonzero()[0]
            B = z[idxB,:]
            D[i,j] = get_mahab_distance(A,B)
            D[j,i] = D[i,j]
    
    return(D)


# split into training and validation class trial level
def training_test_split_trial(condn_data,Y,prop):
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:,:] , condn_data[test_idx,:,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    # training data     
    tmp_data=np.zeros((0,96))
    tmp_y = np.zeros((0,7))
    for i in np.arange(Xtrain.shape[0]):
        tmp = np.squeeze(Xtrain[i,:,:])
        tmp_data = np.concatenate((tmp_data,tmp.T),axis=0)
        tmp1 = Ytrain[i,:]
        tmp1 = np.tile(tmp1,(tmp.shape[1],1))
        tmp_y = np.concatenate((tmp_y,tmp1),axis=0)
    Xtrain = tmp_data
    Ytrain = tmp_y
    # shuffle samples
    idx  = np.random.permutation(Ytrain.shape[0])
    Ytrain = Ytrain[idx,:]
    Xtrain = Xtrain[idx,:]
    
    # testing data 
    tmp_data=np.zeros((0,96))
    tmp_y = np.zeros((0,7))
    for i in np.arange(Xtest.shape[0]):
        tmp = np.squeeze(Xtest[i,:,:])
        tmp_data = np.concatenate((tmp_data,tmp.T),axis=0)
        tmp1 = Ytest[i,:]
        tmp1 = np.tile(tmp1,(tmp.shape[1],1))
        tmp_y = np.concatenate((tmp_y,tmp1),axis=0)
    Xtest = tmp_data
    Ytest = tmp_y
    # shuffle samples
    idx  = np.random.permutation(Ytest.shape[0])
    Ytest = Ytest[idx,:]
    Xtest = Xtest[idx,:]    
    
    return Xtrain,Xtest,Ytrain,Ytest    



# split into training and validation class trial level for online data 
def training_test_split_trial_online(condn_data,Y,prop):
    length = np.arange(Y.shape[0])
    len_cutoff = round(prop*length[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    
    # training data 
    Xtrain = np.zeros((0,96))
    Ytrain = np.zeros((0,7))
    for i in range(len(train_idx)):
        tmp = condn_data[train_idx[i]]
        Xtrain = np.concatenate((Xtrain,tmp.T),axis=0)
        tmp_idx = round(Y[train_idx[i]]-1)
        tmp_y = np.zeros((1,7))
        tmp_y[:,tmp_idx] = 1
        tmp_y = np.tile(tmp_y,(tmp.shape[1],1))
        Ytrain = np.concatenate((Ytrain,tmp_y),axis=0)
    # shuffle samples
    idx  = np.random.permutation(Ytrain.shape[0])
    Ytrain = Ytrain[idx,:]
    Xtrain = Xtrain[idx,:]
    
    # testing data 
    Xtest = np.zeros((0,96))
    Ytest = np.zeros((0,7))
    for i in range(len(test_idx)):
        tmp = condn_data[test_idx[i]]
        Xtest = np.concatenate((Xtest,tmp.T),axis=0)
        tmp_idx = round(Y[test_idx[i]]-1)
        tmp_y = np.zeros((1,7))
        tmp_y[:,tmp_idx] = 1
        tmp_y = np.tile(tmp_y,(tmp.shape[1],1))
        Ytest = np.concatenate((Ytest,tmp_y),axis=0)
    # shuffle samples
    idx  = np.random.permutation(Ytest.shape[0])
    Ytest = Ytest[idx,:]
    Xtest = Xtest[idx,:]
    
    return Xtrain,Xtest,Ytrain,Ytest    

# split into test training splits for the AE
def training_test_val_split_CNN3DAE(xdata,Y,labels,prop):
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, leftover_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xleftover = xdata[train_idx,:,:,:] , xdata[leftover_idx,:,:,:] 
    Ytrain, Yleftover = Y[train_idx,:,:,:] , Y[leftover_idx,:,:,:]
    labels_train, labels_leftover = labels[train_idx] , labels[leftover_idx] 
    # now split left over data in half
    len2_cutoff = round(Yleftover.shape[0]/2)
    Xval,Xtest = Xleftover[:len2_cutoff,:,:,:], Xleftover[len2_cutoff:,:,:,:]
    Yval,Ytest = Yleftover[:len2_cutoff,:,:,:], Yleftover[len2_cutoff:,:,:,:]    
    labels_val,labels_test = labels_leftover[:len2_cutoff], labels_leftover[len2_cutoff:]    
    return Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val


# split into test training splits for the AE with balanced datasets
# within each day, split the data into 70% training, 15% testing and 15% validation 
def training_test_val_split_CNN3DAE_equal_B1(xdata,Y,labels,prop,labels_days):
    days = np.unique(labels_days)
    Xtrain=np.empty((0,40,8,16))
    Xtest=np.empty((0,40,8,16))
    Xval=np.empty((0,40,8,16))
    Ytrain=np.empty((0,40,8,16))
    Ytest=np.empty((0,40,8,16))
    Yval=np.empty((0,40,8,16))
    labels_train=[]
    labels_test=[]
    labels_val=[]
    labels_test_days=[]
    for i in np.arange(len(days)):
        days_idx = np.where(labels_days==days[i])[0]
        xtmp = xdata[days_idx]
        ytmp = Y[days_idx]
        labels_tmp = np.squeeze(labels[days_idx]).astype(int)
        
        # bin count
        bin_count = np.bincount(labels_tmp)
        bin_min = np.where(bin_count==np.min(bin_count))[0]        
        #start first split, get training
        len_cutoff = (prop*bin_count[bin_min])
        # extract len_cutoff elements randomly from each class
        idx0 = np.where(labels_tmp==0)[0]
        idx1 = np.where(labels_tmp==1)[0]
        idx0_main = np.random.permutation(idx0)
        idx1_main = np.random.permutation(idx1)
        
        idx0_train = idx0_main[:round(len_cutoff[0])]
        idx1_train = idx1_main[:round(len_cutoff[0])]
        idx_train = np.concatenate((idx0_train, idx1_train))
        Xtrain_tmp = xtmp[idx_train,:]
        Ytrain_tmp = ytmp[idx_train,:]
        labels_train_tmp = labels_tmp[idx_train].tolist()
        
        Xtrain = np.concatenate((Xtrain,Xtrain_tmp),axis=0)
        Ytrain = np.concatenate((Ytrain,Ytrain_tmp),axis=0)
        labels_train.append(labels_train_tmp)
        
        
        # get leftover and do remaining splits
        idx0_leftover = idx0_main[round(len_cutoff[0]):]
        idx1_leftover = idx1_main[round(len_cutoff[0]):]
        idx_leftover = np.concatenate((idx0_leftover, idx1_leftover))
        labels_leftover = labels_tmp[idx_leftover]
        # bin count
        bin_count = np.bincount(labels_leftover)
        bin_min = np.where(bin_count==np.min(bin_count))[0]        
        # do the second split for validation: equal size
        len_cutoff1 = round(bin_count[bin_min][0]/2)
        idx0_val = idx0_leftover[:len_cutoff1]
        idx0_test = idx0_leftover[len_cutoff1:]
        idx1_val = idx1_leftover[:len_cutoff1]
        idx1_test = idx1_leftover[len_cutoff1:]
        
        idx_val = np.concatenate((idx0_val, idx1_val))
        Xval_tmp = xtmp[idx_val,:]
        Yval_tmp = ytmp[idx_val,:]
        labels_val_tmp = labels_tmp[idx_val].tolist()
        Xval = np.concatenate((Xval,Xval_tmp),axis=0)
        Yval = np.concatenate((Yval,Yval_tmp),axis=0)
        labels_val.append(labels_val_tmp)
        
        idx_test = np.concatenate((idx0_test, idx1_test))
        Xtest_tmp = xtmp[idx_test,:]
        Ytest_tmp = ytmp[idx_test,:]
        labels_test_tmp = labels_tmp[idx_test].tolist()
        Xtest = np.concatenate((Xtest,Xtest_tmp),axis=0)
        Ytest = np.concatenate((Ytest,Ytest_tmp),axis=0)
        labels_test.append(labels_test_tmp)
        tmp = days[i] * np.ones((len(idx_test),1))
        labels_test_days.append(tmp.tolist())
        
        
        
    
    labels_train = np.concatenate(labels_train)
    labels_val = np.concatenate(labels_val)
    labels_test = np.concatenate(labels_test)
    labels_test_days = np.concatenate(labels_test_days)
    
    return Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val,labels_test_days
    

# split into test training splits for the AE with balanced datasets
# within each day, split the data into 70% training, 15% testing and 15% validation 
def training_test_val_split_CNN3DAE_equal(xdata,Y,labels,prop,labels_days):
    days = np.unique(labels_days)
    w,b,h=xdata.shape[1:]
    Xtrain=np.empty((0,w,b,h))
    Xtest=np.empty((0,w,b,h))
    Xval=np.empty((0,w,b,h))
    Ytrain=np.empty((0,w,b,h))
    Ytest=np.empty((0,w,b,h))
    Yval=np.empty((0,w,b,h))
    labels_train=[]
    labels_test=[]
    labels_val=[]
    labels_test_days=[]
    for i in np.arange(len(days)):
        days_idx = np.where(labels_days==days[i])[0]
        xtmp = xdata[days_idx]
        ytmp = Y[days_idx]
        labels_tmp = np.squeeze(labels[days_idx]).astype(int)
        
        # bin count
        bin_count = np.bincount(labels_tmp)
        bin_min = np.where(bin_count==np.min(bin_count))[0]        
        #start first split, get training
        len_cutoff = (prop*bin_count[bin_min])
        # extract len_cutoff elements randomly from each class
        idx0 = np.where(labels_tmp==0)[0]
        idx1 = np.where(labels_tmp==1)[0]
        idx0_main = np.random.permutation(idx0)
        idx1_main = np.random.permutation(idx1)
        
        idx0_train = idx0_main[:round(len_cutoff[0])]
        idx1_train = idx1_main[:round(len_cutoff[0])]
        idx_train = np.concatenate((idx0_train, idx1_train))
        Xtrain_tmp = xtmp[idx_train,:]
        Ytrain_tmp = ytmp[idx_train,:]
        labels_train_tmp = labels_tmp[idx_train].tolist()
        
        Xtrain = np.concatenate((Xtrain,Xtrain_tmp),axis=0)
        Ytrain = np.concatenate((Ytrain,Ytrain_tmp),axis=0)
        labels_train.append(labels_train_tmp)
        
        
        # get leftover and do remaining splits
        idx0_leftover = idx0_main[round(len_cutoff[0]):]
        idx1_leftover = idx1_main[round(len_cutoff[0]):]
        idx_leftover = np.concatenate((idx0_leftover, idx1_leftover))
        labels_leftover = labels_tmp[idx_leftover]
        # bin count
        bin_count = np.bincount(labels_leftover)
        bin_min = np.where(bin_count==np.min(bin_count))[0]        
        # do the second split for validation: equal size
        len_cutoff1 = round(bin_count[bin_min][0]/2)
        idx0_val = idx0_leftover[:len_cutoff1]
        idx0_test = idx0_leftover[len_cutoff1:]
        idx1_val = idx1_leftover[:len_cutoff1]
        idx1_test = idx1_leftover[len_cutoff1:]
        
        idx_val = np.concatenate((idx0_val, idx1_val))
        Xval_tmp = xtmp[idx_val,:]
        Yval_tmp = ytmp[idx_val,:]
        labels_val_tmp = labels_tmp[idx_val].tolist()
        Xval = np.concatenate((Xval,Xval_tmp),axis=0)
        Yval = np.concatenate((Yval,Yval_tmp),axis=0)
        labels_val.append(labels_val_tmp)
        
        idx_test = np.concatenate((idx0_test, idx1_test))
        Xtest_tmp = xtmp[idx_test,:]
        Ytest_tmp = ytmp[idx_test,:]
        labels_test_tmp = labels_tmp[idx_test].tolist()
        Xtest = np.concatenate((Xtest,Xtest_tmp),axis=0)
        Ytest = np.concatenate((Ytest,Ytest_tmp),axis=0)
        labels_test.append(labels_test_tmp)
        tmp = days[i] * np.ones((len(idx_test),1))
        labels_test_days.append(tmp.tolist())
        
        
        
    
    labels_train = np.concatenate(labels_train)
    labels_val = np.concatenate(labels_val)
    labels_test = np.concatenate(labels_test)
    labels_test_days = np.concatenate(labels_test_days)
    
    return Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val,labels_test_days
    
# split into training and validation class 
def training_test_split(condn_data,Y,prop):
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:] , condn_data[test_idx,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    return Xtrain,Xtest,Ytrain,Ytest

# split into training, testing and validation class 
def training_test_split_val(condn_data,Y,prop):
    # prop training, (1-prop)/2 each for val and testing
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, leftover_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xleftover = condn_data[train_idx,:] , condn_data[leftover_idx,:] 
    Ytrain, Yleftover = Y[train_idx,:] , Y[leftover_idx,:]
    # now split left over data in half
    len2_cutoff = round(Yleftover.shape[0]/2)
    Xval,Xtest = Xleftover[:len2_cutoff,:], Xleftover[len2_cutoff:,:]
    Yval,Ytest = Yleftover[:len2_cutoff,:], Yleftover[len2_cutoff:,:]    
    return Xtrain,Xtest,Xval,Ytrain,Ytest,Yval

####### MODELS SECTION
# function to convert one-hot representation back to class numbers
def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata

def convert_to_ClassNumbers_sigmoid(indata):
    with torch.no_grad():
        outdata = torch.sigmoid(indata.float()).cpu()
        outdata[np.where(outdata>0.5)]=1
        outdata[np.where(outdata<=0.5)]=0
    
    return outdata

def convert_to_ClassNumbers_sigmoid_list(indata):
    #outdata = torch.sigmoid(indata.float()).cpu()
    outdata=indata
    outdata[np.where(outdata>0.5)]=1
    outdata[np.where(outdata<=0.5)]=0
    
    return outdata


# create a autoencoder with a classifier layer for separation in latent space
class encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder,self).__init__()
        self.hidden_size2 = round(hidden_size/3)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Tanh()
        self.dropout =  nn.Dropout(p=0.3)
        #self.lnorm1 = nn.LayerNorm(latent_dims,elementwise_affine=False)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)        
        #y=self.gelu(x) 
        #x=self.lnorm1(x)
        #x=self.tanh(x)
        return x
    
class latent_classifier(nn.Module):
    def __init__(self,latent_dims,num_classes):
        super(latent_classifier,self).__init__()
        self.linear1 = nn.Linear(latent_dims,num_classes)
        self.weights = torch.randn(latent_dims,num_classes).to(device)        
    
    def forward(self,x):
        x=self.linear1(x)        
        #x=torch.matmul(x,self.weights)
        return x

class decoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder,self).__init__()
        self.hidden_size2 = round(hidden_size/3)
        self.linear1 = nn.Linear(latent_dims,self.hidden_size2)
        self.linear2 = nn.Linear(self.hidden_size2,hidden_size)
        self.linear3 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
        
        
    def forward(self,x):        
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)        
        return x

class mlp_classifier(nn.Module):
    def __init__(self,input_size,num_nodes,num_classes):
        super(mlp_classifier,self).__init__()
        self.linear1 = nn.Linear(input_size,num_nodes)
        self.linear2 = nn.Linear(num_nodes,num_nodes)
        self.linear3 = nn.Linear(num_nodes,num_nodes)        
        self.linear4 = nn.Linear(num_nodes,num_classes)        
        self.elu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
    
    def forward(self,x):
        x=self.linear1(x)
        x=self.elu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.elu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        x=self.elu(x)
        x=self.dropout(x)
        x=self.linear4(x)
        return x
        
class mlp_classifier_1layer(nn.Module):
    def __init__(self,input_size,num_nodes,num_classes):
        super(mlp_classifier_1layer,self).__init__()
        self.linear1 = nn.Linear(input_size,num_classes)
        
    
    def forward(self,x):
        x=self.linear1(x)
        return x     
    
class mlp_classifier_2Layer(nn.Module):
    def __init__(self,input_size,num_nodes,num_classes):
        super(mlp_classifier_2Layer,self).__init__()
        self.linear1 = nn.Linear(input_size,num_nodes)
        self.linear2 = nn.Linear(num_nodes,num_nodes)
        self.linear3 = nn.Linear(num_nodes,num_classes)
        self.gelu=nn.GELU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self,x):
        x = self.linear1(x)        
        x = self.gelu(x) 
        x = self.dropout(x) # apply the dropout after the activation
        x = self.linear2(x)                
        x = self.gelu(x) 
        x = self.dropout(x) # apply the dropout after the activation
        x = self.linear3(x)
        return x
        

    
# create a autoencoder for b3 with a classifier layer for separation in latent space
class encoder_b3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder_b3,self).__init__()
        self.hidden_size2 = round(hidden_size/2)
        self.hidden_size3 = round(hidden_size/4)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,self.hidden_size3)        
        self.linear4 = nn.Linear(self.hidden_size3,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Tanh()
        self.dropout =  nn.Dropout(p=0.3)
        #self.lnorm1 = nn.LayerNorm(latent_dims,elementwise_affine=False)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear4(x)
        #x=self.lnorm1(x)
        #x=self.tanh(x)
        return x




# class latent_classifier(nn.Module):
#     def __init__(self,latent_dims,num_classes):
#         super(latent_classifier,self).__init__()
#         self.linear1 = nn.Linear(latent_dims,latent_dims*3)
#         self.linear2 = nn.Linear(latent_dims*3,latent_dims*4)
#         self.linear3 = nn.Linear(latent_dims*4,num_classes)
#         #self.weights = torch.randn(latent_dims,num_classes).to(device)        
#         self.gelu =  nn.GELU()
    
#     def forward(self,x):
#         x=self.linear1(x)        
#         x=self.gelu(x)
#         x=self.linear2(x)        
#         x=self.gelu(x)
#         x=self.linear3(x)                
#         #x=torch.matmul(x,self.weights)
#         return x
    
class recon_classifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super(recon_classifier,self).__init__()
        self.linear1 = nn.Linear(input_size,num_classes)
        self.weights = torch.randn(input_size,num_classes)
        self.dropout =  nn.Dropout(p=0.3)
    
    def forward(self,x):
        x=self.linear1(x)
        return x



class decoder_b3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder_b3,self).__init__()
        self.hidden_size2 = round(hidden_size/2)
        self.hidden_size3 = round(hidden_size/4)
        self.linear1 = nn.Linear(latent_dims,self.hidden_size3)
        self.linear2 = nn.Linear(self.hidden_size3,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,hidden_size)
        self.linear4 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
        
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear4(x)        
        return x

# combining all into 
class iAutoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y
    
# combining all into 
class Autoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)        
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z   

# combining all into 
class iAutoencoder_B3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder_B3,self).__init__()
        self.encoder = encoder_b3(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder_b3(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y
    
# # combining all into 
# class Autoencoder(nn.Module):
#     def __init__(self,input_size,hidden_size,latent_dims,num_classes):
#         super(iAutoencoder,self).__init__()
#         self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
#         self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
#         #self.recon_classifier = recon_classifier(input_size,num_classes)
    
#     def forward(self,x):
#         z=self.encoder(x)        
#         z=self.decoder(z)
#         #y=self.recon_classifier(z)
#         return z  
    
# CNN 3D AE encoder
class Encoder3D(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 12, kernel_size=ksize, stride=(1, 1, 2)) 
        self.conv2 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 2, 2)) # downsampling h
        self.conv3 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 1, 2))
        self.conv4 = nn.Conv3d(12, 6, kernel_size=ksize, stride=(1, 1, 1))
        #self.fc = nn.Linear(6 * 7 * 9 *25, num_nodes)    # 6 filters, w, h, d    
        self.elu = nn.ELU()
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)
        #self.pool = nn.AvgPool3d(kernel_size=ksize,stride=1)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x = self.conv2(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x=self.conv3(x)
        #x = self.bn1(x)
        x=self.elu(x)
        
        x=self.conv4(x)
        #x = self.bn2(x)
        x=self.elu(x)
        return x

# CNN 3D AE decoder
class Decoder3D(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D, self).__init__()
        #self.fc = nn.Linear(num_nodes, 6 * 7 * 9 *25)
        self.deconv1 = nn.ConvTranspose3d(6, 12, kernel_size=ksize, stride=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 1, 2))        
        self.deconv3 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 2, 2))                                           
        self.deconv4 = nn.ConvTranspose3d(12, 1, kernel_size=ksize, stride=(1, 1, 2))                                           
        self.elu = nn.ELU()        
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)

    def forward(self, x):
        #x = self.fc(x)
        #x = self.elu(x)
        #x = x.view(x.size(0), 6, 7, 9, 25)
        x = self.deconv1(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv2(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv3(x)
        #x = self.bn1(x)
        x = self.elu(x)
        
        x = self.deconv4(x)
        #x = torch.tanh(x) # squish between -1 and 1
        return x
    
    
# lstm model working in latent space of CNN3D AE
class rnn_lstm(nn.Module):
    def __init__(self,num_classes,input_size,lstm_size):
        super(rnn_lstm,self).__init__()
        self.num_classes = num_classes        
        self.input_size = round(input_size)
        self.lstm_size = round(lstm_size)
        
        self.rnn1=nn.LSTM(input_size=input_size,hidden_size=self.lstm_size,
                          num_layers=1,batch_first=True,bidirectional=False)        
        # self.rnn2=nn.LSTM(input_size=round(self.lstm_size*2),
        #                   hidden_size=round(self.lstm_size/2),
        #                   num_layers=1,batch_first=True,bidirectional=False)      
        self.linear0 = nn.Linear(round(self.lstm_size),num_classes)
                
    
    def forward(self,x):        
        # convert to batch, seq, feature
        x = torch.flatten(x,start_dim=1,end_dim=3)
        x = torch.permute(x,(0,2,1))
        output1, (hn1,cn1) = self.rnn1(x) 
        #output2, (hn2,cn2) = self.rnn2(output1) 
        hn1 = torch.squeeze(hn1)        
        out = self.linear0(hn1)        
        return out
    
class Autoencoder3D(nn.Module):
    def __init__(self, ksize,num_classes,input_size,lstm_size):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D(ksize)
        self.decoder = Decoder3D(ksize)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        self.classifier = rnn_lstm(num_classes,input_size,lstm_size)
        
    def forward(self,x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        logits = self.classifier(latent)
        return recon,logits 


class Encoder3D_B1(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D_B1, self).__init__()
        self.conv1 = nn.Conv3d(1, 12, kernel_size=ksize, stride=(1, 1, 2)) 
        self.conv2 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 2, 2)) # downsampling h
        self.conv3 = nn.Conv3d(12, 6, kernel_size=ksize, stride=(1, 1, 2))
        #self.conv4 = nn.Conv3d(12, 6, kernel_size=ksize, stride=(1, 1, 1))
        #self.fc = nn.Linear(6 * 7 * 9 *25, num_nodes)    # 6 filters, w, h, d    
        self.elu = nn.ELU()
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)
        #self.pool = nn.AvgPool3d(kernel_size=ksize,stride=1)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x = self.conv2(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x=self.conv3(x)
        #x = self.bn1(x)
        x=self.elu(x)
        
        #x=self.conv4(x)
        #x = self.bn2(x)
        #x=self.elu(x)
        return x

# CNN 3D AE decoder
class Decoder3D_B1(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D_B1, self).__init__()
        #self.fc = nn.Linear(num_nodes, 6 * 7 * 9 *25)
        self.deconv1 = nn.ConvTranspose3d(6, 12, kernel_size=ksize, stride=(1, 1, 2))
        self.deconv2 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 2, 2),
                                          output_padding=(0,1,0))        
        self.deconv3 = nn.ConvTranspose3d(12, 1, kernel_size=ksize, stride=(1, 1, 2))                                           
        #self.deconv4 = nn.ConvTranspose3d(12, 1, kernel_size=ksize, stride=(1, 1, 2))                                           
        self.elu = nn.ELU()        
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)

    def forward(self, x):
        #x = self.fc(x)
        #x = self.elu(x)
        #x = x.view(x.size(0), 6, 7, 9, 25)
        x = self.deconv1(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv2(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv3(x)
        #x = self.bn1(x)
        #x = self.elu(x)
        
        #x = self.deconv4(x)
        #x = torch.tanh(x) # squish between -1 and 1
        return x

class Autoencoder3D_B1(nn.Module):
    def __init__(self, ksize,num_classes,input_size,lstm_size):
        super(Autoencoder3D_B1, self).__init__()
        self.encoder = Encoder3D_B1(ksize)
        self.decoder = Decoder3D_B1(ksize)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        self.classifier = rnn_lstm(num_classes,input_size,lstm_size)
        
    def forward(self,x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        logits = self.classifier(latent)
        return recon,logits  


#%%
##### COMPLEX AUTOENCODER FUNCTIONS 

def complex_activation(fr,fi,data):
    act = (fr(data.real) - fi(data.imag)) + 1j*(fr(data.imag) + fi(data.real))
    return act

def apply_complex(a,b):
    act = a+1j*b
    return act


class ComplexConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, strd, pad,dil):
        super(ComplexConv3D, self).__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, stride=strd, padding=pad,dilation=dil)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, stride=strd, padding=pad,dilation=dil)

    def forward(self, real, imag):
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return real_out, imag_out

class ComplexConvTranspose3D(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, strd, pad,dil):
        super(ComplexConvTranspose3D, self).__init__()
        self.real_deconv  = nn.ConvTranspose3d(in_channels, out_channels,kernel_size=ksize,
                                               stride=strd, padding=pad,dilation=dil)
        self.imag_deconv  = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=ksize,
                                               stride=strd, padding=pad,dilation=dil)

    def forward(self, real, imag):
        real_out = self.real_deconv(real) - self.imag_deconv(imag)
        imag_out = self.real_deconv(imag) + self.imag_deconv(real)
        return real_out, imag_out


class Encoder3D_Complex(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D_Complex, self).__init__()
        self.conv1 = ComplexConv3D(1, 8, (3,3,3), (1, 1, 1),0,(1,1,2)) 
        self.conv2 = ComplexConv3D(8, 8, (3,3,3), (1, 1, 1),0,(1,1,2)) 
        self.conv3 = ComplexConv3D(8, 8, (3,3,3), (1, 1, 1),0,(1,1,2))  
        self.conv4 = ComplexConv3D(8, 16, (3,5,7), (1, 1, 1),0,(1,2,2))  
        self.conv5 = ComplexConv3D(16, 16, (3,5,7), (1, 1, 1),0,(1,2,2))  
        self.elu = nn.ELU()
        

    def forward(self, a,b):        
        a,b = self.conv1(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv2(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv3(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv4(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv5(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        return a,b


class Decoder3D_Complex(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D_Complex, self).__init__()
        
        self.deconv1 = ComplexConvTranspose3D(16, 16, (3,5,7), (1, 1, 1),(0,0,0),(1,2,2))
        self.deconv2 = ComplexConvTranspose3D(16, 8, (3,5,7), (1, 1, 1),(0,0,0),(1,2,2))
        self.deconv3 = ComplexConvTranspose3D(8, 8, (3,3,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv4 = ComplexConvTranspose3D(8, 8, (3,3,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv5 = ComplexConvTranspose3D(8, 1, (3,3,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.elu = nn.ELU()        
        
    def forward(self, a,b):        
         a,b = self.deconv1(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv2(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv3(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv4(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv5(a,b)                 
         
                
         
         #x = self.elu(x)        
         #x = self.deconv4(x)
         #x = torch.tanh(x) # squish between -1 and 1
         
         return a,b


class rnn_lstm_complex(nn.Module):
    def __init__(self,num_classes,input_size,lstm_size):
        super(rnn_lstm_complex,self).__init__()
        self.num_classes = num_classes        
        self.input_size = round(input_size)
        self.lstm_size = round(lstm_size)
        
        self.rnn1=nn.LSTM(input_size=self.input_size,hidden_size=self.lstm_size,
                          num_layers=1,batch_first=True,bidirectional=False)        
        # self.rnn2=nn.LSTM(input_size=round(self.lstm_size*2),
        #                   hidden_size=round(self.lstm_size/2),
        #                   num_layers=1,batch_first=True,bidirectional=False)      
        self.linear0 = nn.Linear(round(self.lstm_size),num_classes)
                
    
    def forward(self,a,b):        
        # convert to batch, seq, feature
        x = torch.flatten(a,start_dim=1,end_dim=3)
        x = torch.permute(x,(0,2,1))
        y = torch.flatten(b,start_dim=1,end_dim=3)
        y = torch.permute(y,(0,2,1))
        z = torch.concat((x,y),dim=-1)
        output1, (hn1,cn1) = self.rnn1(z) 
        #output2, (hn2,cn2) = self.rnn2(output1) 
        hn1 = torch.squeeze(hn1)        
        out = self.linear0(hn1)        
        return out

class Autoencoder3D_Complex(nn.Module):
    def __init__(self, ksize,num_classes,input_size,lstm_size):
    #def __init__(self, ksize):
        super(Autoencoder3D_Complex, self).__init__()
        self.encoder = Encoder3D_Complex(ksize)
        self.decoder = Decoder3D_Complex(ksize)
        self.classifier = rnn_lstm_complex(num_classes,input_size,lstm_size)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        
        
    def forward(self,a,b):
        latent_a,latent_b = self.encoder(a,b)
        recon_a,recon_b = self.decoder(latent_a,latent_b)
        logits = self.classifier(latent_a,latent_b)
        #return recon,logits  
        return recon_a,recon_b,logits

############ DEEPER LAYERS ####


class Encoder3D_Complex_deep(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D_Complex_deep, self).__init__()
        self.conv1 = ComplexConv3D(1, 12, (2,2,3), (1, 1, 1),0,(1,1,2)) 
        self.conv2 = ComplexConv3D(12, 12, (2,2,3), (1, 1, 1),0,(1,1,2)) 
        self.conv3 = ComplexConv3D(12, 12, (2,2,3), (1, 1, 1),0,(1,1,2))  
        self.conv4 = ComplexConv3D(12, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
        self.conv5 = ComplexConv3D(16, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
        self.conv6 = ComplexConv3D(16, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
        self.conv7 = ComplexConv3D(16, 24, (2,3,4), (1, 1, 1),0,(1,2,3))  
        self.elu = nn.ELU()
        

    def forward(self, a,b):        
        a,b = self.conv1(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv2(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv3(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv4(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv5(a,b)        
        a,b = self.elu(a),self.elu(b)

        a,b = self.conv6(a,b)        
        a,b = self.elu(a),self.elu(b)       

        a,b = self.conv7(a,b)        
        a,b = self.elu(a),self.elu(b)               
        
        return a,b


class Decoder3D_Complex_deep(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D_Complex_deep, self).__init__()
        
        self.deconv1 = ComplexConvTranspose3D(24, 16, (2,3,4), (1, 1, 1),(0,0,0),(1,2,3))
        self.deconv2 = ComplexConvTranspose3D(16, 16, (2,3,3), (1, 1, 1),(0,0,0),(1,2,2))
        self.deconv3 = ComplexConvTranspose3D(16, 16, (2,3,3), (1, 1, 1),(0,0,0),(1,2,2))
        self.deconv4 = ComplexConvTranspose3D(16, 12, (2,3,3), (1, 1, 1),(0,0,0),(1,2,2))
        self.deconv5 = ComplexConvTranspose3D(12, 12, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv6 = ComplexConvTranspose3D(12, 12, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv7 = ComplexConvTranspose3D(12, 1,  (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.elu = nn.ELU()        
        
    def forward(self, a,b):        
         a,b = self.deconv1(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv2(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv3(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv4(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv5(a,b)         
         a,b = self.elu(a),self.elu(b) 
         
         a,b = self.deconv6(a,b)         
         a,b = self.elu(a),self.elu(b) 
         
         a,b = self.deconv7(a,b)         
         
                
         
         #x = self.elu(x)        
         #x = self.deconv4(x)
         #x = torch.tanh(x) # squish between -1 and 1
         
         return a,b


class rnn_lstm_complex_deep(nn.Module):
    def __init__(self,num_classes,input_size,lstm_size):
        super(rnn_lstm_complex_deep,self).__init__()
        self.num_classes = num_classes        
        self.input_size = round(input_size)
        self.lstm_size = round(lstm_size)
        
        self.rnn1=nn.LSTM(input_size=self.input_size,hidden_size=self.lstm_size,
                          num_layers=1,batch_first=True,bidirectional=False)        
        # self.rnn2=nn.LSTM(input_size=round(self.lstm_size*2),
        #                   hidden_size=round(self.lstm_size/2),
        #                   num_layers=1,batch_first=True,bidirectional=False)      
        self.linear0 = nn.Linear(round(self.lstm_size),num_classes)
                
    
    def forward(self,a,b):        
        # convert to batch, seq, feature
        x = torch.flatten(a,start_dim=1,end_dim=3)
        x = torch.permute(x,(0,2,1))
        y = torch.flatten(b,start_dim=1,end_dim=3)
        y = torch.permute(y,(0,2,1))
        z = torch.concat((x,y),dim=-1)
        output1, (hn1,cn1) = self.rnn1(z) 
        #output2, (hn2,cn2) = self.rnn2(output1) 
        hn1 = torch.squeeze(hn1)        
        out = self.linear0(hn1)        
        return out

class Autoencoder3D_Complex_deep(nn.Module):
    def __init__(self, ksize,num_classes,input_size,lstm_size):
    #def __init__(self, ksize):
        super(Autoencoder3D_Complex_deep, self).__init__()
        self.encoder = Encoder3D_Complex_deep(ksize)
        self.decoder = Decoder3D_Complex_deep(ksize)
        self.classifier = rnn_lstm_complex_deep(num_classes,input_size,lstm_size)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        
        
    def forward(self,a,b):
        latent_a,latent_b = self.encoder(a,b)
        recon_a,recon_b = self.decoder(latent_a,latent_b)
        logits = self.classifier(latent_a,latent_b)
        #return recon,logits  
        return recon_a,recon_b,logits



####### end #######

########### SMALLER ROI COMPLEX CNN AE #####

#in_channels, out_channels, ksize, strd, pad,dil
class Encoder3D_Complex_ROI(nn.Module):
    def __init__(self):
        super(Encoder3D_Complex_ROI, self).__init__()
        self.conv1 = ComplexConv3D(1, 8, (2,2,3), (1, 1, 1),0,(1,1,2)) 
        self.conv2 = ComplexConv3D(8, 8, (2,2,3), (1, 1, 1),0,(1,1,2)) 
        self.conv3 = ComplexConv3D(8, 12, (2,2,3), (1, 1, 1),0,(1,1,2))  
        self.conv4 = ComplexConv3D(12, 12, (2,2,3), (1, 1, 1),0,(1,1,2))  
        self.conv5 = ComplexConv3D(12, 16, (2,2,4), (1, 1, 1),0,(1,1,3))  
        self.conv6 = ComplexConv3D(16, 32, (2,2,4), (1, 1, 1),0,(1,1,3))  
        self.elu = nn.ELU()
        

    def forward(self, a,b):        
        a,b = self.conv1(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv2(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv3(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv4(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv5(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv6(a,b)        
        a,b = self.elu(a),self.elu(b)       
        
        return a,b


#in_channels, out_channels, ksize, strd, pad,dil
class Decoder3D_Complex_ROI(nn.Module):
    def __init__(self):
        super(Decoder3D_Complex_ROI, self).__init__()
        
        self.deconv1 = ComplexConvTranspose3D(32, 16, (2,2,4), (1, 1, 1),(0,0,0),(1,1,3))
        self.deconv2 = ComplexConvTranspose3D(16, 12, (2,2,4), (1, 1, 1),(0,0,0),(1,1,3))
        self.deconv3 = ComplexConvTranspose3D(12, 12, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv4 = ComplexConvTranspose3D(12, 8, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv5 = ComplexConvTranspose3D(8, 8, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.deconv6 = ComplexConvTranspose3D(8, 1, (2,2,3), (1, 1, 1),(0,0,0),(1,1,2))
        self.elu = nn.ELU()        
        
    def forward(self, a,b):        
         a,b = self.deconv1(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv2(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv3(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv4(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv5(a,b)                 
         a,b = self.elu(a),self.elu(b)      
         
         a,b = self.deconv6(a,b)            
                
         
         #x = self.elu(x)        
         #x = self.deconv4(x)
         #x = torch.tanh(x) # squish between -1 and 1
         
         return a,b


class Autoencoder3D_Complex_ROI(nn.Module):
    def __init__(self, num_classes,input_size,lstm_size):
    #def __init__(self, ksize):
        super(Autoencoder3D_Complex_ROI, self).__init__()
        self.encoder = Encoder3D_Complex_ROI()
        self.decoder = Decoder3D_Complex_ROI()
        self.classifier = rnn_lstm_complex(num_classes,input_size,lstm_size)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        
        
    def forward(self,a,b):
        latent_a,latent_b = self.encoder(a,b)
        recon_a,recon_b = self.decoder(latent_a,latent_b)
        logits = self.classifier(latent_a,latent_b)
        #return recon,logits  
        return recon_a,recon_b,logits



########### SMALLER ROI COMPLEX CNN AE WITH TIME FIRST #####

#in_channels, out_channels, ksize, strd, pad,dil
class Encoder3D_Complex_ROI_time(nn.Module):
    def __init__(self):
        super(Encoder3D_Complex_ROI_time, self).__init__()
        self.conv1 = ComplexConv3D(1, 8, (3,2,2), (1, 1, 1),0,(2,1,1)) 
        self.conv2 = ComplexConv3D(8, 8, (3,2,2), (1, 1, 1),0,(2,1,1)) 
        self.conv3 = ComplexConv3D(8, 12, (3,2,2), (1, 1, 1),0,(2,1,1))  
        self.conv4 = ComplexConv3D(12, 12, (3,2,3), (1, 1, 1),0,(2,1,2))  
        self.conv5 = ComplexConv3D(12, 16, (4,2,3), (1, 1, 1),0,(3,1,2))  
        self.conv6 = ComplexConv3D(16, 32, (4,2,3), (1, 1, 1),0,(3,1,2))  
        self.elu = nn.ELU()
        

    def forward(self, a,b):        
        a,b = self.conv1(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv2(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv3(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv4(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv5(a,b)        
        a,b = self.elu(a),self.elu(b)        
        
        a,b = self.conv6(a,b)        
        a,b = self.elu(a),self.elu(b)       
        
        return a,b


#in_channels, out_channels, ksize, strd, pad,dil
class Decoder3D_Complex_ROI_time(nn.Module):
    def __init__(self):
        super(Decoder3D_Complex_ROI_time, self).__init__()
        
        self.deconv1 = ComplexConvTranspose3D(32, 16, (4,2,3), (1, 1, 1),(0,0,0),(3,1,2))
        self.deconv2 = ComplexConvTranspose3D(16, 12, (4,2,3), (1, 1, 1),(0,0,0),(3,1,2))
        self.deconv3 = ComplexConvTranspose3D(12, 12, (3,2,3), (1, 1, 1),(0,0,0),(2,1,2))
        self.deconv4 = ComplexConvTranspose3D(12, 8, (3,2,2), (1, 1, 1),(0,0,0),(2,1,1))
        self.deconv5 = ComplexConvTranspose3D(8, 8, (3,2,2), (1, 1, 1),(0,0,0),(2,1,1))
        self.deconv6 = ComplexConvTranspose3D(8, 1, (3,2,2), (1, 1, 1),(0,0,0),(2,1,1))
        self.elu = nn.ELU()        
        
    def forward(self, a,b):        
         a,b = self.deconv1(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv2(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv3(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv4(a,b)        
         a,b = self.elu(a),self.elu(b)        
         
         a,b = self.deconv5(a,b)                 
         a,b = self.elu(a),self.elu(b)      
         
         a,b = self.deconv6(a,b)            
                
         
         #x = self.elu(x)        
         #x = self.deconv4(x)
         #x = torch.tanh(x) # squish between -1 and 1
         
         return a,b



class rnn_lstm_complex_time(nn.Module):
    def __init__(self,num_classes,input_size,lstm_size):
        super(rnn_lstm_complex_time,self).__init__()
        self.num_classes = num_classes        
        self.input_size = round(input_size)
        self.lstm_size = round(lstm_size)
        
        self.rnn1=nn.LSTM(input_size=self.input_size,hidden_size=self.lstm_size,
                          num_layers=1,batch_first=True,bidirectional=False)        
        # self.rnn2=nn.LSTM(input_size=round(self.lstm_size*2),
        #                   hidden_size=round(self.lstm_size/2),
        #                   num_layers=1,batch_first=True,bidirectional=False)      
        self.linear0 = nn.Linear(round(self.lstm_size),num_classes)
                
    
    def forward(self,a,b):        
        # convert to batch, seq, feature
        x = torch.squeeze(a)
        y = torch.squeeze(b)        
        x = torch.permute(x,(0,2,1))        
        y = torch.permute(y,(0,2,1))
        z = torch.concat((x,y),dim=-1)
        output1, (hn1,cn1) = self.rnn1(z) 
        #output2, (hn2,cn2) = self.rnn2(output1) 
        hn1 = torch.squeeze(hn1)        
        out = self.linear0(hn1)        
        return out

class Autoencoder3D_Complex_ROI_time(nn.Module):
    def __init__(self, num_classes,input_size,lstm_size):
    #def __init__(self, ksize):
        super(Autoencoder3D_Complex_ROI_time, self).__init__()
        self.encoder = Encoder3D_Complex_ROI_time()
        self.decoder = Decoder3D_Complex_ROI_time()
        self.classifier = rnn_lstm_complex_time(num_classes,input_size,lstm_size)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        
        
    def forward(self,a,b):
        latent_a,latent_b = self.encoder(a,b)
        recon_a,recon_b = self.decoder(latent_a,latent_b)
        logits = self.classifier(latent_a,latent_b)
        #return recon,logits  
        return recon_a,recon_b,logits



############## END ###########


def compute_RF(dilation, stride,kernel_size):
    rf=[]
    rf.append(1)
    kernel_size = [x -1 for x in kernel_size]
    #kernel_size = np.array(kernel_size)-np.ones((1,5))
    #kernel_size = kernel_size[0]
    for i in np.arange(len(dilation)):
        eff_k = np.dot(dilation[i] , kernel_size[i] )
        if i==0:
            eff_s=1
        else:
            eff_s = np.prod(stride[:i])
        tmp_r = eff_k*eff_s 
        rf.append(tmp_r)
    
    return rf

# class Encoder3D_Complex(nn.Module):
#     def __init__(self,ksize):
#         super(Encoder3D_Complex, self).__init__()
#         self.conv1 = ComplexConv3D(1, 6, ksize, (1, 1, 1),0) 
#         self.conv2 = ComplexConv3D(6, 6, ksize, (1, 1, 1),0) # downsampling h
#         self.conv3 = ComplexConv3D(6, 4, ksize,(1, 1, 1),0)        
#         self.elu = nn.ELU()
        

#     def forward(self, a,b):        
#         a,b = self.conv1(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         a,b = self.conv2(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         a,b = self.conv3(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         return a,b

# class Encoder3D_Complex(nn.Module):
#     def __init__(self,ksize):
#         super(Encoder3D_Complex, self).__init__()
#         self.conv1 = ComplexConv3D(1, 6, ksize, (1, 1, 2),0) 
#         self.conv2 = ComplexConv3D(6, 6, ksize, (1, 2, 2),0) # downsampling h
#         self.conv3 = ComplexConv3D(6, 4, ksize,(2, 2, 2),0)        
#         self.elu = nn.ELU()
        

#     def forward(self, a,b):        
#         a,b = self.conv1(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         a,b = self.conv2(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         a,b = self.conv3(a,b)        
#         a,b = self.elu(a),self.elu(b)        
        
#         return a,b

#%% model training and validation sections for the complex n/w

# function to validate modes: pass the entire validation data through 
def validation_loss_3DCNNAE_fullVal_complex(model,Xval,Yval,labels_val,batch_val,val_type):
    #crit_classif_val = nn.CrossEntropyLoss(reduction='mean') #if mean, it is over all samples
    crit_classif_val = nn.BCEWithLogitsLoss(reduction='mean')
    crit_recon_val = nn.MSELoss(reduction='mean') # if mean, it is over all elements     
    model.eval()
    with torch.no_grad():
        
        model.eval()
        # split into real and imag
        Xval_real,Xval_imag  = Xval.real,Xval.imag
        Yval_real,Yval_imag  = Yval.real,Yval.imag
        xreal = torch.from_numpy(Xval_real).to(device).float()
        ximag = torch.from_numpy(Xval_imag).to(device).float()
        yreal = torch.from_numpy(Yval_real).to(device).float()
        yimag = torch.from_numpy(Yval_imag).to(device).float()
        z=torch.from_numpy(labels_val).to(device).float()
        
        out_r,out_i,zpred = model(xreal,ximag) 
        loss1 = crit_recon_val(out_r,yreal)
        loss2 = crit_recon_val(out_i,yimag)
        loss_recon = loss1+loss2
        zpred=zpred.squeeze()
        loss_class = crit_classif_val(zpred,z)    
        #loss1  = loss1/x.shape[0]
        #loss2 = loss2/x.shape[0]
        loss_val = 3*loss_recon.item() + loss_class.item()    #30
        
        #zlabels = convert_to_ClassNumbers(z)        
        zlabels=z
        #zpred_labels = convert_to_ClassNumbers(zpred)     
        zpred_labels = convert_to_ClassNumbers_sigmoid(zpred).to(device)     
        accuracy = torch.sum(zlabels == zpred_labels).item()
        accuracy = accuracy/Xval.shape[0]
        #recon_error = (torch.sum(torch.square(out-y))).item()  
        #recon_error = recon_error/x.shape[0]
        recon_error = loss_recon.item()
        
    
    model.train()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error      


# function to validate model 
def validation_loss_3DCNNAE_complex(model,X_test,Y_test,labels_val,batch_val,alp_factor):    
    
    crit_classif_val = nn.BCEWithLogitsLoss(reduction='mean')
    crit_recon_val = nn.MSELoss(reduction='mean') # if mean, it is over all elements         
    
    total_val_loss=0    
    accuracy=0
    
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    model.eval()
    
    loss_recon_batch=[]
    loss_class_batch=[]
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        z=labels_val[idx[i]:idx[i+1]]        
        
        with torch.no_grad():
            
            Xval_real,Xval_imag  = x.real,x.imag
            Yval_real,Yval_imag  = y.real,y.imag
            xreal = torch.from_numpy(Xval_real).to(device).float()
            ximag = torch.from_numpy(Xval_imag).to(device).float()
            yreal = torch.from_numpy(Yval_real).to(device).float()
            yimag = torch.from_numpy(Yval_imag).to(device).float()
            z=torch.from_numpy(z).to(device).float()
            
            out_r,out_i,zpred = model(xreal,ximag) 
            loss1 = crit_recon_val(out_r,yreal)
            loss2 = crit_recon_val(out_i,yimag)
            loss_recon = loss1+loss2
            zpred=zpred.squeeze()
            loss_class = crit_classif_val(zpred,z)   
            
            # normalize loss in  batch size in prop to overall val size (weighted mean)
            alp = x.shape[0]/X_test.shape[0]
            loss_recon = loss_recon.item() * alp
            loss_class = loss_class.item() * alp
            #loss_val = 2.5*loss_recon.item() + loss_class.item()    #30
            loss_recon_batch.append(loss_recon)
            loss_class_batch.append(loss_class)
            
            #zlabels = convert_to_ClassNumbers(z)        
            zlabels=z
            #zpred_labels = convert_to_ClassNumbers(zpred)     
            zpred_labels = convert_to_ClassNumbers_sigmoid(zpred).to(device)     
            
              
            accuracy += torch.sum(zlabels == zpred_labels).item()
           
                
    
    total_val_loss= alp_factor * sum(loss_recon_batch) + sum(loss_class_batch)
    recon_error = alp_factor * sum(loss_recon_batch)    
    accuracy = accuracy/X_test.shape[0]
   
    model.train()
    del xreal, ximag, yreal, yimag, z, out_r, out_i, zpred, zpred_labels
    torch.cuda.empty_cache()
    
    return total_val_loss,accuracy,recon_error


# TRAINING LOOP
def training_loop_iAE3D_Complex(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                      input_size,num_classes,ksize,lstm_size,alp_factor):
    
   
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    recon_criterion = nn.MSELoss(reduction='mean')
    #classif_criterion = nn.CrossEntropyLoss(reduction='mean')    
    classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')    
    
    opt = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    print('Starting training')
    goat_loss=99999
    counter=0
    recon_loss_epochs=[]
    classif_loss_epochs=[]
    total_loss_epochs=[]
    
    for epoch in range(num_epochs):
      # reset memory
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()  # helps reduce fragmentation 
      model.train()
      #shuffle the data    
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      
      if epoch>round(num_epochs*0.6):
          for g in opt.param_groups:
              g['lr']=5e-4
        
      for batch in range(num_batches):
                    
          
          # get the batch 
          samples = idx_split[batch]
          Xtrain_batch = Xtrain[samples,:]
          Ytrain_batch = Ytrain[samples,:]        
          #labels_batch = labels_train[samples,:]
          labels_batch = labels_train[samples]
          
          
          # split into real and imag parts 
          Xtrain_batch_real,Xtrain_batch_imag = Xtrain_batch.real,Xtrain_batch.imag
          Ytrain_batch_real,Ytrain_batch_imag = Ytrain_batch.real,Ytrain_batch.imag
          
          #load to torch and push to GPU
          Xtrain_batch_real = torch.from_numpy(Xtrain_batch_real).to(device).float()
          Xtrain_batch_imag = torch.from_numpy(Xtrain_batch_imag).to(device).float()
          Ytrain_batch_real = torch.from_numpy(Ytrain_batch_real).to(device).float()
          Ytrain_batch_imag = torch.from_numpy(Ytrain_batch_imag).to(device).float()
          labels_batch = torch.from_numpy(labels_batch).to(device).float()  
          
          # forward pass thru network
          opt.zero_grad() 
          recon_r,recon_i,decodes = model(Xtrain_batch_real,Xtrain_batch_imag)
          #latent_activity = model.encoder(Xtrain_batch)      
          
          # get loss  
          loss1 = recon_criterion(recon_r,Ytrain_batch_real)
          loss2 = recon_criterion(recon_i,Ytrain_batch_imag)
          recon_loss = loss1+loss2
          #recon_loss = (recon_criterion(recon,Ytrain_batch))#/Ytrain_batch.shape[0]
          decodes = decodes.squeeze() # for BCE loss
          classif_loss = (classif_criterion(decodes,labels_batch))#/labels_batch.shape[0]      
          loss = alp_factor*recon_loss + classif_loss
          total_loss = loss.item()
          #print(classif_loss.item())
          
          # compute accuracy
          #ylabels = convert_to_ClassNumbers(labels_batch)        
          ylabels =  labels_batch
          with torch.no_grad():
              ypred_labels = convert_to_ClassNumbers_sigmoid(decodes).to(device)    
              #accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
              accuracy = (torch.sum(ylabels == ypred_labels.squeeze()).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
          

      
      # get validation losses with batching of validation samples
      #val_loss,val_acc,val_recon=validation_loss_3DCNNAE(model,Xval,Yval,labels_val,batch_val,1)  
      
      # store losses from each epoch
      recon_loss_epochs.append(alp_factor*recon_loss.item())
      classif_loss_epochs.append(classif_loss.item())
      total_loss_epochs.append(total_loss)

      # get validation loss passing entire validation data through the network
      #val_loss,val_acc,val_recon=validation_loss_3DCNNAE_fullVal_complex(model,Xval,Yval,labels_val,batch_val,1)
      val_loss,val_acc,val_recon=validation_loss_3DCNNAE_complex(model,Xval,Yval,
                                                              labels_val,batch_val,alp_factor)    
      
      #print(torch.cuda.memory_summary())
      print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      if val_loss<goat_loss:
          goat_loss = val_loss
          goat_acc = val_acc*100      
          counter = 0
          print('Goat loss, saving model')      
          torch.save(model.state_dict(), filename)
      else:
          counter += 1
    
      if counter>=patience:
          print('Early stoppping point reached')
          print('Best val loss and val acc  are')
          print(goat_loss,goat_acc)
          break
    
    model_goat = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size)
    #model_goat = Autoencoder3D_Complex_ROI_time(num_classes,input_size,lstm_size)
    #model_goat = Autoencoder3D_B1(ksize,num_classes,input_size,lstm_size)    
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat, goat_acc,recon_loss_epochs,classif_loss_epochs,total_loss_epochs


def test_model_complex(model,Xtest):
    
    Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
    recon_r=[]
    recon_i=[]
    decodes=[]    
    num_batches = math.ceil(Xtest_real.shape[0]/4096)
    idx = (np.arange(Xtest_real.shape[0]))
    idx_split = np.array_split(idx,num_batches)
    model.eval()
    
    for batch in range(num_batches):
       # get the batch 
       samples = idx_split[batch]
       Xtest_real_batch = torch.from_numpy(Xtest_real[samples,:]).to(device).float()
       Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples,:]).to(device).float()
       with torch.no_grad():
           out_r,out_i,dec = model(Xtest_real_batch,Xtest_imag_batch)
       
       out_r = out_r.cpu().detach().numpy()
       out_i = out_i.cpu().detach().numpy()
       dec = dec.cpu().detach().numpy()
        
       recon_r.append(out_r)
       recon_i.append(out_i)
       decodes.append(dec)
       
    recon_r = np.concatenate(recon_r,axis=0) 
    recon_i = np.concatenate(recon_i,axis=0) 
    decodes = np.concatenate(decodes,axis=0) 
    
    return recon_r,recon_i,decodes 
        
#%% PLOTTING FOR COMPLEX KERNELS


def plot_phasor_kernels_side_by_side(weight_real, weight_imag, out_ch=0, in_ch=0):
    """
    Plots phasor arrows for each time slice side by side from a complex-valued 3D convolution kernel.
    """

    # Extract complex weights for selected channel pair
    # Shape: (T, H, W)
    kernel_real = weight_real[out_ch, in_ch]
    kernel_imag = weight_imag[out_ch, in_ch]
    kernel_complex = kernel_real + 1j * kernel_imag

    H, W,T = kernel_complex.shape

    # Create figure with T subplots (1 row, T columns)
    fig, axes = plt.subplots(1, T, figsize=(4*T, 4))

    if T == 1:
        axes = [axes]  # ensure iterable

    for t in range(T):
        ax = axes[t]
        slice_complex = kernel_complex[:,:,t].to('cpu').detach().numpy()

        # Compute magnitude and phase
        magnitude = np.abs(slice_complex)
        phase = np.angle(slice_complex)

        # Grid for quiver
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        U = magnitude * np.cos(phase)
        V = magnitude * np.sin(phase)

        ax.quiver(X, Y, U, V, magnitude, color='black',
                  angles='xy', scale_units='xy', scale=1)
        ax.set_title(f"Time slice {t}")
        #ax.set_xlim(-0.5, W - 0.5)
        #ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect('equal')
        #ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_phasor_kernels_side_by_side_v2(weight_real, weight_imag, out_ch=0, in_ch=0):
    """
    Plots phasor arrows for each time slice side by side from a complex-valued 3D convolution kernel.
    """

    # Extract complex weights for selected channel pair
    # Shape: (T, H, W)
    kernel_real = weight_real[out_ch, in_ch]
    kernel_imag = weight_imag[out_ch, in_ch]
    kernel_complex = kernel_real + 1j * kernel_imag

    H, W,T = kernel_complex.shape

    # Create figure with T subplots (1 row, T columns)
    fig, axes = plt.subplots(1, T)

    if T == 1:
        axes = [axes]  # ensure iterable

    for t in range(T):
        ax = axes[t]
        slice_complex = kernel_complex[:,:,t].to('cpu').detach().numpy()

        # Compute magnitude and phase
        magnitude = np.abs(slice_complex)
        phase = np.angle(slice_complex)

        # Grid for quiver
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        U = magnitude * np.cos(phase)
        V = magnitude * np.sin(phase)
        
        ax.quiver(X, Y, U, V, angles='xy')

        # ax.quiver(X, Y, U, V, magnitude, color='black',
        #           angles='xy', scale_units='xy', scale=1)
        ax.set_title(f"Time slice {t}",fontsize=4)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect('equal')
        #ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)  # Increase thickness

    plt.tight_layout()
    plt.show()




def plot_phasor_kernels_side_by_side_PCA(weights):
    """
    Plots phasor arrows for each time slice side by side from a complex-valued 3D convolution kernel.
    """

    # Extract complex weights for selected channel pair
        
    kernel_complex = weights

    H, W,T = kernel_complex.shape

    # Create figure with T subplots (1 row, T columns)
    fig, axes = plt.subplots(1, T)

    if T == 1:
        axes = [axes]  # ensure iterable

    for t in range(T):
        ax = axes[t]
        slice_complex = kernel_complex[:,:,t]

        # Compute magnitude and phase
        magnitude = np.abs(slice_complex)
        phase = np.angle(slice_complex)

        # Grid for quiver
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        U = magnitude * np.cos(phase)
        V = magnitude * np.sin(phase)
        
        ax.quiver(X, Y, U, V, angles='xy')

        # ax.quiver(X, Y, U, V, magnitude, color='black',
        #           angles='xy', scale_units='xy', scale=1)
        ax.set_title(f"Time slice {t}",fontsize=4)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect('equal')
        #ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)  # Increase thickness

    plt.tight_layout()
    plt.show()


def plot_phasor_frame(x_real, x_imag, t, ax):
    H, W = x_real.shape[:2]
    
    # Grid positions
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Get vectors
    U = x_real[:, :, t]
    V = x_imag[:, :, t]
    
    # Plot arrows
    ax.clear()
    ax.quiver(X, Y, U, V, angles='xy')
    ax.set_title(f'Phasor Field at Time {t}')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.invert_yaxis()

def plot_phasor_frame_time(x_real, x_imag, t, ax):
    H, W = x_real.shape[1:]
    
    # Grid positions
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Get vectors
    U = x_real[t, :, :]
    V = x_imag[t, :, :]
    
    # Plot arrows
    ax.clear()
    ax.quiver(X, Y, U, V, angles='xy')
    ax.set_title(f'Phasor Field at Time {t}')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    #ax.invert_yaxis()


# data augmentation: constant phase shifts, temporal roll, add gaussian noise
#iterations: number of extra samples to create per training sample (augmentation factor)
def complex_data_augmentation(Xtrain,labels_train,sigma,iterations): 
    
    shpe = np.r_[0, Xtrain.shape[1:]]
    Xtrain_aug = np.empty(shpe,dtype=Xtrain.dtype)
    labels_train_aug = np.empty(0,dtype=labels_train.dtype)
    n,c, h, w,t = Xtrain.shape
    
    for j in np.arange(iterations):
        
        # global phase shifts
        thetas = rnd.uniform(0,2*np.pi,size=n)
        phase_shifts = np.exp(1j*thetas)
        phase_shifts = phase_shifts[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        tmp = Xtrain * phase_shifts
        
        # temporal roll of each trial
        shifts = np.random.randint(0, t, size=n)
        idx = (np.arange(t)[None, :] - shifts[:, None]) % t  # shape: (N, T)
        idx = idx[:, None, None, None, :]  # shape: (N, 1, 1, 1, T)                
        idx = np.tile(idx, (1, c, h, w, 1))  # shape: (N, 1, H, W, T)                
        tmp = np.take_along_axis(tmp, idx, axis=-1)
        
        # add small gaussian noise 
        noise = rnd.randn(n,c,h,w,t) * sigma + 1j*rnd.randn(n,c,h,w,t) * sigma
        tmp = tmp + noise
        
        Xtrain_aug = np.concatenate((Xtrain_aug,tmp),axis=0)
        labels_train_aug = np.concatenate((labels_train_aug, labels_train),axis=0)
    
    
    Xtrain_aug = np.concatenate((Xtrain, Xtrain_aug),axis=0)
    labels_train_aug = np.concatenate((labels_train,labels_train_aug),axis=0)
    
    return Xtrain_aug,labels_train_aug
        
        
    
    

#%% #### model training and validation sections 

# function to validate model 
def validation_loss(model,X_test,Y_test,batch_val,val_type):    
    crit_classif_val = nn.CrossEntropyLoss(reduction='sum') #if mean, it is over all samples
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    accuracy=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        with torch.no_grad():                
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                model.eval()
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
                model.train()
            else:
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
            
            ylabels = convert_to_ClassNumbers(y)        
            ypred_labels = convert_to_ClassNumbers(ypred)     
            accuracy += torch.sum(ylabels == ypred_labels).item()
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error




# function to validate model 
def validation_loss_3DCNNAE(model,X_test,Y_test,labels_val,batch_val,val_type):    
    crit_classif_val = nn.CrossEntropyLoss(reduction='sum') #if mean, it is over all samples
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    accuracy=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        z=labels_val[idx[i]:idx[i+1],:]        
        model.eval()
        with torch.no_grad():                     
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                z=torch.from_numpy(z).to(device).float()
                
                out,zpred = model(x) 
                loss1 = crit_recon_val(out,y)
                loss2 = crit_classif_val(zpred,z)
                loss_val += loss1.item() + loss2.item()
                
            else:
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
            
            zlabels = convert_to_ClassNumbers(z)        
            zpred_labels = convert_to_ClassNumbers(zpred)     
            accuracy += torch.sum(zlabels == zpred_labels).item()
            recon_error += (torch.sum(torch.square(out-y))).item()   
            
    model.train()
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error



# function to validate modes: pass the entire validation data through 
def validation_loss_3DCNNAE_fullVal(model,Xval,Yval,labels_val,batch_val,val_type):
    #crit_classif_val = nn.CrossEntropyLoss(reduction='mean') #if mean, it is over all samples
    crit_classif_val = nn.BCEWithLogitsLoss(reduction='mean')
    crit_recon_val = nn.MSELoss(reduction='mean') # if mean, it is over all elements     
    model.eval()
    with torch.no_grad():
        x=torch.from_numpy(Xval).to(device).float()
        y=torch.from_numpy(Yval).to(device).float()
        z=torch.from_numpy(labels_val).to(device).float()
        out,zpred = model(x) 
        loss1 = crit_recon_val(out,y)
        zpred=zpred.squeeze()
        loss2 = crit_classif_val(zpred,z)    
        #loss1  = loss1/x.shape[0]
        #loss2 = loss2/x.shape[0]
        loss_val = 30*loss1.item() + loss2.item()    #30
        
    #zlabels = convert_to_ClassNumbers(z)        
    zlabels=z
    #zpred_labels = convert_to_ClassNumbers(zpred)     
    zpred_labels = convert_to_ClassNumbers_sigmoid(zpred).to(device)     
    accuracy = torch.sum(zlabels == zpred_labels).item()
    accuracy = accuracy/x.shape[0]
    #recon_error = (torch.sum(torch.square(out-y))).item()  
    #recon_error = recon_error/x.shape[0]
    recon_error = crit_recon_val(out,y).item()
    
    
    model.train()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error      
        

# TRAINING LOOP
def training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes):
    
   
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    recon_criterion = nn.MSELoss(reduction='sum')
    classif_criterion = nn.CrossEntropyLoss(reduction='sum')    
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    print('Starting training')
    goat_loss=99999
    counter=0
    model.train()
    for epoch in range(num_epochs):
      #shuffle the data    
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      
      if epoch>round(num_epochs*0.6):
          for g in opt.param_groups:
              g['lr']=1e-4
        
      for batch in range(num_batches):
          # get the batch 
          samples = idx_split[batch]
          Xtrain_batch = Xtrain[samples,:]
          Ytrain_batch = Ytrain[samples,:]        
          
          #push to gpu
          Xtrain_batch = torch.from_numpy(Xtrain_batch).to(device).float()
          Ytrain_batch = torch.from_numpy(Ytrain_batch).to(device).float()          
          
          # forward pass thru network
          opt.zero_grad() 
          recon,decodes = model(Xtrain_batch)
          latent_activity = model.encoder(Xtrain_batch)      
          
          # get loss      
          recon_loss = (recon_criterion(recon,Xtrain_batch))/Xtrain_batch.shape[0]
          classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
          loss = recon_loss + classif_loss
          total_loss = loss.item()
          #print(classif_loss.item())
          
          # compute accuracy
          ylabels = convert_to_ClassNumbers(Ytrain_batch)        
          ypred_labels = convert_to_ClassNumbers(decodes)     
          accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
      
      # get validation losses
      val_loss,val_acc,val_recon=validation_loss(model,Xtest,Ytest,batch_val,1)    
      #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
      
      
      print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      if val_loss<goat_loss:
          goat_loss = val_loss
          goat_acc = val_acc*100      
          counter = 0
          print('Goat loss, saving model')      
          torch.save(model.state_dict(), filename)
      else:
          counter += 1
    
      if counter>=patience:
          print('Early stoppping point reached')
          print('Best val loss and val acc  are')
          print(goat_loss,goat_acc)
          break
    #model_goat = Autoencoder3D(ksize,num_classes,input_size,lstm_size)
    model_goat = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)  
    #model_goat = iAutoencoder_B3(input_size,hidden_size,latent_dims,num_classes)
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat, goat_acc

# TRAINING LOOP
def training_loop_iAE3D(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                      input_size,num_classes,ksize,lstm_size):
    
   
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    recon_criterion = nn.MSELoss(reduction='mean')
    #classif_criterion = nn.CrossEntropyLoss(reduction='mean')    
    classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')    
    
    opt = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    print('Starting training')
    goat_loss=99999
    counter=0
    
    for epoch in range(num_epochs):
      model.train()
      #shuffle the data    
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      
      if epoch>round(num_epochs*0.6):
          for g in opt.param_groups:
              g['lr']=5e-4
        
      for batch in range(num_batches):
          # get the batch 
          samples = idx_split[batch]
          Xtrain_batch = Xtrain[samples,:]
          Ytrain_batch = Ytrain[samples,:]        
          #labels_batch = labels_train[samples,:]
          labels_batch = labels_train[samples]
          
          #push to gpu
          Xtrain_batch = torch.from_numpy(Xtrain_batch).to(device).float()
          Ytrain_batch = torch.from_numpy(Ytrain_batch).to(device).float()  
          labels_batch = torch.from_numpy(labels_batch).to(device).float()  
          
          
          # forward pass thru network
          opt.zero_grad() 
          recon,decodes = model(Xtrain_batch)
          #latent_activity = model.encoder(Xtrain_batch)      
          
          # get loss      
          recon_loss = (recon_criterion(recon,Ytrain_batch))#/Ytrain_batch.shape[0]
          decodes = decodes.squeeze() # for BCE loss
          classif_loss = (classif_criterion(decodes,labels_batch))#/labels_batch.shape[0]      
          loss = 30*recon_loss + classif_loss#30
          total_loss = loss.item()
          #print(classif_loss.item())
          
          # compute accuracy
          #ylabels = convert_to_ClassNumbers(labels_batch)        
          ylabels =  labels_batch
          ypred_labels = convert_to_ClassNumbers_sigmoid(decodes).to(device)    
          #accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          accuracy = (torch.sum(ylabels == ypred_labels.squeeze()).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
      
      # get validation losses with batching of validation samples
      #val_loss,val_acc,val_recon=validation_loss_3DCNNAE(model,Xval,Yval,labels_val,batch_val,1)  
      
      # get validation loss passing entire validation data through the network
      val_loss,val_acc,val_recon=validation_loss_3DCNNAE_fullVal(model,Xval,Yval,labels_val,batch_val,1)
      #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
      
      
      print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      if val_loss<goat_loss:
          goat_loss = val_loss
          goat_acc = val_acc*100      
          counter = 0
          print('Goat loss, saving model')      
          torch.save(model.state_dict(), filename)
      else:
          counter += 1
    
      if counter>=patience:
          print('Early stoppping point reached')
          print('Best val loss and val acc  are')
          print(goat_loss,goat_acc)
          break
    
    model_goat = Autoencoder3D(ksize,num_classes,input_size,lstm_size)
    #model_goat = Autoencoder3D_B1(ksize,num_classes,input_size,lstm_size)    
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat, goat_acc




# function to validate model for mlp
def validation_loss_mlp(model,X_test,Y_test,batch_val,val_type):    
    crit_classif_val = nn.CrossEntropyLoss(reduction='sum') #if mean, it is over all samples
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    accuracy=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        with torch.no_grad():                
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                model.eval()
                ypred = model(x) 
                #loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                #loss_val += loss1.item() + loss2.item()
                loss_val += loss2.item()
                model.train()
            else:
                ypred = model(x) 
                #loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                #loss_val += loss1.item() + loss2.item()
                loss_val +=  loss2.item()
            
            ylabels = convert_to_ClassNumbers(y)        
            ypred_labels = convert_to_ClassNumbers(ypred)     
            accuracy += torch.sum(ylabels == ypred_labels).item()
            #recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    #recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,accuracy

def training_loop_mlp(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,num_nodes,num_classes):
    
   
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    #recon_criterion = nn.MSELoss(reduction='sum')
    classif_criterion = nn.CrossEntropyLoss(reduction='sum')    
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    print('Starting training')
    goat_loss=99999
    counter=0
    model.train()
    for epoch in range(num_epochs):
      #shuffle the data    
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      
      if epoch>round(num_epochs*0.6):
          for g in opt.param_groups:
              g['lr']=1e-4
        
      for batch in range(num_batches):
          # get the batch 
          samples = idx_split[batch]
          Xtrain_batch = Xtrain[samples,:]
          Ytrain_batch = Ytrain[samples,:]        
          
          #push to gpu
          Xtrain_batch = torch.from_numpy(Xtrain_batch).to(device).float()
          Ytrain_batch = torch.from_numpy(Ytrain_batch).to(device).float()          
          
          # forward pass thru network
          opt.zero_grad() 
          decodes = model(Xtrain_batch)
          
          
          # get loss      
          #recon_loss = (recon_criterion(recon,Xtrain_batch))/Xtrain_batch.shape[0]
          classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
          loss = classif_loss
          total_loss = loss.item()
          #print(classif_loss.item())
          
          # compute accuracy
          ylabels = convert_to_ClassNumbers(Ytrain_batch)        
          ypred_labels = convert_to_ClassNumbers(decodes)     
          accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
      
      # get validation losses
      val_loss,val_acc=validation_loss_mlp(model,Xtest,Ytest,batch_val,1)    
      #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
      
      
      print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      if val_loss<goat_loss:
          goat_loss = val_loss
          goat_acc = val_acc*100      
          counter = 0
          print('Goat loss, saving model')      
          torch.save(model.state_dict(), filename)
      else:
          counter += 1
    
      if counter>=patience:
          print('Early stoppping point reached')
          print('Best val loss and val acc  are')
          print(goat_loss,goat_acc)
          break
    model_goat = mlp_classifier_2Layer(input_size,num_nodes,num_classes)  
    #model_goat = iAutoencoder_B3(input_size,hidden_size,latent_dims,num_classes)
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat, goat_acc



# plotting and returning latent activations
def plot_latent(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    #idx1 = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y#y=Y[idx1]    
    z = data#[idx1,:]
    model.eval()
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()        
    #D = sil(z,y)
    D=0
    # scale between 0 and 1
    #z=  (z-np.min(z))/(np.max(z)-np.min(z))    
    fig=plt.figure()
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10',alpha=1)        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return D,z,y,fig

# plotting and returning latent activations and give acuracy
def plot_latent_acc(model, data, Y,dim):
    num_samples = data.shape[0]
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    
    model.eval()
    
    z = model.encoder(data)
    recon,decodes = model(data)    
    ylabels = convert_to_ClassNumbers(Y)        
    ypred_labels = convert_to_ClassNumbers(decodes)       
    accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
    
    y=ylabels  
    ypred = ypred_labels
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()        
    ypred = ypred.to('cpu').detach().numpy()        
    D = sil(z,y)
        
    fig=plt.figure()
    if dim>2:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return D,z,y,fig,accuracy,ypred


# plotting and returning latent activations and give acuracy
def plot_latent_select(model, data, Y,dim,ch):
    num_samples = data.shape[0]
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    
    model.eval()
    
    z = model.encoder(data)
    recon,decodes = model(data)    
    ylabels = convert_to_ClassNumbers(Y)        
    ypred_labels = convert_to_ClassNumbers(decodes)    
        
    y=ylabels    
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()   
    idx=np.array([],dtype=int)
    for i in np.arange(len(ch)):
        idx = np.append(idx,np.where(y==ch[i])[0])
               
    z = z[idx,:]
    y = y[idx]    
        
    fig=plt.figure()
    if dim>2:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return fig


def return_recon(model,data,Y,num_classes=7):
    data = torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y).to('cpu').detach().numpy()
    model.eval()
    hg_recon = []
    beta_recon=[]
    delta_recon=[]
    for query in np.arange(num_classes):
        idx = np.where(Y==query)[0]
        data_tmp = data[idx,:]        
        
        with torch.no_grad():
            recon_data,class_outputs = model(data_tmp)        
                
        recon_data = recon_data.to('cpu').detach().numpy()    
        # hg
        idx = np.arange(2,96,3)
        hg_recon_tmp = recon_data[:,idx]  
        hg_recon.append(hg_recon_tmp)
        # delta
        idx = np.arange(0,96,3)
        delta_recon_tmp = recon_data[:,idx]        
        delta_recon.append(delta_recon_tmp)
        #beta
        idx = np.arange(1,96,3)
        beta_recon_tmp = recon_data[:,idx]  
        beta_recon.append(beta_recon_tmp)
    
    model.train()
    return delta_recon,beta_recon,hg_recon

def return_recon_B2(model,data,Y):
    data = torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y).to('cpu').detach().numpy()
    model.eval()
    hg_recon = []
    beta_recon=[]
    delta_recon=[]
    for query in np.arange(4):
        idx = np.where(Y==query)[0]
        data_tmp = data[idx,:]        
        
        with torch.no_grad():
            recon_data,class_outputs = model(data_tmp)        
                
        recon_data = recon_data.to('cpu').detach().numpy()    
        # hg
        idx = np.arange(2,96,3)
        hg_recon_tmp = recon_data[:,idx]  
        hg_recon.append(hg_recon_tmp)
        # delta
        idx = np.arange(0,96,3)
        delta_recon_tmp = recon_data[:,idx]        
        delta_recon.append(delta_recon_tmp)
        #beta
        idx = np.arange(1,96,3)
        beta_recon_tmp = recon_data[:,idx]  
        beta_recon.append(beta_recon_tmp)
    
    model.train()
    return delta_recon,beta_recon,hg_recon

def return_recon_B3(model,data,Y,num_classes=7):
    data = torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y).to('cpu').detach().numpy()
    model.eval()
    hg_recon = []
    beta_recon=[]
    delta_recon=[]
    for query in np.arange(num_classes):
        idx = np.where(Y==query)[0]
        data_tmp = data[idx,:]        
        
        with torch.no_grad():
            recon_data,class_outputs = model(data_tmp)        
                
        recon_data = recon_data.to('cpu').detach().numpy()    
        # first 253 are delta, next 253 are beta and last 253 are hG
        
        # delta
        idx = np.arange(0,253)
        delta_recon_tmp = recon_data[:,idx]        
        delta_recon.append(delta_recon_tmp)
        # beta
        idx  = np.arange(253,253*2)
        beta_recon_tmp = recon_data[:,idx]  
        beta_recon.append(beta_recon_tmp)
        # hG
        idx=np.arange(253*2,253*3)
        hg_recon_tmp = recon_data[:,idx]  
        hg_recon.append(hg_recon_tmp)        
    
    model.train()
    return delta_recon,beta_recon,hg_recon
    
def get_recon_channel_variances(recon_data):
    l = len(recon_data)
    variances = np.array([])
    for query in np.arange(l):
        tmp = recon_data[query]
        a = np.std(tmp,axis=0)
        variances = np.append(variances,a)
    
    return variances

def get_spatial_correlation(data1,data2,data3):
    corr_coef = []
    for query in np.arange(len(data1)):
        tmp1 = np.mean(data1[query],axis=0)
        tmp2 = np.mean(data2[query],axis=0)
        tmp3 = np.mean(data3[query],axis=0)
        a = np.corrcoef(tmp1,tmp2)[0,1]
        b = np.corrcoef(tmp1,tmp3)[0,1]
        c = np.corrcoef(tmp2,tmp3)[0,1]
        corr_coef.append([a,b,c])
    corr_coef = np.array(corr_coef).flatten()
    return corr_coef
    

def data_aug_mlp(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = []   
    labels_aug=[]
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        
        for i in np.arange(idx_len_aug):
            # randomly get 4 samples and average 
            a = rnd.choice(idx,5,replace=True)
            tmp_data = np.mean(indata[a,:],axis=0)
            b = 0.01 * rnd.randn(96)
            tmp_data = tmp_data + b
            tmp_data = tmp_data/lin.norm(tmp_data)
            condn_data_aug.append(tmp_data)
            labels_aug.append(labels[a,:][0,:])
    
    condn_data_aug = np.array(condn_data_aug)
    labels_aug = np.array(labels_aug)
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        tmp_data = indata[idx,:]
        C = np.cov(tmp_data,rowvar=False) + 1e-12*np.eye(tmp_data.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(tmp_data,axis=0)
        X = rnd.randn(idx_len_aug,96)
        new_data = X @ C12 + m
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,num_labels])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol_feature(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,7])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        tmp_data = indata[idx,:]
        
        # doing hg
        hg = tmp_data[:,np.arange(2,96,3)]
        C = np.cov(hg,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(hg,axis=0)
        X = rnd.randn(idx_len_aug,hg.shape[1])
        new_hg = X @ C12 + m
        
        # doing delta
        delta = tmp_data[:,np.arange(0,96,3)]
        C = np.cov(delta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(delta,rowvar=False) +  1e-12*np.eye(delta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(delta,axis=0)
        X = rnd.randn(idx_len_aug,delta.shape[1])
        new_delta = X @ C12 + m
        
        # doing beta
        beta = tmp_data[:,np.arange(1,96,3)]
        C = np.cov(beta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(beta,rowvar=False) +  1e-12*np.eye(beta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(beta,axis=0)
        X = rnd.randn(idx_len_aug,beta.shape[1])
        new_beta = X @ C12 + m
        
        new_data = np.zeros([idx_len_aug,96])
        new_data[:,np.arange(0,96,3)] = new_delta
        new_data[:,np.arange(1,96,3)] = new_beta
        new_data[:,np.arange(2,96,3)] = new_hg
        
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,7])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels


def data_aug_mlp_chol_feature_equalSize(indata,labels,data_size):    
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    N = (data_size/num_labels) #data aug factor
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N-len(idx))
        tmp_data = indata[idx,:]
        
        if idx_len_aug > 0:
            
            # doing hg
            hg = tmp_data[:,np.arange(2,96,3)]
            C = np.cov(hg,rowvar=False)
            if lin.matrix_rank(C)<32:
                C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
            C12 = lin.cholesky(C)
            m = np.mean(hg,axis=0)
            X = rnd.randn(idx_len_aug,hg.shape[1])
            new_hg = X @ C12 + m
            
            # doing delta
            delta = tmp_data[:,np.arange(0,96,3)]
            C = np.cov(delta,rowvar=False)
            if lin.matrix_rank(C)<32:
                C = np.cov(delta,rowvar=False) +  1e-12*np.eye(delta.shape[1])
            C12 = lin.cholesky(C)
            m = np.mean(delta,axis=0)
            X = rnd.randn(idx_len_aug,delta.shape[1])
            new_delta = X @ C12 + m
            
            # doing beta
            beta = tmp_data[:,np.arange(1,96,3)]
            C = np.cov(beta,rowvar=False)
            if lin.matrix_rank(C)<32:
                C = np.cov(beta,rowvar=False) +  1e-12*np.eye(beta.shape[1])
            C12 = lin.cholesky(C)
            m = np.mean(beta,axis=0)
            X = rnd.randn(idx_len_aug,beta.shape[1])
            new_beta = X @ C12 + m
            
            new_data = np.zeros([idx_len_aug,96]) 
            new_data[:,np.arange(0,96,3)] = new_delta
            new_data[:,np.arange(1,96,3)] = new_beta
            new_data[:,np.arange(2,96,3)] = new_hg
            #add some noise
            new_data = new_data + 0.02*rnd.randn(new_data.shape[0],new_data.shape[1])
            
            # make it unit norm
            for i in np.arange(new_data.shape[0]):
                new_data[i,:] = new_data[i,:]/lin.norm(new_data[i,:])
            
            condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
            tmp_labels = np.zeros([idx_len_aug,num_labels])
            tmp_labels[:,query]=1        
            labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol_feature_equalSize_B3_NoPooling(indata,labels,data_size):    
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    N = (data_size/num_labels) #data aug factor
    feat_size = indata.shape[1]
    condn_data_aug = np.empty([0,feat_size]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N-len(idx))
        tmp_data = indata[idx,:]
        
        # no pooling -> first 253 features are delta, 2nd 253 are beta and last are hG
        new_data = np.zeros([idx_len_aug,feat_size]) 
        for i in np.arange(0,indata.shape[1],253):
            hg = tmp_data[:,i:i+253] # just keeping the name hG
            C = np.cov(hg,rowvar=False)
            if lin.matrix_rank(C)<253:
                C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
            C12 = lin.cholesky(C)
            m = np.mean(hg,axis=0)
            X = rnd.randn(idx_len_aug,hg.shape[1])
            new_hg = X @ C12 + m
            new_data[:,i:i+253] = new_hg
        
        #add some noise
        new_data = new_data + 0.02*rnd.randn(new_data.shape[0],new_data.shape[1])
        
        # make it unit norm
        for i in np.arange(new_data.shape[0]):
            new_data[i,:] = new_data[i,:]/lin.norm(new_data[i,:])
        
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,num_labels])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def get_raw_channnel_variances(indata,labels):       
    idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(idx))
    hg_variances=[]
    delta_variances =[]
    beta_variances = []
    for query in np.arange(num_labels):
        
        idx1 = np.where(idx==query)[0]
        indata_tmp = indata[idx1,:]
            
        # hg
        idx2 = np.arange(2,96,3)
        hg = indata_tmp[:,idx2]      
        hg_variances_tmp = np.std(hg,axis=0)
        hg_variances.append(hg_variances_tmp)
        # delta
        idx2 = np.arange(0,96,3)
        delta = indata_tmp[:,idx2]            
        delta_variances_tmp = np.std(delta,axis=0)
        delta_variances.append(delta_variances_tmp)
        #beta
        idx2 = np.arange(1,96,3)
        beta = indata_tmp[:,idx2]  
        beta_variances_tmp = np.std(beta,axis=0)
        beta_variances.append(beta_variances_tmp)
    
    delta_variances = np.array(delta_variances).flatten()
    beta_variances = np.array(beta_variances).flatten()
    hg_variances = np.array(hg_variances).flatten()
    return delta_variances,beta_variances,hg_variances
    
    

### CENTERED LINEAR KERNAL ALIGNMENT TO COMPARE TWO FIXED LAYERED AUTOENCODERS with same dataset
def linear_cka_dist(input_data,model,model1,shuffle_flag,shuffle_flag1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])            
    
    # running the pairwise comparisons 
    D = np.zeros((6,6))
    for m in np.arange(len(layer)):
        out = input_data
        k=0
        while k<=m:                        
            w = torch.transpose(layer[k],0,1)
            b = bias[k]
            #shuffle
            if shuffle_flag == True : 
                idx = torch.randperm(w.nelement())
                w = w.reshape(-1)[idx].reshape(w.size())
                idx = torch.randperm(b.nelement())
                b = b[idx]
            # compute
            out = torch.matmul(out, w) + b
            out_compute=out
            if  k==5 or k==2:
                out=out
            else:
                out = elu(out)
            k=k+1
        
        for n in np.arange(len(layer1)):        
            out1 = input_data
            k=0
            while k<=n:                        
                w = torch.transpose(layer1[k],0,1)
                b = bias1[k]
                if shuffle_flag1 == True :
                    idx = torch.randperm(w.nelement())
                    w = w.reshape(-1)[idx].reshape(w.size())
                    idx = torch.randperm(b.nelement())
                    b = b[idx]
                # compute                        
                out1 = torch.matmul(out1, w) + b
                out1_compute=out1
                if  k==5 or k==2:
                    out1=out1
                else:
                    out1 = elu(out1)
                k=k+1
            
            ### now get the CKA between out and out1
            out = out_compute
            out1 = out1_compute
            X = out.to('cpu').detach().numpy()
            Y = out1.to('cpu').detach().numpy()
            X = X - np.mean(X,axis=0)
            Y = Y - np.mean(Y,axis=0)    
            ###cka using Hinton paper
            a=lin.norm((X.T@X),'fro')
            b=lin.norm((Y.T@Y),'fro')
            c=(lin.norm((Y.T @ X),'fro'))**2
            d = c/(a*b)
            #print(d)
            ####cka using Williams paper
            # a = (X@X.T)
            # a = np.trace(a.T@a)
            # b = Y@Y.T
            # b = np.trace(b.T@b)
            # c = np.trace((X@X.T) @ (Y@Y.T))
            # d= c/(np.sqrt(a*b))
            #print(d)
            
            # PCA PLOTTING NOT NEEDED
            # x=pca.fit_transform(X)
            # y=convert_to_ClassNumbers(torch.from_numpy(Yonline).to(device).float())
            # y = y.to('cpu').detach().numpy()       
            # fig,(ax1,ax2)=plt.subplots(1,2)            
            # ax1.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')            
            # plt.title(str(m))
            # x=pca.fit_transform(Y)
            # y=convert_to_ClassNumbers(torch.from_numpy(Yonline).to(device).float())
            # y = y.to('cpu').detach().numpy()                               
            # ax2.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')
            # plt.show()
            # plt.title(str(n))
    
            
            ### orthogonal procrustus 1
            #R, sca = op(X, Y)
            #d = (lin.norm(X@R - Y))/(lin.norm(Y))
            ### orthogonal procrustus 2
            # X = X/lin.norm(X,'fro')
            # Y = Y/lin.norm(Y,'fro')
            # a=lin.norm(X,'fro')**2
            # b=lin.norm(Y,'fro')**2
            # c=lin.norm(X.T@Y,'nuc')
            # d = a+b-2*c
            
            D[m,n] = d
            
    return(D)


### CENTERED LINEAR KERNAL ALIGNMENT TO COMPARE TWO FIXED LAYERED AUTOENCODERS with same dataset
def linear_cka_dist_2_Datasets(input_data,input_data1,model,model1,shuffle_flag,shuffle_flag1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)
    input_data1 = torch.from_numpy(input_data1).float().to(device)
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])     
    
    # running the pairwise comparisons layer by layer
    # shuffle the weights between the two layers NEW METHOD
    D = np.zeros((6))
    for m in np.arange(len(layer)):
         out = input_data
         out1 = input_data
         k=0
         while k<=m:                        
             w = torch.transpose(layer[k],0,1)
             b = bias[k]
             
             w1 = torch.transpose(layer1[k],0,1)
             b1 = bias1[k]
             
             w_total = torch.concat([w,w1],dim=0)
             b_total = torch.concat([b,b1],dim=0)
             idx = torch.randperm(w_total.nelement())
             w_total = w_total.reshape(-1)[idx].reshape(w_total.size())
             idx = torch.randperm(b_total.nelement())
             b_total = b_total[idx]
             
             s=round(w_total.size()[0]/2)
             w = w_total[:s,:]
             w1 = w_total[s:,:]
             s=round(b_total.size()[0]/2)
             b = b_total[:s]
             b1 = b_total[s:]
             
             out = torch.matmul(out, w1) + b1 #shuffle and swap weights
             out1 = torch.matmul(out1, w) + b #shuffle and swap weights   
             out_compute=out;
             out1_compute=out1
             
             if  k==5 or k==2:
                 out1=out1
                 out=out
             else:
                 out1 = elu(out1)
                 out = elu(out)
             k=k+1
                        
         ### now get the CKA between out and out1
         out = out_compute
         out1 = out1_compute
         X = out.to('cpu').detach().numpy()
         Y = out1.to('cpu').detach().numpy()
         X = X - np.mean(X,axis=0)
         Y = Y - np.mean(Y,axis=0)    
         ###cka using Hinton paper
         a=lin.norm((X.T@X),'fro')
         b=lin.norm((Y.T@Y),'fro')
         c=(lin.norm((Y.T @ X),'fro'))**2
         d = c/(a*b)
         #print(d)
         
         D[m] = d
             
    return(D)       
    
    # # running the pairwise comparisons  OLD METHOD
    # D = np.zeros((6,6))
    # for m in np.arange(len(layer)):
    #     out = input_data
    #     k=0
    #     while k<=m:                        
    #         w = torch.transpose(layer[k],0,1)
    #         b = bias[k]
    #         #shuffle
    #         if shuffle_flag == True : 
    #             idx = torch.randperm(w.nelement())
    #             w = w.reshape(-1)[idx].reshape(w.size())
    #             idx = torch.randperm(b.nelement())
    #             b = b[idx]
    #         # compute
    #         out = torch.matmul(out, w) + b
    #         out_compute=out
    #         if  k==5 or k==2:
    #             out=out
    #         else:
    #             out = elu(out)
    #         k=k+1
        
    #     for n in np.arange(len(layer1)):        
    #         out1 = input_data1
    #         k=0
    #         while k<=n:                        
    #             w = torch.transpose(layer1[k],0,1)
    #             b = bias1[k]
    #             if shuffle_flag1 == True :
    #                 idx = torch.randperm(w.nelement())
    #                 w = w.reshape(-1)[idx].reshape(w.size())
    #                 idx = torch.randperm(b.nelement())
    #                 b = b[idx]
    #             # compute                        
    #             out1 = torch.matmul(out1, w) + b
    #             out1_compute=out1
    #             if  k==5 or k==2:
    #                 out1=out1
    #             else:
    #                 out1 = elu(out1)
    #             k=k+1
            
    #         ### now get the CKA between out and out1
    #         out = out_compute
    #         out1 = out1_compute
    #         X = out.to('cpu').detach().numpy()
    #         Y = out1.to('cpu').detach().numpy()
    #         X = X - np.mean(X,axis=0)
    #         Y = Y - np.mean(Y,axis=0)    
    #         ###cka using Hinton paper
    #         a=lin.norm((X.T@X),'fro')
    #         b=lin.norm((Y.T@Y),'fro')
    #         c=(lin.norm((Y.T @ X),'fro'))**2
    #         d = c/(a*b)
    #         #print(d)
    #         ####cka using Williams paper
    #         # a = (X@X.T)
    #         # a = np.trace(a.T@a)
    #         # b = Y@Y.T
    #         # b = np.trace(b.T@b)
    #         # c = np.trace((X@X.T) @ (Y@Y.T))
    #         # d= c/(np.sqrt(a*b))
    #         #print(d)
            
    #         # PCA PLOTTING NOT NEEDED
    #         # x=pca.fit_transform(X)
    #         # y=convert_to_ClassNumbers(torch.from_numpy(Yonline).to(device).float())
    #         # y = y.to('cpu').detach().numpy()       
    #         # fig,(ax1,ax2)=plt.subplots(1,2)            
    #         # ax1.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')            
    #         # plt.title(str(m))
    #         # x=pca.fit_transform(Y)
    #         # y=convert_to_ClassNumbers(torch.from_numpy(Yonline).to(device).float())
    #         # y = y.to('cpu').detach().numpy()                               
    #         # ax2.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')
    #         # plt.show()
    #         # plt.title(str(n))
    
            
    #         ### orthogonal procrustus 1
    #         #R, sca = op(X, Y)
    #         #d = (lin.norm(X@R - Y))/(lin.norm(Y))
    #         ### orthogonal procrustus 2
    #         # X = X/lin.norm(X,'fro')
    #         # Y = Y/lin.norm(Y,'fro')
    #         # a=lin.norm(X,'fro')**2
    #         # b=lin.norm(Y,'fro')**2
    #         # c=lin.norm(X.T@Y,'nuc')
    #         # d = a+b-2*c
            
    #         D[m,n] = d
            
    # return(D)



def linear_cka_dist_shuffleLayers_Between_AE(input_data,model,model1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)    
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])            
    
    # running the pairwise comparisons layer by layer
    # shuffle the weights between the two layers
    D = np.zeros((6))
    for m in np.arange(len(layer)):
        out = input_data
        out1 = input_data
        k=0
        while k<=m:                        
            w = torch.transpose(layer[k],0,1)
            b = bias[k]
            
            w1 = torch.transpose(layer1[k],0,1)
            b1 = bias1[k]
            
            w_total = torch.concat([w,w1],dim=0)
            b_total = torch.concat([b,b1],dim=0)
            idx = torch.randperm(w_total.nelement())
            w_total = w_total.reshape(-1)[idx].reshape(w_total.size())
            idx = torch.randperm(b_total.nelement())
            b_total = b_total[idx]
            
            s=round(w_total.size()[0]/2)
            w = w_total[:s,:]
            w1 = w_total[s:,:]
            s=round(b_total.size()[0]/2)
            b = b_total[:s]
            b1 = b_total[s:]
            
            out = torch.matmul(out, w1) + b1 #shuffle and swap weights
            out1 = torch.matmul(out1, w) + b #shuffle and swap weights   
            out_compute=out;
            out1_compute=out1
            
            if  k==5 or k==2:
                out1=out1
                out=out
            else:
                out1 = elu(out1)
                out = elu(out)
            k=k+1
                       
        ### now get the CKA between out and out1
        out = out_compute
        out1 = out1_compute
        X = out.to('cpu').detach().numpy()
        Y = out1.to('cpu').detach().numpy()
        X = X - np.mean(X,axis=0)
        Y = Y - np.mean(Y,axis=0)    
        ###cka using Hinton paper
        a=lin.norm((X.T@X),'fro')
        b=lin.norm((Y.T@Y),'fro')
        c=(lin.norm((Y.T @ X),'fro'))**2
        d = c/(a*b)
        #print(d)
        
        D[m] = d
            
    return(D)







def linear_cka_dist2(input_data,input_data1,model,model1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)
    input_data1 = torch.from_numpy(input_data1).float().to(device)
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])
    
      
    # running the pairwise comparisons 
    D = np.zeros((6,6))
    for m in np.arange(len(layer)):
        out = input_data
        k=0
        while k<=m:                        
            w = torch.transpose(layer[k],0,1)
            b = bias[k]
            out = torch.matmul(out, w) + b
            if k==2 or k==5:
                out=out
            else:
                out = elu(out)
            k=k+1
        
        for n in np.arange(len(layer1)):
            out1 = input_data1
            k=0
            while k<=n:                        
                w = torch.transpose(layer1[k],0,1)
                b = bias1[k]
                out1 = torch.matmul(out1, w) + b
                if k==2 or k==5:
                    out1=out1
                else:
                    out1 = elu(out1)
                k=k+1
            
            ### now get the CKA between out and out1
            X = out.to('cpu').detach().numpy()
            Y = out1.to('cpu').detach().numpy()
            X = X - np.mean(X,axis=0)
            Y = Y - np.mean(Y,axis=0)    
            ###cka
            a=lin.norm((X.T@X),'fro')
            b=lin.norm((Y.T@Y),'fro')
            c=(lin.norm((Y.T @ X),'fro'))**2
            d = c/(a*b)
            ### orthogonal procrustus 1
            #R, sca = op(X, Y)
            #d = (lin.norm(X@R - Y))/(lin.norm(Y))
            ### orthogonal procrustus 2
            # X = X/lin.norm(X,'fro')
            # Y = Y/lin.norm(Y,'fro')
            # a=lin.norm(X,'fro')**2
            # b=lin.norm(Y,'fro')**2
            # c=lin.norm(X.T@Y,'nuc')
            # d = a+b-2*c
            
            D[m,n] = d
            
    return(D)


def shuffle_weights(model,model1):
    model.eval()
    model1.eval()
    elu=nn.ELU()    
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])
    
    # shuffle the weights of first NN
    for n in np.arange(len(layer)):  
        w = torch.transpose(layer[n],0,1)
        b = bias[n]        
        idx = torch.randperm(w.nelement())
        w = w.reshape(-1)[idx].reshape(w.size())
        idx = torch.randperm(b.nelement())
        b = b[idx]
        layer[n] = torch.transpose(w,0,1)
        bias[n] = b
        
    
    model.encoder.linear1.state_dict()['weight'].copy_(layer[0])    
    model.encoder.linear2.state_dict()['weight'].copy_(layer[1])
    model.encoder.linear3.state_dict()['weight'].copy_(layer[2])
    model.decoder.linear1.state_dict()['weight'].copy_(layer[3])
    model.decoder.linear2.state_dict()['weight'].copy_(layer[4])
    model.decoder.linear3.state_dict()['weight'].copy_(layer[5])
    model.encoder.linear1.state_dict()['bias'].copy_(bias[0])
    model.encoder.linear2.state_dict()['bias'].copy_(bias[1])
    model.encoder.linear3.state_dict()['bias'].copy_(bias[2])
    model.decoder.linear1.state_dict()['bias'].copy_(bias[3])
    model.decoder.linear2.state_dict()['bias'].copy_(bias[4])
    model.decoder.linear3.state_dict()['bias'].copy_(bias[5])
    
    
    
    # shuffle the weights of second NN
    for n in np.arange(len(layer1)):  
        w = torch.transpose(layer1[n],0,1)
        b = bias1[n]        
        idx = torch.randperm(w.nelement())
        w = w.reshape(-1)[idx].reshape(w.size())
        idx = torch.randperm(b.nelement())
        b = b[idx]
        layer1[n] = torch.transpose(w,0,1)
        bias1[n] = b
    
    model1.encoder.linear1.state_dict()['weight'].copy_(layer1[0])    
    model1.encoder.linear2.state_dict()['weight'].copy_(layer1[1])
    model1.encoder.linear3.state_dict()['weight'].copy_(layer1[2])
    model1.decoder.linear1.state_dict()['weight'].copy_(layer1[3])
    model1.decoder.linear2.state_dict()['weight'].copy_(layer1[4])
    model1.decoder.linear3.state_dict()['weight'].copy_(layer1[5])
    model1.encoder.linear1.state_dict()['bias'].copy_(bias1[0])
    model1.encoder.linear2.state_dict()['bias'].copy_(bias1[1])
    model1.encoder.linear3.state_dict()['bias'].copy_(bias1[2])
    model1.decoder.linear1.state_dict()['bias'].copy_(bias1[3])
    model1.decoder.linear2.state_dict()['bias'].copy_(bias1[4])
    model1.decoder.linear3.state_dict()['bias'].copy_(bias1[5])
    
    return model,model1


def shuffle_weights_activations(model1,input_data):
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])
    elu = nn.ELU()
    
    out1=input_data
    for n in np.arange(len(layer1)):  
        w = torch.transpose(layer1[n],0,1)
        b = bias1[n]        
        idx = torch.randperm(w.nelement())
        w = w.reshape(-1)[idx].reshape(w.size())
        idx = torch.randperm(b.nelement())
        b = b[idx]        
        out1 = torch.matmul(out1, w) + b
        if n==2 or n==5:
            out1=out1
        else:
            out1 = elu(out1)
    
    return out1
    
    

 
def eval_ae_similarity(model,model1,condn_data_total,condn_data_total1,
                                  Ytotal,Ytotal1):
    
    original_features = []
    recon_features_origManifold = []
    recon_features_swappedManifold = []
    
    # For the first dataset
    idx = np.argmax(Ytotal,axis=1)
    for ii in np.arange(len(np.unique(idx))):            
        # get the indices for movement
        idx1 = np.where(idx==ii)[0]
        input_data = condn_data_total[idx1,:]
        # compute the average
        avg = np.mean(input_data,axis=0)
        # pass it thru AEs
        input_data = torch.from_numpy(input_data).to(device).float()
        z,y=model(input_data) # pass it thru its own AE 
        z1,y1=model1(input_data) # pass it thru another day AE
        z2 = shuffle_weights_activations(model,input_data)
        # get the data
        input_data_recon = z.to('cpu').detach().numpy()
        input_data_recon1 = z1.to('cpu').detach().numpy()
        input_data_recon1_shuffle = z2.to('cpu').detach().numpy()
        input_data = input_data.to('cpu').detach().numpy()
        # compute error norm
        recon_error_sameDayManifold = lin.norm((avg - input_data_recon),'fro')
        recon_error_diffDayManifold = lin.norm((avg - input_data_recon1),'fro')
        raw_error = lin.norm((avg - input_data),'fro')
        recon_error_shuffle = lin.norm((avg - input_data_recon1_shuffle),'fro')
        # store results
        original_features.append(raw_error)
        recon_features_origManifold.append(recon_error_sameDayManifold)
        recon_features_swappedManifold.append(recon_error_diffDayManifold)
    
    # for the second dataset
    idx = np.argmax(Ytotal1,axis=1)
    for ii in np.arange(len(np.unique(idx))):            
        # get the indices for movement
        idx1 = np.where(idx==ii)[0]
        input_data = condn_data_total1[idx1,:]
        # compute the average
        avg = np.mean(input_data,axis=0)
        # pass it thru AEs
        input_data = torch.from_numpy(input_data).to(device).float()
        z,y=model1(input_data) # pass it thru its own AE 
        z1,y1=model(input_data) # pass it thru another day AE
        # get the data
        input_data_recon = z.to('cpu').detach().numpy()
        input_data_recon1 = z1.to('cpu').detach().numpy()
        input_data = input_data.to('cpu').detach().numpy()
        # compute error norm
        recon_error_sameDayManifold = lin.norm((avg - input_data_recon),'fro')
        recon_error_diffDayManifold = lin.norm((avg - input_data_recon1),'fro')
        raw_error = lin.norm((avg - input_data),'fro')
        # store results
        original_features.append(raw_error)
        recon_features_origManifold.append(recon_error_sameDayManifold)
        recon_features_swappedManifold.append(recon_error_diffDayManifold)
        # plt.figure();
        # plt.boxplot([original_features,recon_features_origManifold,
        # recon_features_swappedManifold])
    
    return original_features,recon_features_origManifold,recon_features_swappedManifold
    


# # pass it thru AE after scrambling its weights
# model_shuffle,model1_shuffle = shuffle_weights(model,model1)
# idx1 = np.where(idx==ii)[0]
# input_data = condn_data_total[idx1,:]
# input_data = torch.from_numpy(input_data).to(device).float()
# z2,y2 = model1_shuffle(input_data)
# input_data_recon_shuffle = z2.to('cpu').detach().numpy()
# recon_error_diffDayManifold_shuffle = lin.norm((avg - input_data_recon_shuffle),'fro')
    
    


def fdr_threshold(pval,q,fdrType):
    """
    Implementation of the FDR procedure, from EEGLAB code base by Arnaud Delorme
    Requires as input: pval, the array of p-values
                       q the query p-value to be thresholded (e.g. 0.05)
                       fdrType should be 'Parametric' or 'Nonparametric'
    
    Output is: pid the thresholded multiple comparison p-value from the input pvas
               pval_thresholded, True or False if any of the pvals satisfied the threshold 
    
    """ 
    pval= pval[:,None]
    p = np.sort(pval,axis=0)
    V = len(p)
    I = np.arange(V)+1
    I = I[:,None]
        
    cVID = 1;
    cVN = sum(1./ (np.arange(V)+1))
    
    if fdrType == 'Parametric':
       a=np.where(p<=I/V*q/cVID)[0]       
       if len(a)>0 :
           pid = p[np.max(a)]
       else:
           pid=0
     
    if fdrType == 'Nonparametric':
        a=np.where(p<=I/V*q/cVN)[0]
        if len(a)>0 :
            pid = p[np.max(a)]
        else:
            pid=0
    
    pval_thresholded = pval<=pid
    return pid,pval_thresholded
    
        
        

#bootstrapped test for difference in medians or means etc.
def bootstrap_difference_test(a,b,test_type):    
    if test_type == 'median':
        stat1 = np.median(a)
        stat2 = np.median(b)
    if test_type ==  'mean':
        stat1 = np.mean(a)
        stat2 = np.mean(b)
    
    stat = stat1-stat2
    
    # run the bootstrap statistics i.e., sample with replacement
    if test_type == 'median': 
        a1 = a-np.median(a) 
        b1 = b-np.median(b)
    if test_type == 'mean': 
        a1 = a-np.mean(a) 
        b1 = b-np.mean(b)
    boot_stat = []
    for i in np.arange(1e4):
        atemp = rnd.choice(a1,len(a1),replace=True)        
        btemp = rnd.choice(b1,len(b1),replace=True)        
        if test_type =='median':
            boot_stat.append(np.median(atemp) - np.median(btemp))            
        if test_type =='mean':
            boot_stat.append(np.mean(atemp) - np.mean(btemp))
    
    plt.figure();
    plt.hist(np.abs(boot_stat))
    plt.axvline(np.abs(stat),color='r')
    pvalue = np.sum(np.abs(boot_stat) >= np.abs(stat))/len(boot_stat)
    plt.title('pvalue is ' + str(pvalue))
    return pvalue,boot_stat

    
    
#%% ABLATION EXPERIMENTS AND STUFF


# ==== ABLATION HOOK ====
def make_channel_ablation_hook(indices):
    def hook(module, input, output):
        output[:, indices] = 0
        return output
    return hook

# ==== FIND CONV LAYERS ====
def find_conv_layers_encoder_real(model):
    conv_layers_real = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ('real' in name):
            conv_layers_real[name] = module
    return conv_layers_real

def find_conv_layers_encoder_imag(model):
    conv_layers_imag = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ( 'imag' in name):
            conv_layers_imag[name] = module
    return conv_layers_imag



def find_conv_layers_decoder_real(model):
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d) and ('real' in name ):
            conv_layers[name] = module
    return conv_layers


def find_conv_layers_decoder_imag(model):
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d) and ( 'imag' in name):
            conv_layers[name] = module
    return conv_layers




# ==== ABLATION LOOP ====
def ablate_kernels_encoder(model, Xtest,Ytest):
    conv_layers_real = find_conv_layers_encoder_real(model)
    conv_layers_imag = find_conv_layers_encoder_imag(model)
    a=iter(conv_layers_real.items())
    b=iter(conv_layers_imag.items())
    results = {}
    tmp_ydata_r,tmp_ydata_i = Ytest.real, Ytest.imag
    tmp_ydata_r = torch.from_numpy(tmp_ydata_r)
    tmp_ydata_i = torch.from_numpy(tmp_ydata_i)
    
    for i in np.arange(len(conv_layers_real)):
        real_layer_name,real_layer = next(a)
        imag_layer_name,imag_layer = next(b)        
        lay_name = 'conv' + str(i+1)
        results[lay_name] = {}
        num_kernels = imag_layer.out_channels
        
        for k in range(num_kernels):
            print(f"Ablating complex {lay_name}[{k}]...")
            hook_real = real_layer.register_forward_hook(make_channel_ablation_hook([k]))
            hook_imag = imag_layer.register_forward_hook(make_channel_ablation_hook([k]))
            
            recon_r,recon_i,decodes = test_model_complex(model,Xtest)
            
            recon_r = torch.from_numpy(recon_r)            
            recon_i = torch.from_numpy(recon_i)            
            
            loss =  F.mse_loss(recon_r, tmp_ydata_r) + F.mse_loss(recon_i, tmp_ydata_i)
            results[lay_name][k] = loss
            hook_real.remove()           
            hook_imag.remove()   

    
    return results

def ablate_kernels_decoder(model, Xtest,Ytest):
    conv_layers_real = find_conv_layers_decoder_real(model)
    conv_layers_imag = find_conv_layers_decoder_imag(model)
    a=iter(conv_layers_real.items())
    b=iter(conv_layers_imag.items())
    results = {}
    tmp_ydata_r,tmp_ydata_i = Ytest.real, Ytest.imag
    tmp_ydata_r = torch.from_numpy(tmp_ydata_r)
    tmp_ydata_i = torch.from_numpy(tmp_ydata_i)
    
    for i in np.arange(len(conv_layers_real)):
        real_layer_name,real_layer = next(a)
        imag_layer_name,imag_layer = next(b)        
        lay_name = 'deconv' + str(i+1)
        results[lay_name] = {}
        num_kernels = imag_layer.out_channels
        
        for k in range(num_kernels):
            print(f"Ablating complex {lay_name}[{k}]...")
            hook_real = real_layer.register_forward_hook(make_channel_ablation_hook([k]))
            hook_imag = imag_layer.register_forward_hook(make_channel_ablation_hook([k]))
            
            recon_r,recon_i,decodes = test_model_complex(model,Xtest)
            
            recon_r = torch.from_numpy(recon_r)            
            recon_i = torch.from_numpy(recon_i)            
            
            loss =  F.mse_loss(recon_r, tmp_ydata_r) + F.mse_loss(recon_i, tmp_ydata_i)
            results[lay_name][k] = loss
            hook_real.remove()           
            hook_imag.remove()   

    
    return results



# dist_recon_error_mean=[]
# dist_single_trial_mean=[]
# a = lin.norm(avg)


# for i in np.arange(input_data.shape[0]):
#     err = avg - input_data[i,:]
#     b = lin.norm(err)
#     c = (a-b)/a
#     dist_single_trial_mean.append(b)
    
#     err = avg - input_data_recon[i,:]
#     b = lin.norm(err)
#     c = (a-b)/a
#     dist_recon_error_mean.append(b)

# plt.figure()
# plt.boxplot([dist_single_trial_mean,dist_recon_error_mean])
# plt.ylabel('Error between single trial and mean')
# plt.xticks(ticks=[1,2],labels=('Raw','Reconstructed from AE'))


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import correlate,correlation_lags

# # Generate a noisy sinewave
# t = np.arange(1, 101)
# b = np.abs(np.sin(2 * np.pi * 1 / 100 * t)) + 0.1 * np.random.randn(100)
# b_conv = np.convolve(b, b, mode='same') + np.random.randn(100)

# # Plot 
# plt.figure()
# plt.plot(b_conv)

# # Calculate the autocorrelation
# r = correlate(b_conv, b_conv, mode='full', method='auto')
# lags = correlation_lags(b_conv.shape[0],b_conv.shape[0],mode='full')

# # Plot the cross-correlation
# plt.figure()
# plt.stem(lags,r)
# plt.show()





# # # testing the various sizes of the convolutional layers 
# input = torch.randn(32,1,11,23,40)


# # layer 1: 9,21,99 is output
# m=nn.Conv3d(1,12,kernel_size=2,stride=(1,1,2))
# a = nn.AvgPool3d(kernel_size=2,stride=1)
# bn = nn.BatchNorm3d(num_features=12)
# r = nn.ELU()
# out = m(input)
# out = bn(out)
# out = r(out)
# #out = a(out)

# #layer 2
# m = nn.Conv3d(12,12,kernel_size=2,stride=(1,2,2))
# a = nn.AvgPool3d(kernel_size=2,stride=1)
# out = m(out)
# out = r(out)
# #out = a(out)

# # layer 3
# m = nn.Conv3d(12,12,kernel_size=2,stride=(1,1,2))
# out = m(out)
# out = r(out)

# # layer 4
# m = nn.Conv3d(12,6,kernel_size=2,stride=(1,1,1))
# out = m(out)
# out = r(out)

# # pass to lstm for classification
# tmp = torch.flatten(out,start_dim=1,end_dim=3)
# x=tmp
# x = torch.permute(x,(0,2,1))
# rnn1 = nn.LSTM(input_size=378,hidden_size=48,batch_first=True,bidirectional=False)
# output,(hn,cn) = rnn1(x)
# #output1,(hn1,cn1) = rnn2(output)
# hn=torch.squeeze(hn)
# linear0 = nn.Linear(48,2)
# out=linear0(hn)


# # tmp = out.view()

# # # bottleneck enc side
# # out = out.view(out.size(0), -1) 
# # m = nn.Linear(out.shape[1], 128)    
# # out = m(out)

# # # bottleneck dec side
# # m = nn.Linear(128,6*7*19*25)    
# # out = m(out)

# # layer 4
# #out = out.view(out.size(0), 6,7, 19,25)

# m = nn.ConvTranspose3d(6,12,kernel_size=2,stride=(1,1,1),output_padding=(0,0,0))
# out = m(out)
# out = r(out)

# # layer 3
# m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
# out = m(out)
# out = r(out)

# out_tmp=out;

# # layer 2 want  10,22,100
# out=out_tmp
# m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,2,2),
#                       padding=(0,0,0), output_padding=(0,0,0))
# out = m(out)
# out = r(out)
# print(out.shape)

# out_tmp=out;


# # layer 1 #11,23,200
# out=out_tmp
# m = nn.ConvTranspose3d(12,1,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
# out = m(out)
# out = r(out)
# print(out.shape)

# for B1


# # # testing the various sizes of the convolutional layers 
# input = torch.randn(32,1,6,9,40)


# # layer 1: 9,21,99 is output (this is for full grid)
# m=nn.Conv3d(1,12,kernel_size=2,stride=(1,1,2))
# a = nn.AvgPool3d(kernel_size=2,stride=1)
# bn = nn.BatchNorm3d(num_features=12)
# r = nn.ELU()
# out = m(input)
# #out = bn(out)
# out = r(out)
# #out = a(out)

# #layer 2
# m = nn.Conv3d(12,12,kernel_size=2,stride=(1,2,2))
# a = nn.AvgPool3d(kernel_size=2,stride=1)
# out = m(out)
# out = r(out)
# #out = a(out)

# # layer 3
# m = nn.Conv3d(12,6,kernel_size=2,stride=(1,1,2))
# out = m(out)
# out = r(out)

# # layer 4
# # m = nn.Conv3d(12,6,kernel_size=2,stride=(1,1,1))
# # out = m(out)
# # out = r(out)

# # pass to lstm for classification
# tmp = torch.flatten(out,start_dim=1,end_dim=3)
# x=tmp
# x = torch.permute(x,(0,2,1))
# input_size=x.shape[-1]
# hidden_size=16
# rnn1 = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True,
#                bidirectional=False)
# output,(hn,cn) = rnn1(x)
# #output1,(hn1,cn1) = rnn2(output)
# hn=torch.squeeze(hn)
# linear0 = nn.Linear(hidden_size,2)
# out_class=linear0(hn)


# # tmp = out.view()

# # # bottleneck enc side
# # out = out.view(out.size(0), -1) 
# # m = nn.Linear(out.shape[1], 128)    
# # out = m(out)

# # # bottleneck dec side
# # m = nn.Linear(128,6*7*19*25)    
# # out = m(out)

# # layer 4
# #out = out.view(out.size(0), 6,7, 19,25)

# m = nn.ConvTranspose3d(6,12,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
# out = m(out)
# out = r(out)

# # layer 3
# #m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,2,2),output_padding=(0,1,0))
# m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,2,2),output_padding=(0,0,0))
# out = m(out)
# out = r(out)

# out_tmp=out;

# # # layer 2 want  10,22,100
# # out=out_tmp
# # m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,2,2),
# #                       padding=(0,0,0), output_padding=(0,0,0))
# # out = m(out)
# # out = r(out)
# # print(out.shape)

# # out_tmp=out;


# # layer 1 #11,23,200
# out=out_tmp
# m = nn.ConvTranspose3d(12,1,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
# out = m(out)
# out = r(out)
# print(out.shape)