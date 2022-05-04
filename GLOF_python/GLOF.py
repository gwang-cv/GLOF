'''
GLOF, PR'2021
'''
import numpy as np
from sklearn.neighbors import KDTree

def LOF_MultiScaleScoreV2(X, Y,  vec, d2,testK,thresh,alpha):
    C=np.zeros((1,vec.shape[0]))  #(1,n)
    num=0
    treeX = KDTree(X)
    treeY = KDTree(Y)
    for ks in testK: 
        _, neighborX = treeX.query(X, k=ks+1)  # neighborX (n, k)
        _, neighborY = treeY.query(Y, k=ks+1)
        lofs, c = LOFv2(vec,d2, ks,neighborX,neighborY,alpha)
        C=C+c
        num=num+1
    C=(C)/num
    P= np.where(C <= thresh)  #*np.ones(1,vec.shape[1])
    return P[0], C

def LOFv2(A,d2, k,neighborX,neighborY,alpha):
    if k < 1:
        numrows,_ = A.shape()
        k = round(k * numrows) 
    treeA = KDTree(A)
    k_dist, k_index = treeA.query(A, k+1)  #k_dist, k_index (n,k+1)
    k_index = [x[1:] for x in k_index]  # (n (10,))
    numneigh = [len(x) for x in k_index]
    k_dist1 = [x[-1] for x in k_dist]  #(n,)
    n = len(A[:,0])
    lrd_value = np.zeros((n,1))  #(n,1)
    for i in range(n):
        lv = lrd(A, i, k_dist1, k_index, numneigh[i])
        lrd_value[i] =  lv  # (n,1)
    lof = np.zeros((n,1))
    for i in range(n):
        ls = np.sum(lrd_value[k_index[i]]/lrd_value[i])/numneigh[i]
        lof[i] = ls
    L = neighborX.shape[0]
    neighborIndex = np.hstack((neighborX,neighborY))
    index = np.sort(neighborIndex,axis=1)
    temp1 = np.hstack((np.diff(index,axis = 1),np.ones((L,1))))
    temp2 = (temp1==0).astype('int')  #(n,22)
    ni = np.sum(temp2,axis=1)
    d2i = d2[index]
    #alpha= 0.75 #0.5 #0.76   tune
    c1 = k-ni   
    lof=1/(np.ones((lof.shape[0],lof.shape[1]))+np.exp(1-lof))
    cos_sita= np.tile(lof,(1,temp2.shape[1]))
    c3i = (cos_sita >= alpha).astype('int')  # #(n,n)
    c3i0 = c3i*temp2
    c3 = np.sum(c3i0,axis=1) 
    C =  (c1 + c3) / k  
    return lof,C

def lrd(A, index_p, k_dist,k_index, numneighbors): 
    Temp = np.tile(A[index_p,:], (numneighbors, 1)) -  A[k_index[index_p], :]  ## A[k_index[index_p], :]-> (k,:)
    Temp2 = [[y**2 for y in x] for x in Temp]
    Temp = np.sqrt(np.sum(Temp2,axis=1)) #(k,)
    k_indexs = k_index[index_p] # (10,)
    k_dists = [k_dist[x] for x in k_indexs]
    temp_dis = [Temp.tolist(), k_dists]
    reach_dist = np.max(temp_dis,axis=1)
    lrd_value = numneighbors/np.sum(reach_dist)
    return lrd_value

def GLOF_filter(X, Y):  
    beta1 =0.75 
    alpha =0.75
    testK = [10, 15, 20, 25, 30]
    vec = Y - X  # (n,2)
    d2 = np.sum(vec**2,axis=1)  # (n,)
    P,result = LOF_MultiScaleScoreV2(X, Y,  vec, d2,testK,beta1,alpha)
    P = result < beta1 
    mask = np.zeros((X.shape[0],1))
    mask[P[0]] = 1  
    return mask.flatten().astype('bool')
