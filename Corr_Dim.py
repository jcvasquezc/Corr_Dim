# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:36:04 2016

@author: J. C. Vasquez-Correa
"""

import numpy as np
import math
from statsmodels.tsa.tsatools import lagmat
from sklearn.metrics.pairwise import euclidean_distances as dist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Dim_Corr(datas, Tao, m, graph=False): 
	"""
	Compute the correlation dimension of a time series with a time-lag Tao and an embedding dimension m
	datas--> time series to compute the correlation dimension
	Tao--> time lag computed using the first zero crossing of the auto-correlation function (see Tao func)   
	m--> embeding dimension of the time-series, computed using the false neighbors method (see fnn func)  
	graph (optional)--> plot the phase space (attractor) in 3D
	"""
	x=PhaseSpace(datas, m, Tao, graph)
	ED2=dist(x.T)
	posD=np.triu_indices_from(ED2, k=1)
	ED=ED2[posD]
	max_eps=np.max(ED)
	min_eps=np.min(ED[np.where(ED>0)])
	max_eps=np.exp(math.floor(np.log(max_eps)))
	n_div=int(math.floor(np.log(max_eps/min_eps)))
	n_eps=n_div+1
	eps_vec=range(n_eps)
	unos=np.ones([len(eps_vec)])*-1
	eps_vec1=max_eps*np.exp(unos*eps_vec-unos)
	Npairs=((len(x[1,:]))*((len(x[1,:])-1)))
	C_eps=np.zeros(n_eps)
 
	for i in eps_vec:
        	eps=eps_vec1[i]
        	N=np.where(((ED<eps) & (ED>0)))
        	S=len(N[0])
        	C_eps[i]=float(S)/Npairs

	omit_pts=1 
	k1=omit_pts
	k2=n_eps-omit_pts
	xd=np.log(eps_vec1)
	yd=np.log(C_eps)
	xp=xd[k1:k2]
	yp=yd[k1:k2]
	p = np.polyfit(xp, yp, 1)
	return p[0]


def PhaseSpace(data, m, Tao, graph=False):
  """
  Compute the phase space (attractor) a time series data with a time-lag Tao and an embedding dimension m
  data--> time series
  Tao--> time lag computed using the first zero crossing of the auto-correlation function (see Tao func)   
  m--> embeding dimension of the time-series, computed using the false neighbors method (see fnn func)  
  graph (optional)--> plot the phase space (attractor)
  """		
  ld=len(data)
  x = np.zeros([m, (ld-(m-1)*Tao)])
  for j in range(m):
      l1=(Tao*(j))
      l2=(Tao*(j)+len(x[1,:]))
      x[j,:]=data[l1:l2]
  if graph:
     fig = plt.figure()
     if m>2:
         ax = fig.add_subplot(111, projection='3d')
         ax.plot(x[0,:], x[1,:], x[2,:])
     else:
         ax = fig.add_subplot(111)
         ax.plot(x[0,:], x[1,:])         
  return x


def Tao(data):
    """
    Compute the time-lag of a time series data to build the phase space using the first zero crossing rate criterion
    data--> time series
    """    
    corr=np.correlate(data, data, mode="full")
    corr=corr[len(corr)/2:len(corr)]
    tau=0
    j=0
    while (corr[j]>0):
      j=j+1
    tau=j
    return tau


def fnn(data, maxm):
    """
    Compute the embedding dimension of a time series data to build the phase space using the false neighbors criterion
    data--> time series
    maxm--> maximmum embeding dimension
    """    
    RT=15.0
    AT=2
    sigmay=np.std(data, ddof=1)
    nyr=len(data)
    m=maxm
    EM=lagmat(data, maxlag=m-1)
    EEM=np.asarray([EM[j,:] for j in range(m-1, EM.shape[0])])
    embedm=maxm
    for k in range(AT,EEM.shape[1]+1):
        fnn1=[]
        fnn2=[]
        Ma=EEM[:,range(k)]
        D=dist(Ma)
        for i in range(1,EEM.shape[0]-m-k):
            #print D.shape            
            #print(D[i,range(i-1)])
            d=D[i,:]
            pdnz=np.where(d>0)
            dnz=d[pdnz]
            Rm=np.min(dnz)
            l=np.where(d==Rm)
            l=l[0]
            l=l[len(l)-1]
            if l+m+k-1<nyr:
                fnn1.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/Rm)
                fnn2.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/sigmay)
        Ind1=np.where(np.asarray(fnn1)>RT)
        Ind2=np.where(np.asarray(fnn2)>AT)
        if len(Ind1[0])/float(len(fnn1))<0.1 and len(Ind2[0])/float(len(fnn2))<0.1:
            embedm=k
            break
    return embedm