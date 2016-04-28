Corr_Dim
-------
Python implementation to Compute the Correlation Dimension of a Time Series x[n]


Usage
-------
::
  m=fnn(x, 15)
  tau=Tao(x)
  cd=Dim_Corr(x, tau, m)

Example 1 
-------
::
  import numpy as np
  t=np.asarray(range(1000))/1000.0
  x=np.sin(2*np.pi*10*t)
  m=fnn(x, 15)
  tau=Tao(x)
  cd=Dim_Corr(x, tau, m, True)
  print 'embeding dimension='+str(m)
  print 'time-lag='+str(tau)
  print 'correlation dimension='+str(cd)
  

