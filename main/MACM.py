import numpy as np
def MACM(df,lag):
    T, N = np.shape(df)
    mean_df=np.mean(df,0)
    cov=np.zeros((lag+1,N,N))
    for i in range(lag+1):
        for t in range(i, T):
            cov[i] += np.outer(df[t]-mean_df, df[t-i]-mean_df)/(T-i)
    MACM=np.zeros((lag,N,N))

    cov_0=np.outer(np.sqrt(np.diag(cov[0])),np.sqrt(np.diag(cov[0])))
    for k in range(lag):
        MACM[k]=cov[k+1]/cov_0
    return MACM

