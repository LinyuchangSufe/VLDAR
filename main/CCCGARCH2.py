from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma


class mgarch:
    
    def __init__(self, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
            
    def garch_fit(self, returns):
        
        res = minimize( self.garch_loglike, (0.01, 0.01, 0.94), args = returns,
              bounds = ((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        LogL =0.5* np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns**2)/(2*var_t))
        return -LogL

    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)     
        for i in range(T):
            if i==0:
                var_t[i] = returns[i]**2
            else: 
                var_t[i] = omega + alpha*(returns[i-1]**2) + beta*var_t[i-1]
        return var_t        
    
    
    def fit(self, returns):
        self.rt = returns
        
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        self.params_garch=np.zeros((self.N,3))
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            self.params_garch[i] = np.array(self.garch_fit(self.rt[:,i].ravel('F')))
            D_t[:,i] = np.sqrt(self.garch_var(self.params_garch[i], self.rt[:,i].ravel('F')))
        self.D_t = D_t
        self.et=self.rt/D_t
#         for i in range(1,self.T):
#             dts = np.diag(D_t[i])
#             dtinv = np.linalg.inv(dts)
#             et[i] = dtinv*self.rt[i].T
        self.R=np.corrcoef(self.et.T)
        
            
        return {'params_garch':self.params_garch, 'R': self.R} 
    
    def predict(self):
        """
        return predict S.E
        """
        T=self.T
        var_pre=np.zeros(self.N)
        for k in range(self.N):
            params=self.params_garch[k]
            omega=params[0]
            alpha=params[1]
            beta=params[2]
            y=self.rt[T-1,k]
            var_pre[k]=omega+alpha*y**2+beta*self.D_t[T-1,k]**2
        return np.sqrt(var_pre)
        
# def GARCH_DGP(eta,a,A,B):

#     T=len(eta)
#     y=np.zeros(eta.shape)
#     var=np.zeros(T)
#     for t in range(T):
#         if t==0:
#             var[t]=a
#         else:
#             var[t]=a+A*y[t-1]**2+B*var[t-1]
            
#         y[t]=np.sqrt(var[t])*eta[t]
#     return y
# def MGARCH_DGP(meta,params):

#     T,N=np.shape(meta)
#     y=np.zeros(meta.shape)
#     for k in range(N):
#         a,A,B=params[k]
#         y[:,k]=GARCH_DGP(meta[:,k],a,A,B)
#     return y