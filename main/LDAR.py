import numpy as np
import pandas as pd
import time
from scipy import stats
from scipy import optimize as opt
from tqdm import tqdm




class LDAR(object):
    """
    self-weighted Gaussian QMLE method for LDAR model
    """
    def __init__(self, y):
        self.y=y # T*N matrix

    
    def weight(self, p, q, r_m=0):
        '''
        根据p 计算weight N-p
        r_m=0:表示根据p，q的值来判断是否使用selfweight。若r_m=-1,则unweighted；若r_m！=0，则根据r_m的值来表示weight滞后阶数

        '''

        y = self.y

        if r_m == 0:
            r = max(p, q)
            if p > q:
                a = np.linalg.norm(y, ord=2, axis=1)
                y_95_quan = np.quantile(a, 0.95)
                a[a < y_95_quan] = 0
                a_t = np.convolve(a, np.ones(p), 'valid')
                weight = np.zeros(np.shape(a_t))
                weight[a_t == 0] = 1
                weight[a_t != 0] = y_95_quan**2*a_t[a_t != 0]**(-2)  # T-p+1
            # weight=np.concatenate((np.zeros(p-1),weight))
            else:
                weight = np.ones(np.shape(y)[0]-p+1)
        elif r_m == -1:
            r = max(p, q)
            weight = np.ones(np.shape(y)[0])
        else:
            p = r_m
            r = r_m
            a = np.linalg.norm(y, ord=2, axis=1)
            y_95_quan = np.quantile(a, 0.95)
            a[a < y_95_quan] = 0
            a_t = np.convolve(a, np.ones(p), 'valid')
            weight = np.zeros(np.shape(a_t))
            weight[a_t == 0] = 1
            weight[a_t != 0] = y_95_quan**2*a_t[a_t != 0]**(-2)  # T-p+1
        return weight[r-p:-1]  # 去掉最后一个和前r-p个，dim=T-r
    
    def likelihood(self,lam,p,q,r_m):
        """
        return likelihood
        """
        y=self.y
        T=y.size
        
        lam_loc=lam[:p]
        lam_var=lam[p:]
        

        Y=y.T  #Y is N*T matrix

        y_abs=np.abs(y)
        if r_m==0:
            r=max(p,q)
        else:
            r=r_m
        y_trim = y[r:].T # N*(T-r) matrix
        y_abs=np.abs(y) # T*N
        y_abs_trim = y_abs[r:].T # N*(T-r) matrix
            
        weight=self.weight(p,q,r_m)
        
        Z=np.zeros((p,T-r))
        Z_abs=np.zeros((q+1,T-r))
        for i in range(T-r):
            Z[:,i]=(y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:,i]=np.append(1,(y_abs[r+i-q:r+i][::-1]).ravel('C'))
            
        epsilon_trim = y_trim-lam_loc@Z
        H_trim=lam_var@ Z_abs
        
        eta_trim=epsilon_trim / H_trim
        weight_eta_trim=eta_trim@np.diag(np.sqrt(weight))

        
        likelihood=-np.sum(weight*np.log(H_trim))-0.5*np.sum(weight*eta_trim*eta_trim)
        return -likelihood
    
    
    
    
    def direct(self,lam,p,q,r_m=0):
        """
        direct=jac@hess
        """
        if r_m==0:
            r=max(p,q)
        else:
            r=r_m
        y=self.y
        T=y.size
        
        lam_loc=lam[:p]
        lam_var=lam[p:]
        
        y_trim = y[r:].T # N*(T-r) matrix
        y_abs=np.abs(y) # T*N
        y_abs_trim = y_abs[r:].T # N*(T-r) matrix
        weight=self.weight(p,q,r_m)
        
        Z=np.zeros((p,T-r))
        Z_abs=np.zeros((q+1,T-r))
        for i in range(T-r):
            Z[:,i]=(y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:,i]=np.append(1,(y_abs[r+i-q:r+i][::-1]).ravel('C'))
            
        epsilon_trim = y_trim-lam_loc@Z
        H_trim=lam_var@ Z_abs

         
        jac_phi=0
        jac_alpha=0
        s_d_loc=0
        s_d_loc_var=0
        s_d_var=0
        for t in range(T-r):
            wt=weight[t]
            y_mean=Z[:,t].ravel('C')
            y_var=Z_abs[:,t].ravel('C')
            h_var=H_trim[t]
            epsilon=epsilon_trim[t]
            jac_phi-=wt*y_mean*(epsilon)/h_var**2
            jac_alpha+=wt*y_var/h_var*(1-epsilon**2/h_var**2)
            #s_d_loc
            s_d_loc-=wt*np.outer(y_mean,y_mean)/(h_var**2)
            #s_d_loc_var
            s_d_loc_var-=wt*np.outer(y_mean,y_var)*epsilon/(h_var**3)
            # s_d_var
            s_d_var+=wt*np.outer(y_var,y_var)/(h_var**2)*(1-3*epsilon**2/(h_var**2))
        jac=np.concatenate((jac_phi,jac_alpha))
        hess=-np.vstack((np.hstack((s_d_loc,s_d_loc_var)),
                      np.hstack((s_d_loc_var.T,s_d_var))))
            
        return jac,hess
    def bound_param(self,param,lower,upper):
        """
        将所有参数限制在lower和upper里面
        """
        n=len(param)
        lower=lower*np.ones(n)
        upper=upper*np.ones(n)
        param=np.maximum(lower,param)
        param=np.minimum(upper,param)
        return param
    
    def fit(self,p,q,max_iter=10,total_tol=1e-3,r_m=0,result_show = True):
        """
        step_select: lam_var参数迭代中步长选取
                    0:固定步长1 ;1: Arimijo;2:BB 3:黄金分割
        max_iter:整体迭代最大次数
        total_tol：全部参数的toleran，定义为参数变化的比率小于一定值
        max_iter_var:lam_var参数迭代最大次数
        var_tol:lam_var参数的toleran，定义为参数变化的比率小于一定值
        r_m=0:表示根据p，q的值来判断是否使用selfweight。若r_m！=0，则根据r_m的值来表示weight滞后阶数
        """
        self.p=p
        self.q=q
        if r_m==0:
            r=max(p,q)
        else:
            r=r_m
        y=self.y
        T=y.size
        
        lam=np.concatenate((np.zeros(p),0.1*np.ones(q+1)))
        

        lam_loc=lam[:p]
        lam_var=lam[p:]


        weight=self.weight(p,q,r_m)
        half_weight=np.sqrt(weight)
        
        all_diff=pd.DataFrame(columns=['jac_diff','lam_tol']) ## 用来记录不同部分的前后次迭代的变化大小

        for i in range(max_iter):
            lam_0=lam.copy()
            lam_before_norm=np.linalg.norm(lam_0,ord=np.inf)
 
    
            lower=np.array([-2]*(p)+[0.001]*(q+1))
            upper=np.array([2]*p+[10]+[2]*(q))

            jac,hess=self.direct(lam,p,q,r_m)

            direct=np.linalg.inv(hess)@jac

            a=-4;b=4;c=a+0.382*(b-a);d=a+0.618*(b-a)
            for k in range(10):

                fc=self.likelihood(self.bound_param(lam-c*direct,lower,upper),p,q,r_m)
                fd=self.likelihood(self.bound_param(lam-d*direct,lower,upper),p,q,r_m)
                if fc<fd:
                    a=a;b=d;d=c;c=a+0.382*(b-a)
                else:
                    b=b;a=c;c=d;d=a+0.618*(b-a)
                if b-a<0.1:
                    stepsize=(a+b)/2
                    break
            stepsize=(a+b)/2

            lam=self.bound_param(lam-stepsize*direct,lower,upper)
            print("stepsize:{}".format(stepsize))

            jac_diff=np.linalg.norm(jac,ord=np.inf)
            
            tol_lam=np.linalg.norm(lam-lam_0,ord=np.inf)
            
            
            all_diff.loc[i]=[jac_diff,tol_lam]
            
            if jac_diff < total_tol:
                if result_show == True:
                    print("===================================================")
                    print("break for reach tol")
                    print("detial each difference:\n{}".format(all_diff))
                    print("loc:\n{} \n var:\n{} ".format(lam[:p],lam[p:]))
                    print("===================================================")
                self.lam=lam
                return lam
            
        if result_show == True:
            print("===================================================")
            print("break for reach tol")
            print("detial each difference:\n{}".format(all_diff))
            print("loc:\n{} \n var:\n{} ".format(lam[:p],lam[p:]))
            print("===================================================")
        self.lam=lam
        return lam
        
    def y_pre(self):
        
        p=self.p
        q=self.q
        r=max(p,q)
        y=self.y
        T=y.size        
        lam=self.lam
        d=p+q+1
        lam_loc=lam[:p]
        lam_var=lam[p:]
        
        y_trim = y[r:].T # N*(T-r) matrix
        y_abs=np.abs(y) # T*N
        y_abs_trim = y_abs[r:].T # N*(T-r) matrix
            
        
        Z=np.zeros((p,T-r))
        Z_abs=np.zeros((q+1,T-r))
        for i in range(T-r):
            Z[:,i]=(y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:,i]=np.append(1,(y_abs[r+i-q:r+i][::-1]).ravel('C'))
        
        y_pre=lam_loc@((y[T-p:T][::-1]).ravel('C'))
        h_pre=lam_var@(np.append(1,(y_abs[T-q:T][::-1]).ravel('C')))
        epsilon_trim = y_trim-lam_loc@Z
        H_trim=lam_var@ Z_abs

        eta_trim=epsilon_trim/H_trim
        
        
        result={"y_pre":y_pre,
               "h_pre":h_pre,
               "H_trim":H_trim,
               "epsilon_trim":epsilon_trim,
               "eta_trim":eta_trim}
        return pd.Series(result)
    

    

        
    def Asymptotic_deviation(self):
        '''
        求出理论上的渐近标准差
        Sigma and Omega
        ------------------------
        param fitted_lam:(m*m*(p+q)+m+m*(m-1)//2)list or array
        param y:(n * m) - data
        param p:int - order of condition mean 
        param q:int - order of condition variance 
        '''
#         lam=self.param
        lam=np.array(self.lam) #若是list 转化为array
        y=self.y
        T=y.size
        p=self.p
        q=self.q
        r=max(p,q)
        weight=self.weight(p,q,r_m=0)

        d=1+p+q
        lam_loc=lam[:p]
        lam_var=lam[p:]
        
        y_trim = y[r:].T # N*(T-r) matrix
        y_abs=np.abs(y) # T*N
        y_abs_trim = y_abs[r:].T # N*(T-r) matrix
            
        
        Z=np.zeros((p,T-r))
        Z_abs=np.zeros((q+1,T-r))
        for i in range(T-r):
            Z[:,i]=(y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:,i]=np.append(1,(y_abs[r+i-q:r+i][::-1]).ravel('C'))
        
        y_pre=lam_loc@Z
        epsilon_trim = y_trim-lam_loc@Z
        H_trim=lam_var@ Z_abs
        eta_trim=epsilon_trim/H_trim

        Sigma1=0
        Sigma2=0
        Omega11=0
        Omega12=0
        Omega22=0

        k1=np.mean((eta_trim@np.sqrt(np.diag(weight)))**3)
        k2=np.mean((eta_trim@np.sqrt(np.diag(weight)))**4)-1
        for t in range(T-r):
            wt=weight[t-r]
            y_mean=Z[:,t]
            y_var=Z_abs[:,t]
            h_var=H_trim[t]
            Omega11+=wt*np.outer(y_mean,y_mean)/(h_var**2)
            Omega12+=wt**0.5*k1*np.outer(y_mean,y_var)/(h_var**2)
            Omega22+=k2*np.outer(y_var,y_var)/(h_var**2)
            Sigma1+=wt*np.outer(y_mean,y_mean)/(h_var**2)
            Sigma2+=2*wt*np.outer(y_var,y_var)/(h_var**2)
        Omega=np.vstack((np.hstack((Omega11,Omega12)),
                        np.hstack((Omega12.T,Omega22))))/(T-r)
        Sigma=np.vstack((np.hstack((Sigma1,np.zeros((p,q+1)))),
                        np.hstack((np.zeros((q+1,p)),Sigma2))))/(T-r)
            
        Sigma_inv=np.linalg.inv(Sigma)
        A_D=np.sqrt(np.diagonal(Sigma_inv@Omega@Sigma_inv)/T)
        result=pd.Series(dict(zip(['Omega','Sigma','A_D'],[Omega,Sigma,A_D])))
        return result