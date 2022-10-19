import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import stats


class VDAR(object):
    """
    self-weighted Gaussian QMLE method for VLDAR model
    """

    def __init__(self, y):
        self.y = y

    def K(self, N):
        '''
        d_vec'(gamma)/d_(sigama)
        innovation项关于sigama求导
        '''
        k = np.zeros((0, N*N))
        unit_matrix_m = np.eye(N)
        for i in range(N-1):
            zeros = np.zeros((N-i-1, N*i))
            unit = np.eye(N-i-1)
            left = unit_matrix_m[i+1:]
            r = np.zeros(N)
            r[i] = 1
            right = np.kron(unit, r)

            k_ = np.hstack((zeros, left, right))
            k = np.append(k, k_, axis=0)
        return(k)

    
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
 
    
    def loss_func_alpha(self, lam_var,lam_loc, p, q,r_m=0):
        """
        带入数据y和初始参数，算出似然函数
        ------------------------------------
        param y:(n \times m) - data
        param m:scalar - dimension
        param lam_loc:(m*m*p)list or array - 条件均值部分参数
        param lam_var:(m*m*q)list or array - 条件方差部分参数
        param lam_cov:(m*(m-1)/2)list or array - innovation 项非对角线元素
        param size:scalar - 样本量
        """
        r_m=0
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)
#         lam_loc = lam[:N*N*p]
#         lam_var = lam[N*N*p:N*N*(p+q)+N]
        weight = self.weight(p, q, r_m)
        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q
        likelihood=0
        for t in range(r,T):
            wt = weight[t-r]
            wt=1
            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, (y[t-q:t][::-1]**2).ravel('C'))
            epsilon=y[t]-lam_loc_mat@y_mean
 
            h=lam_var_mat@y_var
            likelihood += -0.5*wt*np.sum(np.log(h))-0.5 *wt *np.sum(epsilon**2/h)
        return -likelihood/(T-r)
    
    def loss_func(self, lam, p, q,r_m=0):
        """
        带入数据y和初始参数，算出似然函数
        ------------------------------------
        param y:(n \times m) - data
        param m:scalar - dimension
        param lam_loc:(m*m*p)list or array - 条件均值部分参数
        param lam_var:(m*m*q)list or array - 条件方差部分参数
        param lam_cov:(m*(m-1)/2)list or array - innovation 项非对角线元素
        param size:scalar - 样本量
        """
        r_m=0
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)
        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        
        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q
        likelihood=0
        for t in range(r,T):

            wt=1
            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, (y[t-q:t][::-1]**2).ravel('C'))
            epsilon=y[t]-lam_loc_mat@y_mean
 
            h=lam_var_mat@y_var
            likelihood += -0.5*wt*np.sum(np.log(h))-0.5 *wt *np.sum(epsilon**2/h)
        return -likelihood/(T-r)
    
    
    def fit_package(self,p,q,init_value=None):
        
        self.p = p
        self.q = q
        T, N = np.shape(y)
        if np.sum(init_value == None):
            lam_loc = np.zeros(N*N*p).ravel('F')
            lam_var = np.ones(N)
            for i in range(q):
                lam_var = np.concatenate((lam_var, (np.full((N,N), 0.1)).ravel('F')))
            lam = np.concatenate((lam_loc, lam_var))
        else:
            lam = init_value
        from scipy.optimize import SR1
        from scipy.optimize import minimize
        from scipy.optimize import Bounds
        bounds = Bounds([-np.inf]*N*N*p+[0.1**(N**2)]*(N*N*q+N), 
                        [np.inf]*N*N*p+[10]*(N)+[1]*(N*N*q))
        res = minimize(self.loss_func, lam,args=(p,q), bounds=bounds)
        return res
        
    def direct_alpha(self, lam_var, lam_loc,  p, q, r_m=0):
        '''
        jac function
        '''
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q

        cov_mat = np.eye(N)
        y_trim = y[r:].T  # N*(T-r) matrix
        y_sq = y**2  # T*N
        y_sq_trim = y_sq[r:].T  # N*(T-r) matrix

        weight = self.weight(p, q, r_m)

        Z = np.zeros((N*p, T-r))
        Z_sq = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_sq[:, i] = np.append(1, (y_sq[r+i-q:r+i][::-1]).ravel('C'))

        
        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_sq
        epsilon_div_H=epsilon_trim/H_trim
        jac_phi=0
        jac_alpha=0
        hess_phi_phi=0
        hess_phi_alpha=0
        hess_alpha_alpha=0
        for t in range(r,T):
            wt = weight[t-r]

            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, (y_sq[t-q:t][::-1]).ravel('C'))
            epsilon=y[t]-lam_loc_mat@y_mean
 
            h=lam_var_mat@y_var
#             jac_phi+=wt*np.outer(epsilon/h,y_mean).ravel('F')
            jac_alpha-=0.5*wt*np.outer(1/h-epsilon**2/h**2,y_var).ravel('F')
#             hess_phi_phi -=wt* np.kron(np.outer(y_mean, y_mean),
#                                      np.diag(1/h))
#             hess_phi_alpha -= wt* np.kron(np.outer(y_mean, y_var),
#                                          np.diag(epsilon/(h**2) ))
            
            hess_alpha_alpha += wt* np.kron(np.outer(y_var, y_var),
                                      np.diag(0.5*1/(h**2)-(epsilon**2)/(h**3)) )
#         jac_phi = (epsilon_div_H@np.diag(weight)@Z.T).ravel('F')
        
#         jac_alpha = -0.5*((1/H_trim-(epsilon_trim**2)/(H_trim**2))@np.diag(weight)@Z_sq.T).ravel('F')
#         print(jac_phi_1-jac_phi)
#         print(jac_alpha-jac_alpha_1)
#         f_d = -np.concatenate((jac_phi, jac_alpha))

#         for t in range(T-r):
#             wt=weight[t]
#             hess_phi_phi -=wt* np.kron(np.outer(Z[:, t], Z[:, t]),
#                                      np.diag(1/H_trim[:, t]))
#             hess_phi_alpha -= wt* np.kron(np.outer(Z[:, t], Z_sq[:, t]),
#                                          np.diag(epsilon_trim[:,t]/(H_trim[:,t]**2) ))
            
#             hess_alpha_alpha += wt* np.kron(np.outer(Z_sq[:, t], Z_sq[:, t]),
#                                       np.diag(0.5*1/(H_trim[:,t]**2)-(epsilon_trim[:,t]**2)/(H_trim[:,t]**3)) )
#         hess = -np.block([[hess_phi_phi, hess_phi_alpha],
#                       [hess_phi_alpha.T, hess_alpha_alpha]])

        return -jac_alpha/(T-r), -hess_alpha_alpha/(T-r)

    def bound_param(self, param, lower, upper):
        """
        将所有参数限制在lower和upper里面
        """
        n = len(param)
        lower=np.array(lower)
        upper=np.array(upper)
#         lower = lower*np.ones(n)
#         upper = upper*np.ones(n)
        param = np.maximum(lower, param)
        param = np.minimum(upper, param)
        return param

    def fit(self, p, q, step_select=3, max_iter=10, max_iter_var=3, var_tol=1e-2, total_tol=1e-2, r_m=0, init_value=None, result_show=False):
        """
        step_select: lam_var参数迭代中步长选取
                    0:固定步长1 ;1: Arimijo;2:BB 3:黄金分割
        max_iter:整体迭代最大次数
        total_tol：全部参数的toleran，定义为参数变化的比率小于一定值
        max_iter_var:lam_var参数迭代最大次数
        var_tol:lam_var参数的toleran，定义为参数变化的比率小于一定值
       init_value: 0表示采用对角元素符合平稳条件。 或者自定义一个init_value
        r_m=0:表示根据p，q的值来判断是否使用selfweight。若r_m=-1: 表示unweighted method;若r_m！=0，则根据r_m的值来表示weight滞后阶数
        result_show:是否要print结果，默认不print
        """
        self.p = p
        self.q = q

        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)
        if np.sum(init_value == None):
            lam_loc = np.zeros(N*N*p).ravel('F')
            lam_var = np.ones(N)
            for i in range(q):
                lam_var = np.concatenate((lam_var, (np.full((N,N), 0.1)).ravel('F')))
            lam = np.concatenate((lam_loc, lam_var))
        else:
            lam = init_value

        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]


        lam_loc_mat = lam_loc.reshape((N, N*p), order='F')  # 将条件均值部分参数矩阵 N \times N*p
        # 将条件方差部分参数矩阵 N \times N*q+1
        lam_var_mat = lam_var.reshape((N, 1+N*q), order='F')


        weight = self.weight(p, q, r_m)
        half_weight = np.sqrt(weight)

        # 用来记录不同部分的前后次迭代的变化大小
        all_diff = pd.DataFrame(
            columns=['loc_ratio', 'var_ratio',  'var_direct', 'lam_ratio'])

        for i in range(max_iter):
            lam_0 = lam.copy()

            # 更新mean

            part1 = 0
            part2 = 0
            for t in range(r, T):

                wt = weight[t-r]
                y_mean = y[t-p:t][::-1].ravel('C')
                y_var = np.append(1, (y[t-q:t][::-1]**2).ravel('C'))
                cond_mean = lam_loc_mat@y_mean
                h_var = lam_var_mat@y_var

                part1 += wt*np.kron(np.outer(y_mean, y_mean), np.diag(1/h_var))
                part2 += wt*np.kron(y_mean, (1/h_var)*y[t])

            lam[:N*N*p] = np.linalg.inv(part1)@part2
           


            # 更新var

            lower = np.array([0.1**(N**3)]*N+[0.1**(N**3)]*(N*N*q))
            upper = np.array([10]*N+[1]*(N*N*q))
            for j in range(max_iter_var):

                lam_var_before = lam[N*N*p:N*N*(p+q)+N].copy()
                jac, hess = self.direct_alpha(lam_var, lam_loc,  p, q, r_m)
                
                direct = np.linalg.inv(hess)@jac

                f=self.loss_func_alpha(self.bound_param(lam_var, lower, upper), lam_loc, p, q, r_m)
                # 黄金分割法
                a = -2
                b = 2
                c = a+0.382*(b-a)
                d = a+0.618*(b-a)
                for k in range(10**(N)):
                    fc = self.loss_func_alpha(self.bound_param(
                        lam_var-c*direct, lower, upper), lam_loc, p, q, r_m)
                    fd = self.loss_func_alpha(self.bound_param(
                        lam_var-d*direct, lower, upper), lam_loc, p, q, r_m)

                    if fc < fd:
                        a = a
                        b = d
                        d = c
                        c = a+0.382*(b-a)
                    else:
                        b = b
                        a = c
                        c = d
                        d = a+0.618*(b-a)
                    if b-a < 0.1**(N*2):
                        stepsize = (a+b)/2
                        break
                stepsize = (a+b)/2

                lam_var_after = self.bound_param(lam_var-stepsize*direct, lower, upper)

                f_after = self.loss_func_alpha(lam_var_after, lam_loc, p, q, r_m)
                if f_after < f:
                    lam[N*N*p:N*N*(p+q)+N] = lam_var_after
                else:
                    stepsize = 0
                    lam[N*N*p:N*N*(p+q)+N] = lam_var_before

                print("stepsize:{}".format(stepsize))
                if i>=1 or j>=1:
                    var_diff_ratio_inner = np.linalg.norm((lam[N*N*p:N*N*(p+q)+N]-lam_var_before)/lam_var_before, ord=np.inf)
                    print(var_diff_ratio_inner)
                    if var_diff_ratio_inner < var_tol:
                        break


            if i >=1:
            
                loc_diff_ratio = np.linalg.norm((lam[:N*N*p]-lam_0[:N*N*p])/lam_0[:N*N*p])  # 记录loc前后的变化
                var_diff_ratio = np.linalg.norm((lam[N*N*p:N*N*(p+q)+N]-lam_0[N*N*p:N*N*(p+q)+N])/lam_0[N*N*p:N*N*(p+q)+N], ord=np.inf)
                var_direct = np.linalg.norm(direct, ord=np.inf)  # 取梯度绝对值的最大值
                lam_ratio = np.linalg.norm((lam-lam_0)/lam_0, ord=np.inf)

                all_diff.loc[i] = [loc_diff_ratio, var_diff_ratio, var_direct, lam_ratio]

                if lam_ratio < total_tol:
                    if result_show == True:
                        print(all_diff)
                        print("===================================================")
                        print("break for reach tol")
                        print("detial each difference:\n{}".format(all_diff))
                        print("loc:\n{} \n var:\n{} ".format(
                            lam_loc_mat, lam_var_mat))
                        print("===================================================")
                    self.lam = lam

                    return lam
        if result_show == True:
            print(all_diff)
            print("===================================================")
            print("break for reach max_iter")
            print("detial each difference:\n{}".format(all_diff))
            print("loc:\n{} \n var:\n{} ".format(
                lam_loc_mat, lam_var_mat))
            print("===================================================")
        self.lam = lam
        return lam

#

   

    
    def y_pre(self):

        p = self.p
        q = self.q
        r = max(p, q)
        y = self.y
        T, N = np.shape(y)
        lam = self.lam
        d = (p+q)*N*N+N+N*(N-1)//2
        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 N \times N*p
        # 将条件方差部分参数矩阵 N \times N*q+1
        lam_var_mat = lam_var.reshape((N, 1+N*q), order='F')

        cov_mat = np.eye(N)

        y_trim = y[r:].T  # N*(T-r) matrix
        y_sq = y**2  # T*N
        y_sq_trim = y_sq[r:].T  # N*(T-r) matrix

        Z = np.zeros((N*p, T-r))
        Z_sq = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_sq[:, i] = np.append(1, (y_sq[r+i-q:r+i][::-1]).ravel('C'))

        y_pre = lam_loc_mat@((y[T-p:T][::-1]).ravel('C'))
        h_pre = lam_var_mat @ ( np.append(1, (y_sq[T-q:T][::-1]).ravel('C')))
        var_pre=np.diag(h_pre)
        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_sq
        var_ios = np.zeros((T-r, N, N))
        for i in range(T-r):
            var_ios[i] = np.diag(H_trim[:,i])

        eta_trim = epsilon_trim/np.sqrt(H_trim)
        result = {"y_pre": y_pre,
              "var_pre": var_pre,
              "var_ios":var_ios,
              "epsilon_ios": epsilon_trim.T,
              "eta_ios": eta_trim.T}
        return pd.Series(result)