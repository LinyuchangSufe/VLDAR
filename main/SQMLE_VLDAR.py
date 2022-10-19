import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import stats


class SQMLE_VLDAR(object):
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
 
    
    def loss_fun(self, lam, p, q, r_m=0):
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
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        y_trim = y[r:].T  # N*(T-r) matrix
        y_abs = np.abs(y)  # T*N
        y_abs_trim = y_abs[r:].T  # N*(T-r) matrix

        weight = self.weight(p, q, r_m)

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs

        eta_trim = epsilon_trim / H_trim
        weight_eta_trim = eta_trim@np.diag(np.sqrt(weight))
        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        likelihood = -np.sum(weight*np.sum(np.log(H_trim), 0))-0.5*np.sum(weight)*np.log(
            cov_mat_det)-0.5*np.trace(cov_mat_inv@(weight_eta_trim@weight_eta_trim.T))
        return -likelihood/(T-r)/N

    def jac_hess(self, lam, p, q, r_m=0):
        '''
        jac function
        '''
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        y_trim = y[r:].T  # N*(T-r) matrix
        y_abs = np.abs(y)  # T*N
        y_abs_trim = y_abs[r:].T  # N*(T-r) matrix

        weight = self.weight(p, q, r_m)

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs

        eta_trim = epsilon_trim / H_trim

        part_phi = 1/H_trim*(cov_mat_inv@eta_trim)@np.diag(weight)
        jac_phi = (part_phi@Z.T).ravel('F')

        part_alpha = (1/H_trim-(eta_trim/H_trim) *
                (cov_mat_inv@eta_trim))@np.diag(weight)
        jac_alpha = -(part_alpha@Z_abs.T).ravel('F')

        part_sigma = (eta_trim@np.diag(weight)@eta_trim.T).ravel('F')
        jac_sigma = -0.5*self.K(N)@(np.sum(weight)*np.kron(np.eye(N), cov_mat_inv)@(np.eye(N).ravel('F'))
                                 - np.kron(cov_mat_inv, cov_mat_inv)@part_sigma)

        f_d = -np.hstack((jac_phi, jac_alpha, jac_sigma))
        hess_phi_phi=0
        hess_phi_alpha=0
        hess_phi_sigma=0
        hess_alpha_alpha=0
        hess_alpha_sigma=0
        hess_sigma_sigma=0

        for t in range(T-r):
            wt=weight[t]
            hess_phi_phi -=wt* np.kron(np.outer(Z[:, t], Z[:, t]),
                                    np.outer(1/H_trim[:, t], 1/H_trim[:, t])*cov_mat_inv)
            hess_phi_alpha += wt* np.kron(np.outer(Z[:, t], Z_abs[:, t]),
                                    np.outer(1/H_trim[:, t], 1/H_trim[:, t]) *
                                    (cov_mat_inv@np.diag(eta_trim[:, t])+np.diag(cov_mat_inv@eta_trim[:, t])))
            hess_phi_sigma += wt* np.kron(Z[:, t].reshape((N*p,1)), np.diag(1/H_trim[:, t])@np.kron((eta_trim[:, t]@
                cov_mat_inv).reshape((1,N)), cov_mat_inv)@((self.K(N)).T))
            hess_alpha_alpha += wt* np.kron(np.outer(Z_abs[:, t], Z_abs[:, t]),
                                      np.outer(1/H_trim[:, t], 1/H_trim[:, t]) *(np.eye(N)\
                                      - np.outer(eta_trim[:, t], eta_trim[:, t])*cov_mat_inv\
                                      - 2*np.diag(eta_trim[:, t]*(cov_mat_inv@eta_trim[:, t]))))
            hess_alpha_sigma -= wt* np.kron(Z_abs[:, t].reshape((N*q+1,1)),
                                     np.diag(1/H_trim[:, t] * eta_trim[:, t]) @ np.kron(cov_mat_inv@eta_trim[:, t], cov_mat_inv)@(self.K(N).T))
            hess_sigma_sigma += wt* 0.5*(self.K(N)@(np.kron(cov_mat_inv, cov_mat_inv)\
                                             - np.kron(cov_mat_inv, cov_mat_inv@np.outer(eta_trim[:, t], eta_trim[:, t])@cov_mat_inv)\
                                             - np.kron(cov_mat_inv@np.outer(eta_trim[:, t], eta_trim[:, t])@cov_mat_inv, cov_mat_inv))@self.K(N).T)

        hess = -np.block([[hess_phi_phi, hess_phi_alpha, hess_phi_sigma],
                      [hess_phi_alpha.T, hess_alpha_alpha, hess_alpha_sigma],
                      [hess_phi_sigma.T, hess_alpha_sigma.T, hess_sigma_sigma]])

        return f_d/(T-r)/N, hess/(T-r)/N

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

    def fit(self, p, q,max_iter=10, total_tol=1e-2, r_m=0, init_value=None, result_show=False):
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
                lam_var = np.concatenate(
                    (lam_var, np.full((N,N), 1/(4*N**2)).ravel('F')))
            lam = np.concatenate((lam_loc, lam_var, np.zeros(N*(N-1)//2)))
        else:
            lam = init_value

        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 N \times N*p
        # 将条件方差部分参数矩阵 N \times N*q+1
        lam_var_mat = lam_var.reshape((N, 1+N*q), order='F')

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)

        weight = self.weight(p, q, r_m)
        half_weight = np.sqrt(weight)

        # 用来记录不同部分的前后次迭代的变化大小
        all_diff = pd.DataFrame(
            columns=['lam_ratio', 'lam_direct'])

        for i in range(max_iter):
            lam_before = lam.copy()
            lower = np.array([-np.inf]*N*N*p+[0.1**(N+1)]
                             * N+[0.1**(N+1)]*(N*N*q)+[-0.999]*(N*(N-1)//2))
            upper = np.array([np.inf]*N*N*p+[10]*N+[1]*(N*N*q)+[0.999]*(N*(N-1)//2))


            
            jac, hess = self.jac_hess(lam, p, q, r_m)

            direct = np.linalg.inv(hess)@jac
            f = self.loss_fun(lam, p, q, r_m)  # 记录此可的loss

            # 黄金分割法
            a = -2
            b = 2
            c = a+0.382*(b-a)
            d = a+0.618*(b-a)

            for k in range(10**(N-1)):
                fc = self.loss_fun(self.bound_param(
                    lam-c*direct, lower, upper), p, q, r_m)
                fd = self.loss_fun(self.bound_param(
                    lam-d*direct, lower, upper), p, q, r_m)

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
                if b-a < 0.1**(N):
                    stepsize = (a+b)/2
                    break
            stepsize = (a+b)/2

            lam_after = self.bound_param(
                lam-stepsize*direct, lower, upper)

            f_after = self.loss_fun(lam_after, p, q, r_m)
            if f_after < f:
                lam = lam_after
            else:
                stepsize = 0
                lam = lam_before
            print("stepsize:{}".format(stepsize))
            if i>=1:
                lam_diff_ratio = np.linalg.norm((lam-lam_before)/lam_before, ord=np.inf)
                direct = np.linalg.norm(direct, ord=np.inf)  # 取梯度绝对值的最大值
                all_diff.loc[i] = [lam_diff_ratio, direct]

                if lam_diff_ratio < total_tol:
                    if result_show == True:
                        print(all_diff)
                        print("===================================================")
                        print("break for reach tol")
                        print("detial each difference:\n{}".format(all_diff))
                        print("loc:\n{} \n var:\n{} \n cov:\n{}".format(lam[:N*N*p].reshape((N, N*p), order='F'),\
                                                                        lam[N*N*p:N*N*p+N+N*N*q].reshape((N, N*q+1), order='F'),\
                                                                        lam[-N*(N-1)//2:]))
                        print("===================================================")
                    self.lam = lam

                    return lam
        if result_show == True:
            print(all_diff)
            print("===================================================")
            print("break for reach max_iter")
            print("detial each difference:\n{}".format(all_diff))
            print("loc:\n{} \n var:\n{} \n cov:\n{}".format(lam[:N*N*p].reshape((N, N*p), order='F'),\
                                                                    lam[N*N*p:N*N*p+N+N*N*q].reshape((N, N*q+1), order='F'),\
                                                                    lam[-N*(N-1)//2:]))
            print("===================================================")
        self.lam=lam
        return lam

#

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
        lam = np.array(self.lam)  # 若是list 转化为array
        y = self.y
        T, N = np.shape(y)
        p = self.p
        q = self.q
        eta = self.y_pre()['eta_trim'].T
        weight = self.weight(p, q, r_m=0)

        d = (p+q)*N*N+N+N*(N-1)//2
        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 N \times N*p
        # 将条件方差部分参数矩阵 N \times N*q+1
        lam_var_mat = lam_var.reshape((N, 1+N*q), order='F')

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        d_D_T_d_h = np.zeros((N, N*N))  # 用于dV_dtheta 求导

        for i in range(N):
            d_D_T_d_h[i, i*N+i] = 1

        Sigma = 0
        Omega = 0
        d_V_d_theta_T = np.zeros((N*N, d))

        r = max(p, q)
        eta_vec_T_eta_eta = 0
        vec_T_eta_eta_vec_eta_eta = 0
        for t in range(T-r):

            eta_vec_T_eta_eta += np.outer(eta[t],np.outer(eta[t], eta[t]).ravel('F'))/(T-r)
            vec_T_eta_eta_vec_eta_eta += np.outer(np.outer(eta[t], eta[t]).ravel(
                'F'), np.outer(eta[t], eta[t]).ravel('F'))/(T-r)

        for t in range(r, T):
            wt = weight[t-r]

            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, np.abs(y[t-q:t][::-1]).ravel('C'))

            cond_mean = lam_loc_mat@y_mean
            epsilon = y[t]-cond_mean
            h_var = lam_var_mat@y_var
            D = np.diag(h_var)
            D_inv = np.diag(1/h_var)
            V_inv = D_inv@cov_mat_inv@D_inv

            d_epsilon_d_phi_T = -np.kron(y_mean, np.eye(N))
            d_epsilon_d_theta_T = np.hstack((d_epsilon_d_phi_T, np.zeros((N, d-N*N*p))))

            d_h_d_alpha_T = np.kron(y_var, np.eye(N))
            d_h_d_theta_T = np.hstack((np.zeros((N, N*N*p)), d_h_d_alpha_T, np.zeros((N, N*(N-1)//2))))
            d_V_d_theta_T[:, N*N*p:N*N*(p+q)+N] = (np.kron(D@cov_mat, np.eye(N))+np.kron(np.eye(N), D@cov_mat))@d_D_T_d_h.T@d_h_d_alpha_T
            d_V_d_theta_T[:, N*N*(p+q)+N:] = np.kron(D, D)@self.K(N).T

            Sigma += wt*((0.5*d_V_d_theta_T.T@np.kron(V_inv, V_inv)@d_V_d_theta_T
                         + d_epsilon_d_theta_T.T@V_inv@d_epsilon_d_theta_T))/(T-r)

            part_Omega_1 = D_inv@cov_mat_inv@eta_vec_T_eta_eta@np.kron(cov_mat_inv@D_inv, cov_mat_inv@D_inv)
            part_Omega_2 = np.kron(D_inv@cov_mat_inv, D_inv@cov_mat_inv)@vec_T_eta_eta_vec_eta_eta@np.kron(cov_mat_inv@D_inv, cov_mat_inv@D_inv)
            Omega += wt**2 * (d_epsilon_d_theta_T.T@V_inv@d_epsilon_d_theta_T\
                              - 0.5*d_epsilon_d_theta_T.T@part_Omega_1@d_V_d_theta_T\
                              - 0.5*d_V_d_theta_T.T@part_Omega_1.T@d_epsilon_d_theta_T\
                              + 0.25*d_V_d_theta_T.T@(part_Omega_2-np.outer(V_inv.ravel('F'), V_inv.ravel('F')))@d_V_d_theta_T)/(T-r)
        Sigma_inv = np.linalg.inv(Sigma)
        A_D = np.sqrt(np.diagonal(Sigma_inv@Omega@Sigma_inv/T))
        result = pd.Series(
            dict(zip(['Omega', 'Sigma', 'A_D'], [Omega, Sigma, A_D])))
        return result

    def cov(self, df, mean_df, lag, M):
        T, N= np.shape(df)
        cov= 0
        for t in range(M, T):
            cov += np.outer(df[t]-mean_df, df[t-lag]-mean_df)
        return cov/(T-M)

    def portmanteau_test(self, M =6):
        '''
        portmanteau test
        input:
        M:max lag order
        ------------------
        output:
        stat_mean,check mean.
        stat_var_abs,check var by abs
        stat_var_sq,check var by sq
        stat_mix_abs,check mix
        stat_mix_sq,check mix
        '''
        p= self.p
        q= self.q
        y= self.y
        weight= self.weight(p, q)
        T, N= np.shape(y)

        lam= self.lam

        lam_loc= lam[:N*N*p]
        lam_var= lam[N*N*p:N*N*(p+q)+N]
        lam_cov= lam[N*N*(p+q)+N:]

        lam_loc_mat= lam_loc.reshape(
            (N, N*p), order ='F')  # 将条件均值部分参数矩阵 N \times N*p
        # 将条件方差部分参数矩阵 N \times N*q+1
        lam_var_mat = lam_var.reshape((N, 1+N*q), order ='F')

        cov_mat= np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)]= lam_cov
        cov_mat= cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv= np.linalg.inv(cov_mat)
        cov_mat_det= np.linalg.det(cov_mat)

        y_abs= np.abs(y)
        r= max(p, q)
        y_trim= y[r:].T  # N*(T-r) matrix
        y_abs= np.abs(y)  # T*N
        y_abs_trim= np.abs(y_trim)  # N*(T-r) matrix

        Z= np.zeros((N*p, T-r))
        Z_abs= np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i]= (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i]= np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim= y_trim-lam_loc_mat@Z
        H_trim= lam_var_mat @ Z_abs

        eta= (epsilon_trim / H_trim).T
        sgn_eta= np.sign(eta)
        zeta= np.diag(weight)@eta
        zeta_abs= np.abs(zeta)

        mean_eta_abs= np.mean(np.abs(eta), 0)
        mean_zeta= np.mean(zeta, 0)
        mean_zeta_abs= np.mean(zeta_abs, 0)
        mean_zeta_sq= np.mean(zeta**2, 0)
        mean_sgn_eta = np.mean(sgn_eta, 0)

        # 算规范化的矩阵
        C_0 = self.cov(zeta, mean_zeta, 0, M)

        C_abs0 = self.cov(zeta_abs, mean_zeta_abs, 0, M)
        C_sq0 = self.cov(zeta**2, mean_zeta_sq, 0, M)
#         for t in range(T-r):
#             C_0+=np.outer((zeta[t]-mean_zeta),(zeta[t]-mean_zeta))/(T-r)
#             C_abs0+=np.outer(zeta_abs[t]-mean_zeta_abs,zeta_abs[t]-mean_zeta_abs)/(T-r)
#             C_sq0+=np.outer(zeta[t]**2-mean_zeta_sq,zeta[t]**2-mean_zeta_sq)/(T-r)
        C_d0_half_inv= np.diag(1/np.sqrt(np.diagonal(C_0)))
        C_dabs0_half_inv= np.diag(1/np.sqrt(np.diagonal(C_abs0)))
        C_dsq0_half_inv= np.diag(1/np.sqrt(np.diagonal(C_sq0)))

        # 计算样本的相关系数拉之后的向量
        R_mean= np.zeros((N, N*M))
        R_var_abs= np.zeros((N, N*M))
        R_var_sq= np.zeros((N, N*M))
        for k in range(M):
            R_mean[:, k*N:(k+1)*N]=C_d0_half_inv@self.cov(zeta,
                                                            mean_zeta, k+1, M)@C_d0_half_inv
            R_var_abs[:, k*N:(k+1)*N]=  C_dabs0_half_inv@self.cov(zeta_abs,
                                                                  mean_zeta_abs, k+1, M)@C_dabs0_half_inv
            R_var_sq[:, k*N:(k+1)*N] =  C_dsq0_half_inv@self.cov(zeta ** 2, mean_zeta_sq, k+1, M)@C_dsq0_half_inv
        r_mean= R_mean.ravel('F')
        r_var_abs= R_var_abs.ravel('F')
        r_var_sq= R_var_sq.ravel('F')

        # 计算相关系数拉之后的向量的方差
        d= (p+q)*N*N+N
        U_mean= np.zeros((N*N*M, d))
        U_var_abs= np.zeros((N*N*M, d))
        U_var_sq= np.zeros((N*N*M, d))

        G_mean= 0
        G_var_abs= 0
        G_var_sq= 0
        G_mix_abs= 0
        G_mix_sq= 0
        for t in range(M, T-r):
            part_1= (np.diag(1/H_trim[:, t])@cov_mat_inv @ np.outer(eta[t], Z[:, t])).ravel('F')
            part_2 = -(np.outer(1/H_trim[:, t]-eta[t]/H_trim[:, t]* (cov_mat_inv@eta[t]), Z_abs[:, t])).ravel('F')
            d_lt_d_lambda = np.concatenate((part_1, part_2))

            v_mean = np.zeros(N*N*M)
            v_var_abs = np.zeros(N*N*M)
            v_var_sq = np.zeros(N*N*M)
            for k in range(M):
                # r_mean

                v_mean[k*N*N:(k+1)*N*N] = np.outer(zeta[t],zeta[t-k-1]).ravel('F')

#                 U_1=-weight[t]*np.kron(np.outer(zeta[t-k-1].reshape((N,1)),np.kron(np.concatenate((Z[:,t].ravel('F'),np.zeros(N*q+1))).reshape((1,N*(p+q)+1)),np.diag(1/H_trim[:,t])))
                U_mean[k*N*N:(k+1)*N*N] -= (weight[t]*np.kron(np.outer(zeta[t-k-1], np.concatenate(
                    (Z[:, t].ravel('F'), np.zeros(N*q+1)))), np.diag(1/H_trim[:, t])))/(T-r-M)

                # r_var_abs
                v_var_abs[k*N*N:(k+1)*N*N] = np.outer(zeta_abs[t] -\
                                                      mean_zeta_abs, zeta_abs[t-k-1]-mean_zeta_abs).ravel('F')

                U_var_abs[k*N*N:(k+1)*N*N] -= weight[t]*(np.kron(np.outer((zeta_abs[t-k-1]-mean_zeta_abs), np.concatenate((Z[:, t].ravel('F'), np.zeros(N*q+1)))), np.diag(mean_sgn_eta/H_trim[:, t])) \
                                                         + np.kron(np.outer((zeta_abs[t-k-1]-mean_zeta_abs), np.concatenate((np.zeros(N*p), Z_abs[:, t]))), np.diag(mean_eta_abs/H_trim[:, t])))/(T-r-M)

                # r_var_sq
                v_var_sq[k*N*N:(k+1)*N*N] = np.outer(zeta[t]**2 -\
                                                     np.ones(N), zeta[t-k-1]**2-np.ones(N)).ravel('F')
                U_var_sq[k*N*N:(k+1)*N*N] += (-2*np.kron(np.outer(zeta[t-k-1]**2-np.ones(N), np.concatenate((np.zeros(N*p), Z_abs[:, t].ravel('F')))), \
                                                         np.diag(weight[t]/H_trim[:, t])))/(T-r-M)
            v_mix_abs = np.concatenate((v_mean, v_var_abs, weight[t]*d_lt_d_lambda))
            G_mix_abs += np.outer(v_mix_abs, v_mix_abs)/(T-r-M)

            v_mix_sq = np.concatenate((v_mean, v_var_sq, weight[t]*d_lt_d_lambda))
            G_mix_sq += np.outer(v_mix_sq, v_mix_sq) / (T-r-M)

            v_mean_ = np.concatenate((v_mean, weight[t]*d_lt_d_lambda))
            G_mean += np.outer(v_mean_, v_mean_)/(T-r-M)

            v_var_abs_ = np.concatenate((v_var_abs, weight[t]*d_lt_d_lambda))
            G_var_abs += np.outer(v_var_abs_, v_var_abs_)/(T-r-M)

            v_var_sq_ = np.concatenate((v_var_sq, weight[t]*d_lt_d_lambda))
            G_var_sq += np.outer(v_var_sq_, v_var_sq_)/(T-r-M)

        Sigma = self.Asymptotic_deviation()['Sigma']
        Sigma_part = Sigma[:d, :d]

        Sigma_inv = np.linalg.inv(Sigma_part)

        E_wt2 = np.mean(weight**2)
#         coe_abs_part=np.diag(1/np.sqrt(E_wt2*(np.ones(N)-(np.diag(np.outer(mean_eta_abs,mean_eta_abs))))))
#         coe_abs=np.kron(np.eye(M),np.kron(C_dabs0_half_inv,C_dabs0_half_inv))
        coe_abs = np.kron(np.eye(M), np.kron(C_dabs0_half_inv, C_dabs0_half_inv))
        coe_sq = np.kron(np.eye(M), np.kron(C_dsq0_half_inv, C_dsq0_half_inv))


#         U_mean=U_mean/(T-r-M)
        H_mean = 1/E_wt2*np.hstack((np.eye(N*N*M), U_mean@Sigma_inv))
        V_mean = H_mean@G_mean@H_mean.T

#         U_var_abs=U_var_abs/(T-r-M)
        H_var_abs = np.hstack((coe_abs, coe_abs@U_var_abs@Sigma_inv))
        V_var_abs = H_var_abs@G_var_abs@H_var_abs.T


#         U_var_sq=U_var_sq/(T-r-M)
        H_var_sq = np.hstack((coe_sq, coe_sq@U_var_sq@Sigma_inv))
        V_var_sq = H_var_sq@G_var_sq@H_var_sq.T

        r_mix_abs = np.concatenate((r_mean, r_var_abs))
        H_mix_abs_left = np.vstack((np.hstack((E_wt2**(-1)*np.eye(N*N*M), np.zeros((N*N*M, N*N*M)))),
                                    np.hstack((np.zeros((N*N*M, N*N*M)), coe_abs))))
        H_mix_abs_right = np.vstack((E_wt2**(-1)*U_mean, coe_abs@U_var_abs))@Sigma_inv
        H_mix_abs = np.hstack((H_mix_abs_left, H_mix_abs_right))
        V_mix_abs = H_mix_abs@G_mix_abs@H_mix_abs.T

        r_mix_sq = np.concatenate((r_mean, r_var_sq))
        H_mix_sq_left = np.vstack((np.hstack((E_wt2**(-1)*np.eye(N*N*M), np.zeros((N*N*M, N*N*M)))),
                                   np.hstack((np.zeros((N*N*M, N*N*M)), coe_sq))))
        H_mix_sq_right = np.vstack((E_wt2**(-1)*U_mean, coe_sq@U_var_sq))@Sigma_inv
        H_mix_sq = np.hstack((H_mix_sq_left, H_mix_sq_right))
        V_mix_sq = H_mix_sq@G_mix_sq@H_mix_sq.T

        stat_mean =(T-r-M)* r_mean@np.linalg.inv(V_mean)@r_mean
        stat_var_abs =(T-r-M)* r_var_abs@np.linalg.inv(V_var_abs)@r_var_abs
        stat_var_sq =(T-r-M)* r_var_sq@np.linalg.inv(V_var_sq)@r_var_sq
        stat_mix_abs =(T-r-M)* r_mix_abs@np.linalg.inv(V_mix_abs)@r_mix_abs
        stat_mix_sq =(T-r-M)* r_mix_sq@np.linalg.inv(V_mix_sq)@r_mix_sq

        stat_name = ['stat_mean', 'stat_var_abs','stat_var_sq', 'stat_mix_abs', 'stat_mix_sq']
        stat = np.array([stat_mean, stat_var_abs, stat_var_sq,
                        stat_mix_abs, stat_mix_sq])
        p_value = 1-np.array([stats.chi2.cdf(stat_mean, N*N*M),
                             stats.chi2.cdf(stat_var_abs, N*N*M),
                             stats.chi2.cdf(stat_var_sq, N*N*M),
                             stats.chi2.cdf(stat_mix_abs, 2*N*N*M),
                             stats.chi2.cdf(stat_mix_sq, 2*N*N*M)])
        values = np.vstack((stat, p_value)).T
#         result={'stat_mean':stat_mean,
#                 'stat_var_abs':stat_var_abs,
#                 'stat_var_sq':stat_var_sq,
#                 'stat_mix_abs':stat_mix_abs,
#                 'stat_mix_sq':stat_mix_sq}
        result_Se = pd.DataFrame(
            values, columns=['stat', 'p_value'], index=stat_name)

        return result_Se
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

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        y_trim = y[r:].T  # N*(T-r) matrix
        y_abs = np.abs(y)  # T*N
        y_abs_trim = y_abs[r:].T  # N*(T-r) matrix

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        y_pre = lam_loc_mat@Z
        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs
        var = np.zeros((T-r, N, N))
        for i in range(T-r):
            var[i] = np.outer(H_trim[:, i], H_trim[:, i])*cov_mat

        eta_trim = epsilon_trim/H_trim
        result = {"y_pre": y_pre,
                  "var": var,
                  "epsilon_trim": epsilon_trim,
                  "eta_trim": eta_trim}
        return result