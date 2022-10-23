import numpy as np
import pandas as pd
import time
from scipy import stats
from scipy import optimize as opt
from tqdm import tqdm


class SQMLE_VLDAR_BCD(object):
    """
    self-weighted Gaussian QMLE method for VLDAR model
    """

    def __init__(self, y):
        self.y = y  # T*N matrix

    def stationary(self, N, kappa):
        """
        a sufficient condition for stationary
        output:
        e need to <1
        """
        lam = self.lam
        p = self.p
        q = self.q
        lam = np.array(lam)
        r = max(p, q)
        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)

        lam_loc_tilde = np.append(lam_loc, [0]*(r-p)*N*N)
        lam_alpha = np.append(lam_var[N:], [0]*(r-q)*N**2)

        lam_loc_mat = lam_loc_tilde.reshape((N, N*r), order='F')
        lam_alpha_mat = lam_alpha.reshape((N, N*r), order='F')

        size = 1000
        eta = stats.multivariate_normal.rvs([0]*N, cov_mat, size)

        e_ik = np.zeros((r, N))
        for k in range(N):
            for i in range(r):
                e_ik_p = 0
                e_ik_n = 0
                for n in range(size):
                    eta_ = eta[n]

                    e_p = lam_loc_mat[:, i*N+k]+eta_*lam_alpha_mat[:, i*N+k]

                    e_n = lam_loc_mat[:, i*N+k]-eta_*lam_alpha_mat[:, i*N+k]

                    e_ik_p += (np.linalg.norm(e_p, ord=kappa))**kappa

                    e_ik_n += (np.linalg.norm(e_n, ord=kappa))**kappa
                e_ik[i, k] = max(e_ik_p/size, e_ik_n/size)
        e_k = np.sum(e_ik, axis=0)
        e = max(e_k)
        return e

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

    def loss_func_alpha(self, lam_var, lam_loc, lam_cov, p, q, r_m=0):
        """
        构建一个只关于条件方差部分的loss，其他已知道的数当作常数省略。
        """

        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

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
        self.y_trim = y_trim
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
        eta_trim_weight = eta_trim@np.diag(np.sqrt(weight))

        part_1 = np.sum(weight*np.sum(np.log(H_trim), axis=0))
        part_2 = np.trace(cov_mat_inv@eta_trim_weight@eta_trim_weight.T)
        loss_alpha = part_1+1/2*part_2
        return loss_alpha/N/(T-r)

    def jac_alpha(self, lam_var, lam_loc, lam_cov, p, q, r_m=0):
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

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

        weight = self.weight(p, q, r_m)

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs

        eta_trim = epsilon_trim / H_trim

        part = (1/H_trim-(eta_trim/H_trim) *
                (cov_mat_inv@eta_trim))@np.diag(weight)

        jac = (part@Z_abs.T).ravel('F')

        return jac/N/(T-r)

    def hess_alpha(self, lam_var, lam_loc, lam_cov, p, q, r_m=0):
        '''
        '''
        y = self.y
        T, N = np.shape(y)
        weight = self.weight(p, q, r_m)

        lam_loc_mat = lam_loc.reshape(
            (N, N*p), order='F')  # 将条件均值部分参数矩阵 m \times m*p
        lam_var_mat = lam_var.reshape(
            (N, 1+N*q), order='F')  # 将条件方差部分参数矩阵 m \times m*q

        cov_mat = np.zeros((N, N))
        cov_mat[np.tril_indices(N, -1)] = lam_cov
        cov_mat = cov_mat+cov_mat.T+np.eye(N)

        cov_mat_inv = np.linalg.inv(cov_mat)
        cov_mat_det = np.linalg.det(cov_mat)

        d_vecGamma_T_d_sigama_I_gamma_inv = self.K(
            N)@np.kron(np.eye(m), cov_mat_inv)

        # 二阶导数

        s_d_var = np.zeros((m*m*q+m, m*m*q+m))

        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        for t in range(r, T):
            wt = weight[t-r]

            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, np.abs(y[t-q:t][::-1]).ravel('C'))

            cond_mean = lam_loc_mat@y_mean
            h_var = lam_var_mat@y_var

            eta_vec = (y[t]-cond_mean)/h_var
            eta_diag = np.diag(eta_vec)
            d_h_d_alpha_T = np.kron(y_var, np.eye(m))
            d_h_T_d_alpha_D_inv = d_h_d_alpha_T.T/h_var

            # s_d_var
            s_d_var += wt*d_h_T_d_alpha_D_inv@(np.eye(N)-eta_diag@cov_mat_inv@eta_diag-2*eta_diag @
                                               np.diag(cov_mat_inv@eta_vec))@d_h_T_d_alpha_D_inv.T

        return -s_d_var/N/(T-r)

    def direct_alpha(self, lam_var, lam_loc, lam_cov, p, q, r_m=0):
        """
        direct=jac@hess
        """
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
        y = self.y
        T, N = np.shape(y)

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
        weight = self.weight(p, q, r_m)

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs
        H_trim_inv = 1/H_trim
        eta_trim = epsilon_trim * H_trim_inv
        cov_mat_inv_times_eta_trim = cov_mat_inv@eta_trim

        eta_trim_div_H_trim = eta_trim*H_trim_inv

        part = (H_trim_inv-(eta_trim_div_H_trim) *
                (cov_mat_inv_times_eta_trim))@np.diag(weight)

        jac = (part@Z_abs.T).ravel('F')
        part_hess = H_trim_inv*H_trim_inv-2*eta_trim_div_H_trim * \
            cov_mat_inv_times_eta_trim*H_trim_inv

        hess = 0
        for t in range(T-r):
            hess -= weight[t]*np.kron(np.outer(Z_abs[:, t], Z_abs[:, t]),
                                      np.diag(part_hess[:, t])-np.outer(eta_trim_div_H_trim[:, t], eta_trim_div_H_trim[:,t])*cov_mat_inv)

        return jac/N/(T-r), hess/N/(T-r)

    def bound_param(self, param, lower, upper):
        """
        将所有参数限制在lower和upper里面
        """
        n = len(param)
        lower = lower*np.ones(n)
        upper = upper*np.ones(n)
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
                lam_var = np.concatenate((lam_var, np.full((N,N), 1/(4*N**2)).ravel('F')))
            lam = np.concatenate((lam_loc, lam_var, np.zeros(N*(N-1)//2)))
        else:
            lam = init_value

        lam_loc = lam[:N*N*p]
        lam_var = lam[N*N*p:N*N*(p+q)+N]
        lam_cov = lam[N*N*(p+q)+N:]

        lam_loc_mat = lam_loc.reshape((N, N*p), order='F')  # 将条件均值部分参数矩阵 N \times N*p
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
            columns=['loc_ratio', 'var_ratio', 'cov_ratio', 'var_direct', 'lam_ratio'])
        t_loc = 0  # 用来分别记录三个部分迭代所需要花的时间
        t_var = 0
        t_cov = 0
        for i in range(max_iter):
            lam_0 = lam.copy()

            # 更新mean
            t_1 = time.time()
            part1 = 0
            part2 = 0
            for t in range(r, T):

                wt = weight[t-r]
                y_mean = y[t-p:t][::-1].ravel('C')
                y_var = np.append(1, np.abs(y[t-q:t][::-1]).ravel('C'))
                cond_mean = lam_loc_mat@y_mean
                h_var = lam_var_mat@y_var
                V = np.diag(h_var)@cov_mat@np.diag(h_var)
                V_inv = np.diag(1/h_var)@cov_mat_inv@np.diag(1/h_var)

                part1 += wt*np.kron(np.outer(y_mean, y_mean), V_inv)
                part2 += wt*np.kron(y_mean, V_inv@y[t])

            lam[:N*N*p] = np.linalg.inv(part1)@part2
           

            t_loc += time.time()-t_1

            # 更新var

            t_2 = time.time()
            lower = np.array([0.1**(N**3)]*N+[0.1**(N**3)]*(N*N*q))
            upper = np.array([10]*N+[1]*(N*N*q))
            for j in range(max_iter_var):
                lam_var_before = lam[N*N*p:N*N*(p+q)+N].copy()
                jac, hess = self.direct_alpha(lam_var, lam_loc, lam_cov, p, q, r_m)
                
                direct = np.linalg.inv(hess)@jac
                f=self.loss_func_alpha(self.bound_param(lam_var, lower, upper), lam_loc, lam_cov, p, q, r_m)
                # 黄金分割法
                a = -2
                b = 2
                c = a+0.382*(b-a)
                d = a+0.618*(b-a)
                for k in range(10**(N)):
                    fc = self.loss_func_alpha(self.bound_param(
                        lam_var-c*direct, lower, upper), lam_loc, lam_cov, p, q, r_m)
                    fd = self.loss_func_alpha(self.bound_param(
                        lam_var-d*direct, lower, upper), lam_loc, lam_cov, p, q, r_m)

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
                    if b-a < 0.1**(N**2):
                        stepsize = (a+b)/2
                        break
                stepsize = (a+b)/2

                lam_var_after = self.bound_param(lam_var-stepsize*direct, lower, upper)

                f_after = self.loss_func_alpha(lam_var_after, lam_loc, lam_cov, p, q, r_m)
                if f_after < f:
                    lam[N*N*p:N*N*(p+q)+N] = lam_var_after
                else:
                    stepsize = 0
                    lam[N*N*p:N*N*(p+q)+N] = lam_var_before

#                 print("stepsize:{}".format(stepsize))
                if i>=1 or j>=1:
                    var_diff_ratio_inner = np.linalg.norm((lam[N*N*p:N*N*(p+q)+N]-lam_var_before)/lam_var_before, ord=np.inf)
#                     print(var_diff_ratio_inner)
                    if var_diff_ratio_inner < var_tol:
                        break

            
            t_var += time.time()-t_2

            # 更新corr
            t_3 = time.time()
            Gamma = 0

            for t in range(r, T):
                wt = weight[t-r]
                y_mean = y[t-p:t][::-1].ravel('C')
                y_var = np.append(1, np.abs(y[t-q:t][::-1]).ravel('C'))
                cond_mean = lam_loc_mat@y_mean
                h_var = lam_var_mat@y_var

                epsilon = y[t]-cond_mean
                eta = epsilon/h_var
                Gamma += wt*np.outer(eta, eta)

            Gamma_diag_half_inv = np.diag(1/np.sqrt(np.diagonal(Gamma)))
            Gamma = Gamma_diag_half_inv@Gamma@Gamma_diag_half_inv

            lam[N*N*(p+q)+N:] = Gamma[np.tril_indices(N, -1)]

            cov_mat = Gamma

            cov_mat_inv = np.linalg.inv(cov_mat)
#             cov_mat_det=np.linalg.det(cov_mat)
            t_cov += time.time()-t_3


            if i >=1:
            
                loc_diff_ratio = np.linalg.norm((lam[:N*N*p]-lam_0[:N*N*p])/lam_0[:N*N*p])  # 记录loc前后的变化
                var_diff_ratio = np.linalg.norm((lam[N*N*p:N*N*(p+q)+N]-lam_0[N*N*p:N*N*(p+q)+N])/lam_0[N*N*p:N*N*(p+q)+N], ord=np.inf)
                cov_diff_ratio = np.linalg.norm((lam[N*N*(p+q)+N:]-lam_0[N*N*(p+q)+N:])/lam_0[N*N*(p+q)+N:])
                var_direct = np.linalg.norm(direct, ord=np.inf)  # 取梯度绝对值的最大值
                lam_ratio = np.linalg.norm((lam-lam_0)/lam_0, ord=np.inf)

                all_diff.loc[i] = [loc_diff_ratio, var_diff_ratio,
                                   cov_diff_ratio, var_direct, lam_ratio]

                if lam_ratio < total_tol:
                    if result_show == True:
                        print(all_diff)
                        print("===================================================")
                        print("break for reach tol")
                        print("detial each difference:\n{}".format(all_diff))
                        t_consume = pd.Series(
                            {'t_loc': t_loc, 't_var': t_var, 't_cov': t_cov})
                        print("time consume for each part:\n{}".format(t_consume))

                        print("loc:\n{} \n var:\n{} \n cov:\n{}".format(
                            lam_loc_mat, lam_var_mat, cov_mat))
                        print("===================================================")
                    self.lam = lam

                    return lam
        if result_show == True:
            print(all_diff)
            print("===================================================")
            print("break for reach max_iter")
            print("detial each difference:\n{}".format(all_diff))
            t_consume = pd.Series(
                {'t_loc': t_loc, 't_var': t_var, 't_cov': t_cov})
            print("time consume for each part:\n{}".format(t_consume))

            print("loc:\n{} \n var:\n{} \n cov:\n{}".format(
                lam_loc_mat, lam_var_mat, cov_mat))
            print("===================================================")
        self.lam = lam
        return lam

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

        y_pre = lam_loc_mat@((y[T-p:T][::-1]).ravel('C'))
        h_pre = lam_var_mat @ ( np.append(1, (y_abs[T-q:T][::-1]).ravel('C')))
        var_pre=np.outer(h_pre,h_pre)*cov_mat
        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs
        var_ios = np.zeros((T-r, N, N))
        for i in range(T-r):
            var_ios[i] = np.outer(H_trim[:, i], H_trim[:, i])*cov_mat

        eta_trim = epsilon_trim/H_trim
        result = {"y_pre": y_pre,
              "var_pre": var_pre,
              "var_ios":var_ios,
              "epsilon_ios": epsilon_trim.T,
              "eta_ios": eta_trim.T}
        return pd.Series(result)

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
        eta = self.y_pre()['eta_ios']
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

            eta_vec_T_eta_eta += np.outer(eta[t],
                                          np.outer(eta[t], eta[t]).ravel('F'))/(T-r)
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
            d_epsilon_d_theta_T = np.hstack(
                (d_epsilon_d_phi_T, np.zeros((N, d-N*N*p))))

            d_h_d_alpha_T = np.kron(y_var, np.eye(N))
            d_h_d_theta_T = np.hstack(
                (np.zeros((N, N*N*p)), d_h_d_alpha_T, np.zeros((N, N*(N-1)//2))))
            d_V_d_theta_T[:, N*N*p:N*N*(p+q)+N] = (np.kron(D@cov_mat, np.eye(
                N))+np.kron(np.eye(N), D@cov_mat))@d_D_T_d_h.T@d_h_d_alpha_T
            d_V_d_theta_T[:, N*N*(p+q)+N:] = np.kron(D, D)@self.K(N).T

            Sigma += wt*((0.5*d_V_d_theta_T.T@np.kron(V_inv, V_inv)@d_V_d_theta_T
                         + d_epsilon_d_theta_T.T@V_inv@d_epsilon_d_theta_T))/(T-r)

            part_Omega_1 = D_inv@cov_mat_inv@eta_vec_T_eta_eta@np.kron(
                cov_mat_inv@D_inv, cov_mat_inv@D_inv)
            part_Omega_2 = np.kron(
                D_inv@cov_mat_inv, D_inv@cov_mat_inv)@vec_T_eta_eta_vec_eta_eta@np.kron(cov_mat_inv@D_inv, cov_mat_inv@D_inv)
            Omega += wt**2 * (d_epsilon_d_theta_T.T@V_inv@d_epsilon_d_theta_T
                              - 0.5*d_epsilon_d_theta_T.T@part_Omega_1@d_V_d_theta_T
                              - 0.5*d_V_d_theta_T.T@part_Omega_1.T@d_epsilon_d_theta_T
                              + 0.25*d_V_d_theta_T.T@(part_Omega_2-np.outer(V_inv.ravel('F'), V_inv.ravel('F')))@d_V_d_theta_T)/(T-r)
        Sigma_inv = np.linalg.inv(Sigma)
        A_D = np.sqrt(np.diagonal(Sigma_inv@Omega@Sigma_inv/T))
        result = pd.Series(
            dict(zip(['Omega', 'Sigma', 'A_D'], [Omega, Sigma, A_D])))
        return result

    def Asymptotic_deviation_2(self):
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
        eta = self.y_pre()['eta_ios']
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

        Sigma = 0
        Omega = 0

        r = max(p, q)
        eta_tilde_cov_eta_eta_T_cov_inv_eta_tilde = 0
        vec_eta_eta_Tvec_T_eta_eta_T = 0
        eta_eta_T_gamma_eta_tilde = 0
        eta_vec_T_eta_eta_T = 0
        eta_tilde_cov_eta_vec_T_eta_eta_T = 0
        cov_mat_tilde_col = np.zeros((N, N*N))
        for i in range(N):
            cov_mat_tilde_col[:, i*N:(i+1)*N] = np.diag(cov_mat[i])

        for t in range(T-r):

            eta_tilde_cov_eta_eta_T_cov_inv_eta_tilde += np.diag(
                eta[t])@cov_mat_inv@np.outer(eta[t], eta[t])@cov_mat_inv@np.diag(eta[t])/(T-r)
            vec_eta_eta_Tvec_T_eta_eta_T += np.outer(np.outer(eta[t], eta[t]).ravel(
                'F'), np.outer(eta[t], eta[t]).ravel('F'))/(T-r)
            eta_eta_T_gamma_eta_tilde += np.outer(
                eta[t], eta[t])@cov_mat_inv@np.diag(eta[t])/(T-r)
            eta_vec_T_eta_eta_T += np.outer(eta[t],
                                            np.outer(eta[t], eta[t]).ravel('F'))/(T-r)
            eta_tilde_cov_eta_vec_T_eta_eta_T += np.diag(eta[t])@cov_mat_inv@np.outer(
                eta[t], (np.outer(eta[t], eta[t])).ravel('F'))/(T-r)

        for t in range(r, T):
            wt = weight[t-r]

            y_mean = y[t-p:t][::-1].ravel('C')
            y_var = np.append(1, np.abs(y[t-q:t][::-1]).ravel('C'))

            cond_mean = lam_loc_mat@y_mean
            epsilon = y[t]-cond_mean
            h_var = lam_var_mat@y_var
            D = np.diag(h_var)
            D_inv = np.diag(1/h_var)

            d_epsilon_d_phi_T = -np.kron(y_mean, np.eye(N))
            d_epsilon_d_theta_T = np.hstack(
                (d_epsilon_d_phi_T, np.zeros((N, d-N*N*p))))
            d_h_d_alpha_T = np.kron(y_var, np.eye(N))
            d_h_d_theta_T = np.hstack(
                (np.zeros((N, N*N*p)), d_h_d_alpha_T, np.zeros((N, N*(N-1)//2))))
            d_gamma_d_sigma_T = self.K(N).T
            d_gamma_d_theta_T = np.hstack(
                (np.zeros((N**2, N*N*(p+q)+N)), d_gamma_d_sigma_T))

            Omega += wt**2 * (d_epsilon_d_theta_T.T@D_inv@cov_mat_inv@D_inv@d_epsilon_d_theta_T
                              + d_h_d_theta_T.T@D_inv@(eta_tilde_cov_eta_eta_T_cov_inv_eta_tilde-np.full(
                                  (N, N), 1))@D_inv@d_h_d_theta_T
                              + 0.25 *
                              d_gamma_d_theta_T.T@np.kron(np.eye(N),
                                                          cov_mat_inv)
                              @ (np.kron(cov_mat_inv, np.eye(N))@vec_eta_eta_Tvec_T_eta_eta_T@np.kron(cov_mat_inv, np.eye(N))-np.outer(np.eye(N).ravel('F'), np.eye(N).ravel('F')))
                              @ np.kron(np.eye(N), cov_mat_inv)@d_gamma_d_theta_T
                              - d_epsilon_d_theta_T.T@D_inv@cov_mat_inv@eta_eta_T_gamma_eta_tilde@D_inv@d_h_d_theta_T
                              - (d_epsilon_d_theta_T.T@D_inv@cov_mat_inv @
                                 eta_eta_T_gamma_eta_tilde@D_inv@d_h_d_theta_T).T
                              - 0.5*d_epsilon_d_theta_T.T@D_inv@cov_mat_inv@eta_vec_T_eta_eta_T@np.kron(
                                  cov_mat_inv, cov_mat_inv)@d_gamma_d_theta_T
                              - (0.5*d_epsilon_d_theta_T.T@D_inv@cov_mat_inv@eta_vec_T_eta_eta_T @
                                 np.kron(cov_mat_inv, cov_mat_inv)@d_gamma_d_theta_T).T
                              + 0.5*d_h_d_theta_T.T@D_inv@(eta_tilde_cov_eta_vec_T_eta_eta_T@np.kron(cov_mat_inv, np.eye(
                                  N))-np.outer(np.ones(N), np.eye(N).ravel('F')))@np.kron(np.eye(N), cov_mat_inv)@d_gamma_d_theta_T
                              + (0.5*d_h_d_theta_T.T@D_inv@(eta_tilde_cov_eta_vec_T_eta_eta_T@np.kron(cov_mat_inv, np.eye(N))-np.outer(np.ones(N), np.eye(N).ravel('F')))@np.kron(np.eye(N), cov_mat_inv)@d_gamma_d_theta_T).T)/(T-r)

            Sigma += wt*(d_epsilon_d_theta_T.T@D_inv@cov_mat_inv@D_inv@d_epsilon_d_theta_T
                         + d_h_d_theta_T.T@D_inv@(cov_mat_inv*cov_mat+np.eye(N))@D_inv@d_h_d_theta_T
                         + 0.5 *
                         d_gamma_d_theta_T.T@np.kron(cov_mat_inv,
                                                     cov_mat_inv)@d_gamma_d_theta_T
                         + d_h_d_theta_T.T@D_inv@cov_mat_tilde_col@np.kron(
                             cov_mat_inv, cov_mat_inv)@d_gamma_d_theta_T
                         + (d_h_d_theta_T.T@D_inv@cov_mat_tilde_col@np.kron(cov_mat_inv, cov_mat_inv)@d_gamma_d_theta_T).T)/(T-r)

        Sigma_inv = np.linalg.inv(Sigma)
        A_D = np.sqrt(np.diagonal(Sigma_inv@Omega@Sigma_inv/T))
        result = pd.Series(
            dict(zip(['Omega', 'Sigma', 'A_D'], [Omega, Sigma, A_D])))
        return result

    def likelihood(self, lam, p, q, r_m):
        """
        return likelihood
        """
        y = self.y
        weight = self.weight(p, q, r_m)
        T, N = np.shape(y)

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

        Y = y.T  # Y is N*T matrix

        y_abs = np.abs(y)
        if r_m == 0:
            r = max(p, q)
        else:
            r = r_m
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

        likelihood = -np.sum(weight*np.sum(np.log(H_trim), 0))-0.5*np.sum(weight)*np.log(
            cov_mat_det)-0.5*np.trace(cov_mat_inv@(weight_eta_trim@weight_eta_trim.T))
        return likelihood

    def BIC(self, r_m=4):
        '''
        r_m: choose max p,q
        '''
        y = self.y
        T, N = np.shape(y)

        BIC_table = np.zeros((r_m, r_m))

        for p in tqdm(range(1, r_m+1)):
            for q in range(1, r_m+1):
                model = SQMLE_VLDAR_BCD(y)
                lam = model.fit(p, q, step_select=3, max_iter=10,
                                max_iter_var=4, var_tol=1e-2, total_tol=1e-2, r_m=r_m)
                likeli = self.likelihood(lam, p, q, r_m)
                BIC_table[p-1, q-1] = -2*likeli + \
                    (N**2*(p+q)+N+N*(N-1)/2)*np.log(T-r_m)
        BIC_df = pd.DataFrame(BIC_table)
        BIC_df.index = BIC_df.index+1
        BIC_df.columns = list(range(1, r_m+1))
        Best_pq_BIC = BIC_df.stack().idxmin()

        return Best_pq_BIC, BIC_df

    def cov(self, df, mean_df, lag,M):
        T, N = np.shape(df)
        cov = 0
        for t in range(M, T):
            cov += np.outer(df[t]-mean_df, df[t-lag]-mean_df)
        return cov/(T-M)

    def portmanteau_test(self, M=6):
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
        p = self.p
        q = self.q
        y = self.y
        weight = self.weight(p, q)
        T, N = np.shape(y)

        lam = self.lam

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

        y_abs = np.abs(y)
        r = max(p, q)
        y_trim = y[r:].T  # N*(T-r) matrix
        y_abs = np.abs(y)  # T*N
        y_abs_trim = np.abs(y_trim)  # N*(T-r) matrix

        Z = np.zeros((N*p, T-r))
        Z_abs = np.zeros((N*q+1, T-r))
        for i in range(T-r):
            Z[:, i] = (y[r+i-p:r+i][::-1]).ravel('C')
            Z_abs[:, i] = np.append(1, (y_abs[r+i-q:r+i][::-1]).ravel('C'))

        epsilon_trim = y_trim-lam_loc_mat@Z
        H_trim = lam_var_mat @ Z_abs

        eta = (epsilon_trim / H_trim).T
        sgn_eta = np.sign(eta)
        zeta = np.diag(weight)@eta
        zeta_abs = np.abs(zeta)

        mean_eta_abs = np.mean(np.abs(eta), 0)
        mean_zeta = np.mean(zeta, 0)
        mean_zeta_abs = np.mean(zeta_abs, 0)
        mean_zeta_sq = np.mean(zeta**2, 0)
        mean_sgn_eta = np.mean( sgn_eta, 0)

        # 算规范化的矩阵
        C_0 = self.cov(zeta, mean_zeta, 0,M)

        C_abs0 = self.cov(zeta_abs, mean_zeta_abs, 0,M)
        C_sq0 = self.cov(zeta**2, mean_zeta_sq, 0,M)
#         for t in range(T-r):
#             C_0+=np.outer((zeta[t]-mean_zeta),(zeta[t]-mean_zeta))/(T-r)
#             C_abs0+=np.outer(zeta_abs[t]-mean_zeta_abs,zeta_abs[t]-mean_zeta_abs)/(T-r)
#             C_sq0+=np.outer(zeta[t]**2-mean_zeta_sq,zeta[t]**2-mean_zeta_sq)/(T-r)
        C_d0_half_inv = np.diag(1/np.sqrt(np.diagonal(C_0)))
        C_dabs0_half_inv = np.diag(1/np.sqrt(np.diagonal(C_abs0)))
        C_dsq0_half_inv = np.diag(1/np.sqrt(np.diagonal(C_sq0)))

        # 计算样本的相关系数拉之后的向量
        R_mean = np.zeros((N, N*M))
        R_var_abs = np.zeros((N, N*M))
        R_var_sq = np.zeros((N, N*M))
        for k in range(M):
            R_mean[:, k*N:(k+1)*N] =C_d0_half_inv@self.cov(zeta,
                                                            mean_zeta, k+1,M)@C_d0_half_inv
            R_var_abs[:, k*N:(k+1)*N] =  C_dabs0_half_inv@self.cov(zeta_abs,
                                                                  mean_zeta_abs, k+1,M)@C_dabs0_half_inv
            R_var_sq[:, k*N:(k+1)*N] =  C_dsq0_half_inv@self.cov(zeta ** 2, mean_zeta_sq, k+1,M)@C_dsq0_half_inv
        r_mean = R_mean.ravel('F')
        r_var_abs = R_var_abs.ravel('F')
        r_var_sq = R_var_sq.ravel('F')

        # 计算相关系数拉之后的向量的方差
        d = (p+q)*N*N+N
        U_mean = np.zeros((N*N*M, d))
        U_var_abs = np.zeros((N*N*M, d))
        U_var_sq = np.zeros((N*N*M, d))

        G_mean = 0
        G_var_abs = 0
        G_var_sq = 0
        G_mix_abs = 0
        G_mix_sq = 0
        for t in range(M, T-r):
            part_1 = (np.diag(1/H_trim[:, t])@cov_mat_inv @
                      np.outer(eta[t], Z[:, t])).ravel('F')
            part_2 = -(np.outer(1/H_trim[:, t]-eta[t]/H_trim[:, t]
                                * (cov_mat_inv@eta[t]), Z_abs[:, t])).ravel('F')
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
                v_var_abs[k*N*N:(k+1)*N*N] = np.outer(zeta_abs[t] -
                                                      mean_zeta_abs, zeta_abs[t-k-1]-mean_zeta_abs).ravel('F')

                U_var_abs[k*N*N:(k+1)*N*N] -= weight[t]*(np.kron(np.outer((zeta_abs[t-k-1]-mean_zeta_abs), np.concatenate((Z[:, t].ravel('F'), np.zeros(N*q+1)))), np.diag(mean_sgn_eta/H_trim[:, t])) +
                                                         np.kron(np.outer((zeta_abs[t-k-1]-mean_zeta_abs), np.concatenate((np.zeros(N*p), Z_abs[:, t]))), np.diag(mean_eta_abs/H_trim[:, t])))/(T-r-M)

                # r_var_sq
                v_var_sq[k*N*N:(k+1)*N*N] = np.outer(zeta[t]**2 -
                                                     np.ones(N), zeta[t-k-1]**2-np.ones(N)).ravel('F')
                U_var_sq[k*N*N:(k+1)*N*N] += (-2*np.kron(np.outer(zeta[t-k-1]**2-np.ones(N), np.concatenate(
                    (np.zeros(N*p), Z_abs[:, t].ravel('F')))), np.diag(weight[t]/H_trim[:, t])))/(T-r-M)
            v_mix_abs = np.concatenate(
                (v_mean, v_var_abs, weight[t]*d_lt_d_lambda))
            G_mix_abs += np.outer(v_mix_abs, v_mix_abs)/(T-r-M)

            v_mix_sq = np.concatenate(
                (v_mean, v_var_sq, weight[t]*d_lt_d_lambda))
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
        coe_abs = np.kron(np.eye(M), np.kron(
            C_dabs0_half_inv, C_dabs0_half_inv))
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
        H_mix_abs_right = np.vstack(
            (E_wt2**(-1)*U_mean, coe_abs@U_var_abs))@Sigma_inv
        H_mix_abs = np.hstack((H_mix_abs_left, H_mix_abs_right))
        V_mix_abs = H_mix_abs@G_mix_abs@H_mix_abs.T

        r_mix_sq = np.concatenate((r_mean, r_var_sq))
        H_mix_sq_left = np.vstack((np.hstack((E_wt2**(-1)*np.eye(N*N*M), np.zeros((N*N*M, N*N*M)))),
                                   np.hstack((np.zeros((N*N*M, N*N*M)), coe_sq))))
        H_mix_sq_right = np.vstack(
            (E_wt2**(-1)*U_mean, coe_sq@U_var_sq))@Sigma_inv
        H_mix_sq = np.hstack((H_mix_sq_left, H_mix_sq_right))
        V_mix_sq = H_mix_sq@G_mix_sq@H_mix_sq.T
        self.V_mean = np.diag(V_mean)
        self.V_var_abs = np.diag(V_var_abs)
        self.V_var_sq = np.diag(V_var_sq)
        self.V_mix_abs = np.diag(V_mix_abs)
        self.V_mix_sq = np.diag(V_mix_sq)
        stat_mean =(T-r-M)* r_mean@np.linalg.inv(V_mean)@r_mean
        stat_var_abs =(T-r-M)* r_var_abs@np.linalg.inv(V_var_abs)@r_var_abs
        stat_var_sq =(T-r-M)* r_var_sq@np.linalg.inv(V_var_sq)@r_var_sq
        stat_mix_abs =(T-r-M)* r_mix_abs@np.linalg.inv(V_mix_abs)@r_mix_abs
        stat_mix_sq =(T-r-M)* r_mix_sq@np.linalg.inv(V_mix_sq)@r_mix_sq

        stat_name = ['stat_mean', 'stat_var_abs',
                     'stat_var_sq', 'stat_mix_abs', 'stat_mix_sq']
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
    