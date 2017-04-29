import cvxpy as cvx
import numpy as np
from multiprocessing import Pool, Array, Value


class CapResProb():

    def __init__(self, A, S, p, c):
        '''
        :param A: incidence matrix
        :param S: matrix of source vectors
        :param p: price vector
        :param c: capacity
        :return: None
        '''
        self.A = A
        self.S = S
        self.p = p
        self.c = c
        self.n, self.K = S.shape
        self.m = p.shape[0]
        if not A.shape == (self.n, self.m):
            raise ValueError('shape mismatched')

    def solve_admm(self, mu = 0.05, alpha = 1.9, eps_rel=1e-2, bound_interval=10,
                   max_iter=100, n_thread=64, solver = 'MOSEK'):
        '''
        parallel ADMM
        :param mu: ADMM penalty parameter (unscaled)
        :param alpha: over-relaxation parameter
        :param eps_rel: relative suboptimal accuracy
        :param bound_inverval: number of iterations between every 2 updates of bounds
        :param max_iter: maximum number of iterations
        :param n_thread: number of processes
        :param solver: subproblem solver
        :return: flow matrix F, dual variable Pi, upper bounds U, lower bounds L
        '''

        # shared memory
        F = Array('d',self.m*self.K)
        Ft = Array('d',self.m*self.K)
        Pi = Array('d',self.m*self.K)
        Ft_pre = Array('d',self.m*self.K)
        rho = Value('d',mu)

        def getarray(F):    return np.reshape(F, (self.K, self.m)).T
        def getcol(F,k):    return np.array(F[k*self.m:(k+1)*self.m])
        def getrow(F,j):    return np.array(F[j:(self.K-1)*self.m+j+1:self.m])
        def putcol(F,f,k):  F[k*self.m:(k+1)*self.m] = f
        def putrow(F,f,j):  F[j:(self.K-1)*self.m+j+1:self.m] = f

        global flow_update
        def flow_update(k):
            '''
            call flow update and write memory
            :param k: column index
            :return: None
            '''
            putcol(F, self._flow_update(getcol(Ft,k), getcol(Pi,k), k,
                   rho.value, solver)[0], k)

        global get_L
        def get_L(k):
            '''
            call flow update and obtain lower bound for a scenario price
            :param k: column index
            :return: lower bound for one scenario price
            '''
            return self._flow_update(np.zeros(self.m), getcol(Pi,k), k, 0, solver)[1]

        global solve_heuristic
        def solve_heuristic(k):
            '''
            solve heuristic for one column and write memory
            :param k: column index
            :return: heuristic objective value for one column
            '''
            f, val = self._flow_update(getcol(Ft,k), self.p/self.K, k, 0, solver)
            putcol(Ft, f, k)
            return val

        global res_update
        def res_update(j):
            '''
            call reservation update and write memory
            :param j: row index
            :return: None
            '''
            putrow(Ft, self._res_update(alpha*getrow(F,j) +
                   (1-alpha)*getrow(Ft,j), getrow(Pi,j), j, rho.value), j)


        pool1 = Pool(processes = n_thread)
        pool2 = Pool(processes = n_thread)


        print("Starting ADMM")

        ## initialization by heuristic
        L = [sum(pool1.map(solve_heuristic, [k for k in range(self.K)]))]
        U = [self.get_U(getarray(Ft))]
        [putcol(Pi, self.p/self.K, k) for k in range(self.K)]
        print('lower bound from heur.: %f, upper bound from heuristic: %f'
              % (L[0], U[0]))

        rho.value = mu * np.sum(self.p)/np.amax(np.sum(getarray(Ft),0)) # scaling of rho

        primal_res = []
        dual_res = []

        ## ADMM loop
        for l in range(1, max_iter+1):
            pool1.map(flow_update, [k for k in range(self.K)]) # flow update
            Ft_pre[:] = np.array(Ft[:]) # save the previous update on Ft
            pool2.map(res_update, [j for j in range(self.m)]) # reservation update
            self._price_update(F, Ft, Ft_pre, Pi, rho.value, alpha) # price update

            U.append(self.get_U(getarray(F))) # upper bound
            if l % bound_interval == 0: # lower bound updated every bound_interval iterations
                L.append(sum(pool1.map(get_L, [j for j in range(self.K)]))) # sum of lower bounds for all scenario prices
            else:
                L.append(L[l-1])

            print('iteration %d:  lower bound: %f, upper bound: %f' % (l, L[l], U[l]))
            if U[l] - L[l] < eps_rel*L[l]:
                break

        print("Terminating ADMM")

        pool1.terminate()
        pool2.terminate()
        return getarray(F), getarray(Pi), U, L


    def _flow_update(self, ft, pi, k, rho, solver='ECOS'):
        f = cvx.Variable(self.m)
        constrs = [self.A*f + self.S[:,k] == 0, f >= 0, f <= self.c]
        obj = f.T*pi + (rho/2.)*cvx.sum_squares(f - ft)
        prob = cvx.Problem(cvx.Minimize(obj), constrs)
        prob.solve(solver=solver)
        return np.array(f.value), prob.value
    
    def _res_update(self, f, pi, j, rho):
        K = f.size
        beta = self.p[j] / rho
        u0 = f + pi / rho
        u = np.sort(u0)[::-1]
        cumsum = 0
        for k in range(K-1):
            cumsum += u[k]
            t = (cumsum - beta)/(k+1)
            if u[k+1] <= t:
                break
        f = cvx.min_elemwise(u0,t).value
        return np.array(f)
    
    def _price_update(self, F, Ft, Ft_pre, Pi, rho, alpha):
        Pi[:] = np.array(Pi[:]) + rho*(alpha*np.array(F[:]) + \
                (1-alpha)*np.array(Ft_pre[:]) - np.array(Ft[:]))

    def get_U(self, F):
        '''
        :param F: feasible flows
        :return: upper bound
        '''
        return (self.p.T * cvx.max_entries(F, axis=1)).value


    def solve_cvx(self, solver='MOSEK'):
        '''
        solve the full capacity reservation problem
        :param solver: solver interfaced via CVXPY
        :return: objective value, flow matrix, dual variable
        '''
        F = cvx.Variable(self.m, self.K)
        Ft = cvx.Variable(self.m, self.K)
        cost = self.p.T * cvx.max_entries(Ft, axis = 1)
        constr = F <= Ft
        constrs = [self.A*F + self.S == 0, F >=0, constr] + \
                  [F[:,k] <= self.c for k in range(self.K)]
        prob = cvx.Problem(cvx.Minimize(cost), constrs)
        print("Solver "+solver+" begins:")
        prob.solve(solver = solver)
        print("Solver "+solver+" ends")
        return cost.value, F.value, constr.dual_value
