from cap_res_prob import CapResProb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(0)

# Small example:
n = 20  # Number of nodes.
m = 50  # Number of edges.
K = 10  # Number of scenarios.

# Large example:
# n = 2000  # Number of nodes.
# m = 5000  # Number of edges.
# K = 1000  # Number of scenarios.

# Data generation.
A_complete = np.zeros((n,int(n*(n-1)*0.5)))
count = 0
for i in range(n-1):
    for j in range(i+1,n):
        A_complete[i,count] = 1
        A_complete[j,count] = -1
        count += 1
edges = np.random.permutation(n*(n-1))[0:m]
A = np.hstack([A_complete, -A_complete])[:,edges]  # Adjacency matrix.

p = np.random.rand(m,1)*2  # Prices on edges.
S = -A.dot(np.random.rand(m,K))*5  # Source vectors.
c = np.ones(m)*5  # Edge capacities.

# Algorithm parameters.
max_iter = 100
ep = 1e-2
mu = 0.05
prob = CapResProb(A, S, p, c)


mos_start_time = time.time()
J_mos, F_mos, Pi_mos = prob.solve_cvx(solver = 'MOSEK')
print('Mosek run time = %d' % (time.time() - mos_start_time))


start_time = time.time()
F_admm, Pi_admm, U, L = prob.solve_admm(solver='MOSEK', mu=mu)
print("ADMM run time = %d" % (time.time() - start_time))


print('J_star = %f' % J_mos)
print('L[0]   = %f' % L[0])
print('U[0]   = %f' % U[0])


L_best = [L[0]]
U_best = [U[0]]
for i in range(len(L)-1):
    L_best.append(max(L_best[i], L[i+1]))
    U_best.append(min(U_best[i], U[i+1]))


plt.figure()
plt.subplot(211)
plt.plot(U_best, linewidth = 2.0)
plt.plot(L_best, linewidth = 2.0)
plt.legend(["U(l)", "L(l)"], loc = 4)
plt.subplot(212)
plt.semilogy((np.array(U_best)-np.array(L_best))/np.array(L_best), linewidth = 2.0)
plt.semilogy((np.array(U_best)-J_mos)/J_mos, linewidth = 2.0)
plt.legend(["rel. gap", "rel. subopt."], loc = 1)
plt.savefig("bounds.pdf")
