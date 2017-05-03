opt_cap_res
====

The Python package opt_cap_res (optimal capacity reservation) is for reserving link capacity in a network in such a way that any of a given set of flow scenarios can be supported. It solves the following problem
```
minimize    p^T r
subject to  A f^(k) = s^(k), 0 <= f^(k) <= r, k = 1, ..., K,
            r <= c.
```
The price vector ```p```, the graph incidence matrix ```A```, the source vectors ```s^(k), k = 1, ..., K```, and the edge capacity vector ```c``` are given, and the variables to be determined are the flow policy ```f^(k), k = 1, ..., K```, and the reservation vector ```r```.

For more information please see our paper [A Distributed Method for Optimal Capacity Reservation](https://stanford.edu/~boyd/papers/opt_cap_res.html).

Installation
------------
You should first install [CVXPY](http://ww.cvxpy.org/), following the instructions [here](http://www.cvxpy.org/en/latest/install/index.html).

Illustrative example
------------

In a simple example we have ```n = 5``` nodes, ```m = 10``` edges, and ```K = 8``` scenarios. 
The randomly generated graph is as follows.

<img src="/figures/graph.png" alt="Graph" width="300" height="300"/>

We use price vector ```p = 1``` and capacity vector ```c = 1```. The scenario source vectors were randomly generated.

The code to call the solving method is as follows.
```python
prob = CapResProb(A, S, p, c)
F, Pi, U, L = prob.solve_admm()
```
The result gives flow policy matrix ```F```, the scenario prices ```Pi```, and upper and lower bounds on the objective ```U``` and ```L```.

The optimal reservation cost is ```6.0```, and the cost of a heuristic flow policy, which greedily minimizes the
cost for each source separately but does not coordinate the flows for the different sources to reduce the reservation cost,
is ```7.6```. (The lower bound from the heuristic policy is ```2.3```.) The optimal and heuristic flow policies are shown in the following figure. 

<img src="/figures/edge_flows.png" alt="Edge flows" width="600" height="400"/>

The upper plot shows the optimal policy, and the lower plot shows the heuristic policy. For each plot, the bars show the flow policy; the 10 groups are the edges, and the 8 bars are the edge flows under each scenario. The line above each group of bars is the reservation for that edge.

Optimal scenario prices are given in the following table. 

| Edge\  Scenario    |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8|
| --------- |:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---------:|
|  1        |            |            |  1.0       |            |            |            |            |           |
|  2        |            | 0.33       |            |  0.33      |            |            |  0.33      |           |
|  3        |            |            |  0.38      |            |  0.28      |            |            | 0.33      |
|  4        |            |            |  1.0       |            |            |            |            |           |
|  5        |   1.0      |            |            |            |            |            |            |           |
|  6        |   1.0      |            |            |            |            |            |            |           |
|  7        |   0.38     |            |  0.62      |            |            |            |            |           |
|  8        |            |  0.33      |            |  0.33      |            |            |  0.33      |           |
|  9        |            |            |            |            |            |  1.0       |            |           |
|  10       |            |  0.33      |            |  0.33      |            |            |  0.33      |           |
