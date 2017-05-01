opt_cap_res
====

The package opt_cap_res (optimal capacity reservation) is for reserving link capacity in a network in such a way that any of a given set of flow scenarios can be supported. It solves the following problem
```
minimize    p^T r
subject to  AF(s) + s = 0, 0 <= F(s) <= r, ∀s ∈ S
            r <= c.
```
The price vector ```p```, the graph incidence matrix ```A```, the finite set of source vectors ```S```, and the edge capacity vector ```c``` are given, and the variables to be determined are the flow policy ```F``` and the reservation vector ```r```.

For more information please see our paper [A Distributed Method for Optimal Capacity Reservation](https://stanford.edu/~boyd/papers/opt_cap_res.html).

Installation
------------
You should first install [CVXPY](http://ww.cvxpy.org/), following the instructions [here](http://www.cvxpy.org/en/latest/install/index.html).

Illustrative example
------------

In a simple example we have ```n = 5``` nodes, ```m = 10``` edges, and ```K = 8``` scenarios. 
The randomly generated graph is as follows.

![Graph](/figures/graph.pdf?raw=true "Graph")

We use price vector ```p = 1``` and capacity vector ```c = 1```. The scenario source vectors were randomly generated.

The optimal reservation cost is ```6.0```, and the objective of the heuristic policy is ```7.6```. (The lower bound from the heuristic policy is ```2.3```.) The optimal and heuristic flow policies are shown in the following figure. 

![edge_flows](/figures/edge_flows.pdf?raw=true "Edge flows")

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
