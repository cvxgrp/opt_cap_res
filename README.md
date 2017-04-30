opt_cap_res
====

The package opt_cap_res (optimal capacity reservation) is for reserving link capacity in a network in such a way that any of a given set of flow scenarios can be supported. It solves the following problem
```
minimize    p^T r
subject to  AF(s) + s = 0, 0 <= F(s) <= r, ∀s ∈ S
            r <= c.
```
The price vector ```p```, the graph incidence matrix ```A```, the set of source vectors ```S```, and the edge capacity vector ```c``` are given, and the variables to be determined are the flow policy ```F``` and the reservation vector ```r```.

For more information please see our paper [A Distributed Method for Optimal Capacity Reservation](https://stanford.edu/~boyd/papers/opt_cap_res.html).

Installation
------------
You should first install [CVXPY](http://ww.cvxpy.org/), following the instructions [here](http://www.cvxpy.org/en/latest/install/index.html).

Illustrative example
------------
