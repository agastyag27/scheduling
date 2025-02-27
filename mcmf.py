import networkx as nx
import random

from constants import Constants

def get_fudge(temp):
    if temp < Constants.eps:
        return 0
    a = 0
    g = random.randint(0, 2**32 - 1)
    while g - (temp+1) * (g // (temp+1)) < temp:
        a += 1
        g = random.randint(0, 2**32 - 1)
    return a

class MCMF:
    def __init__(self, N, temp=0):
        self.N = N
        self.temp = temp
        self.G = nx.DiGraph()
        # Initialize nodes with zero demand.
        for i in range(N):
            self.G.add_node(i, demand=0)
        # Dictionary to store lower bounds for edges.
        self.lower_bounds = {}
        # This will hold the computed flow after bounded_flow is called.
        self.flow = None

    def add_edge(self, u, v, cap, cost, to_fudge=True):
        # Optionally add random “fudge” to cost.
        if to_fudge and Constants.fudge:
            cost += get_fudge(self.temp)
        self.G.add_edge(u, v, capacity=cap, weight=cost)
        self.lower_bounds[(u, v)] = 0

    def add_bounded(self, u, v, lb, ub, cost, to_fudge=True):
        if to_fudge and Constants.fudge:
            cost += get_fudge(self.temp)
        # Add edge for the extra capacity (upper bound minus lower bound).
        if ub > lb:
            self.G.add_edge(u, v, capacity=ub - lb, weight=cost)
        self.lower_bounds[(u, v)] = lb
        # Adjust node demands so that at least lb flow is pushed on this edge.
        self.G.nodes[u]['demand'] += lb
        self.G.nodes[v]['demand'] -= lb

    def bounded_flow(self, s, t):
        # Add an edge from t back to s to allow circulation.
        self.G.add_edge(t, s, capacity=Constants.INF, weight=0)
        try:
            flowCost, flowDict = nx.network_simplex(self.G)
        except nx.NetworkXUnfeasible:
            self.flow = None
            return (False, None)
        self.flow = flowDict
        return (True, flowCost)

    def get_flow(self, u, v):
        if self.flow is None:
            raise ValueError("Flow has not been computed. Ensure that bounded_flow() returns a feasible solution before calling get_flow().")
        # The actual flow is the computed flow plus the lower bound.
        return self.flow.get(u, {}).get(v, 0) + self.lower_bounds.get((u, v), 0)