#pragma once

#define rep(i, a, b) for(int i = a; i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()

#include "includes_and_constants.h"

int get_fudge(ld temp) {
	// choose a random positive integer noise
	
	if (temp < eps) return 0;
	int a = 0;
	unsigned int g = gen();
	while (g-(temp+1)*((long long)(g/(temp+1))) < temp) {
		a++;
		g = gen();
	}
	return a;
}

// from the kactl codebook with modifications

/**
 * Author: Stanford
 * Date: Unknown
 * Source: Stanford Notebook
 * Description: Min-cost max-flow. cap[i][j] != cap[j][i] is allowed; double edges are not.
 *  If costs can be negative, call setpi before maxflow, but note that negative cost cycles are not supported.
 *  To obtain the actual flow, look at positive values only.
 * Status: Tested on kattis:mincostmaxflow, stress-tested against another implementation
 * Time: Approximately O(E^2)
 */

struct MCMF {
	int N;
	vector<vi> ed, red;
	vector<VL> cap, flow, cost, min_flow;
	vi seen;
	VL dist, pi;
	vector<pii> par;
	vi net; // positive => connect source, negative => connect sink
	bool is_neg;
	ld temp;
	MCMF(int N, ld temp = 0) :
		N(N), ed(N), red(N), cap(N, VL(N)), flow(cap), cost(cap),
		seen(N), dist(N), pi(N), par(N), net(N, 0), min_flow(cap), is_neg(0), temp(temp) {
			// cerr << N << '\n';
		}
	// pass in N+2 to this function

	void addEdge(int from, int to, ll cap, ll cost, bool to_fudge = fudge) {
		// assert(from < N);
		// assert(to < N);
		// assert(cost == 0);
		if (fudge) {
			// cost = max(min(0ll, cost), cost-get_fudge(temp));
			cost += get_fudge(temp);
		}
		this->cap[from][to] = cap;
		this->cost[from][to] = cost;
		ed[from].push_back(to);
		red[to].push_back(from);
		if (cost < 0) is_neg = 1;
	}

	void addBounded(int from, int to, ll lb, ll ub, ll cost, bool to_fudge = fudge) {
		// assert(from < N);
		// assert(to < N);
		if (ub>lb) addEdge(from, to, ub-lb, cost, to_fudge);
		min_flow[from][to] = lb;
		net[from] -= lb;
		net[to] += lb;
	}

	pair<bool, ll> bounded_flow(int s, int t) {
		// assert(s < N);
		// assert(t < N);
		addEdge(t, s, INF, 0);
		for (int i = 0; i < N-2; i++) {
			if (net[i] > 0) addEdge(N-2, i, net[i], 0, 0);
			if (net[i] < 0) addEdge(i, N-1, -net[i], 0, 0);
		}
		// if (is_neg) setpi(N);
		pair<ll, ll> res = maxflow(N-2, N-1);
		// cerr << flow[t][s] << endl;
		for (int v: ed[N-2])
			if (flow[N-2][v] != cap[N-2][v]) {
				return pair<bool, ll>(0, res.second);
			}
		for (int v: red[N-1])
			if (flow[v][N-1] != cap[v][N-1]) {
				return pair<bool, ll>(0, res.second);;
			}
		return pair<bool, ll>(1, res.second);
	}

	void path(int s) {
		fill(all(seen), 0);
		fill(all(dist), INF);
		dist[s] = 0; ll di;

		__gnu_pbds::priority_queue<pair<ll, int>> q;
		vector<decltype(q)::point_iterator> its(N);
		q.push({0, s});

		auto relax = [&](int i, ll cap, ll cost, int dir) {
			ll val = di - pi[i] + cost;
			if (cap && val < dist[i]) {
				dist[i] = val;
				par[i] = {s, dir};
				if (its[i] == q.end()) its[i] = q.push({-dist[i], i});
				else q.modify(its[i], {-dist[i], i});
			}
		};

		while (!q.empty()) {
			s = q.top().second; q.pop();
			seen[s] = 1; di = dist[s] + pi[s];
			for (int i : ed[s]) if (!seen[i])
				relax(i, cap[s][i] - flow[s][i], cost[s][i], 1);
			for (int i : red[s]) if (!seen[i])
				relax(i, flow[i][s], -cost[i][s], 0);
		}
		rep(i,0,N) pi[i] = min(pi[i] + dist[i], INF);
	}

	ll get_flow(int i, int j) {
		return flow[i][j] + min_flow[i][j];
	}

	pair<ll, ll> maxflow(int s, int t) {
		// assert(s < N);
		// assert(t < N);
		if (is_neg) setpi(s);
		ll totflow = 0, totcost = 0;
		while (path(s), seen[t]) {
			ll fl = INF;
			for (int p,r,x = t; tie(p,r) = par[x], x != s; x = p)
				fl = min(fl, r ? cap[p][x] - flow[p][x] : flow[x][p]);
			totflow += fl;
			for (int p,r,x = t; tie(p,r) = par[x], x != s; x = p)
				if (r) flow[p][x] += fl;
				else flow[x][p] -= fl;
		}
		rep(i,0,N) rep(j,0,N) totcost += cost[i][j] * flow[i][j];
		return {totflow, totcost};
	}

	// If some costs can be negative, call this before maxflow:
	void setpi(int s) { // (otherwise, leave this out)
		// assert(false); // giving negative cycles somehow
		fill(all(pi), INF); pi[s] = 0;
		int it = N, ch = 1; ll v;
		while (ch-- && it--)
			rep(i,0,N) if (pi[i] != INF)
				for (int to : ed[i]) if (cap[i][to])
					if ((v = pi[i] + cost[i][to]) < pi[to])
						pi[to] = v, ch = 1;
		assert(it >= 0); // negative cost cycle
	}
};