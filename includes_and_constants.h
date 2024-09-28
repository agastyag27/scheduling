#pragma once

#include <bits/stdc++.h>
#include <bits/extc++.h>

const int NUM_PERIODS = 7;

using namespace std;
typedef long double ld;
typedef array<int, NUM_PERIODS> arr_num_per;
typedef array<bool, NUM_PERIODS> bool_arr_num_per;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> pii;
typedef pair<int, pii> pipii;
typedef long long ll;
typedef vector<ll> VL;
const ld eps = 1e-5;
const ld max_temp = 20; // maybe too large?


const ll INF = numeric_limits<ll>::max() / 4;
const int SEED = 0;
mt19937 gen(SEED);
const bool fudge = 1;
const ld spread_weight = 5;
const ld class_weight = 20;
const ld period_weight = 1;
const int num_sub_iters = 10;
const int num_hill_climbs = 100;
const int zero_cost_weight = 170; // at this cost, teachers would be somewhat ambivalent about teaching this class
const int classes_out_of_range_penalty = 100;