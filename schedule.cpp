#include "includes_and_constants.h"
#include "mcmf.h"
#include "csvreader.h"

int n, m; // n is for classes, m is for teachers

vector<arr_num_per> schedule; // 0 is blank, otherwise 1 indexed

csv_file classes;
csv_file teachers;
csv_file presets;
vvi people_teaching;
vi num_preset;
vi num_avail;

bool make_teacher_assignments(vvi &is_teaching, ld temp = 0) {
    MCMF teacher_class_matching(3*m+n+4, temp); // +2 because of ranges
    int source = 3*m+n, sink = 3*m+n+1;
    int ex = 0;
    is_teaching.clear();
    is_teaching.resize(m);

    for (int i = 0; i < m; i++) {
        int extra_teaching = teachers.get(i, "sections");// - num_preset[i];
        assert(extra_teaching >= 0);
        if (extra_teaching == 0) continue;
        int lim = min(num_avail[i], 2);
        int num_teaching;
        if (extra_teaching <= 2 ||
        	(extra_teaching == 3 && (gen()&1)) ||
        	(extra_teaching == 4 && gen() % 4 == 0)) {
            teacher_class_matching.addBounded(source, i, 1, 1, 0);
        	num_teaching = 1;
        }
        else {
            teacher_class_matching.addBounded(source, i, lim, lim, 0);
            num_teaching = 2;
        }
        ex += num_teaching;
        if (num_teaching == 1 && extra_teaching == 4) {
	        teacher_class_matching.addEdge(i, 2*m+i, 1, 0); // no teacher can teach 2x college prep
	    }
	    else {
	    	teacher_class_matching.addEdge(i, m+i, 1, 0);
	        teacher_class_matching.addEdge(i, 2*m+i, 2, 0); // no teacher can teach 2x college prep
	    }
        for (int v = 0; v < n; v++) {
            string class_name = classes.entry_names[v];
            if (teachers.get(i, class_name) == 0) continue;
            if (classes.get(v, "isCollegePrep")) {
                teacher_class_matching.addBounded(m+i, 3*m+v, 0, 1,
                    zero_cost_weight - teachers.get(i, class_name));
            }
            else {
                teacher_class_matching.addBounded(2*m+i, 3*m+v, 0, 1, 
                    zero_cost_weight - teachers.get(i, class_name));
            }
        }
    }
    for (int i = 0; i < n; i++) {
        teacher_class_matching.addBounded(3*m+i, sink,
            classes.get(i, "minTeachers"), classes.get(i, "maxTeachers"), 0);
    }
    teacher_class_matching.bounded_flow(source, sink);

    for (int i = 0; i < m; i++) {
        int max_secs = 0;
        for (int j = 0; j < n; j++) {
            if (teacher_class_matching.get_flow(m+i, 3*m+j) ||
                teacher_class_matching.get_flow(2*m+i, 3*m+j)) {
                is_teaching[i].push_back(j);
                max_secs += classes.get(j, "sections");
            }
        }
        if (max_secs < teachers.get(i, "sections")) return 0;
    }
    return 1;
}

int _abs(int v) {
	return v < 0 ? -v: v;
}

string get_class(int v) {
	return v == 0 ? "" : v < 0 ? classes.entry_names[-v-1] + " prep" : classes.entry_names[v-1];
}

int encode(int v, bool is_prep) {
	return is_prep ? -v-1 : v+1;
}

int decode(int v) {
	if (v == 0) return -1; // yes, i know
	return v-1;
}

void print_matching(vvi &is_teaching, bool to_cout = 0) {
	ofstream *file_out = new ofstream("class_assignments.txt");
	ostream *fout = (to_cout ? &cerr : file_out);
	for (int i = 0; i < m; i++) {
		(*fout) << teachers.entry_names[i];
		for (int j = 0; j < 4-teachers.entry_names[i].size()/4; j++) (*fout) << '\t';
		for (int v: is_teaching[i]) {
			(*fout) << classes.entry_names[v] << ' ';
		}
		(*fout) << '\n';
	}
	if (!to_cout) file_out->close();
}

bool check_section_feasable(vvi &is_teaching) {
	MCMF section_number_matching(m+n+4);
	int src = m+n, snk = m+n+1;
	int in = 0, out = 0;
	for (int i = 0; i < m; i++) {
		int numSections = teachers.get(i, "sections");
		in += numSections;
		section_number_matching.addBounded(src, i, numSections, numSections, 0, 0);
		for (int v: is_teaching[i]) {
			section_number_matching.addBounded(i, m+v, 1,
				classes.get(v, "isCollegePrep") ? 3 : 4, 0, 0);
		}
	}
	for (int i = 0; i < n; i++) {
		int numSections = classes.get(i, "sections");
		out += numSections;
		section_number_matching.addBounded(m+i, snk, numSections, numSections, 0, 0);
	}
	return (section_number_matching.bounded_flow(src, snk).first);
}

vector<bool_arr_num_per> assign_preps() {
	vector<bool_arr_num_per> preps(n, bool_arr_num_per());

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < NUM_PERIODS; j++) {
			preps[i][j] = 0;
		}
	}

	vi requires_prep;
	for (int i = 0; i < n; i++)
		if (classes.get(i, "needsCommonPrep") &&
			people_teaching[i].size() > 1) requires_prep.push_back(i);
	for (int i = 1; i < requires_prep.size(); i++)
		swap(requires_prep[i], requires_prep[gen()%(i+1)]);

	for (int v: requires_prep) {
		bool avail[NUM_PERIODS]; fill(avail, avail+NUM_PERIODS, 1);
		for (int i: people_teaching[v]) {
			for (int j = 0; j < NUM_PERIODS; j++)
				avail[j] &= (teachers.get(i, "per"+to_string(j+1))>0) & (!preps[i][j]);
		}
		vi p;
		for (int i = 0; i < NUM_PERIODS; i++)
			if (avail[i]) p.push_back(i);
		if (p.empty()) {
			cerr << classes.entry_names[v] << '\n';
			assert(false);
		}
		int prep = p[gen()%p.size()];
		for (int i: people_teaching[v]) {
			preps[i][prep] = 1;
			schedule[i][prep] = encode(v, 1); // does this line need to go?
		}
	}
	return preps;
}

void print_schedule(ld cost) {
	ofstream fout("schedule.txt");
	fout << cost << '\n';
	fout << "name,per1,per2,per3,per4,per5,per6,per7\n";
	for (int i = 0; i < m; i++) {
		fout << teachers.entry_names[i];
		for (int t = 0; t < NUM_PERIODS; t++) fout << ',' << get_class(schedule[i][t]);
		fout << '\n';
	}
	fout.close();
}

ld evaluate_spread_single(int cla, int cnt) {
	return (cla == -1) ? 0 : 
		(ld)cnt*cnt / classes.get(cla, "sections") + 3*(cnt == 0);
}

ld evaluate_spread(vvi &count) {
	ld sum = 0;
	for (int t = 0; t < NUM_PERIODS; t++) {
		for (int i = 0; i < n; i++) {
			sum += evaluate_spread_single(i, count[t][i]);
		}
	}
	return sum;
}

ld get_point_period_cost(int i, int j) {
	return 5-teachers.get(i, "per"+to_string(j+1));
}

ld evaluate_period_assignments() {
	ld sum = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < NUM_PERIODS; j++) {
			if (schedule[i][j]) sum += get_point_period_cost(i, j);
		}
	}
	return sum;
}

ld evaluate_class_assignments(vvi &classes_teaching) {
	ld sum = 0;
	for (int i = 0; i < m; i++) {
		for (int v: classes_teaching[i]) sum += zero_cost_weight - teachers.get(i, classes.entry_names[v]);
	}
	return sum;
}

ld evaluate_schedule(vvi &count, vvi &classes_teaching) {
	return spread_weight*evaluate_spread(count) + period_weight*evaluate_period_assignments() + class_weight*evaluate_class_assignments(classes_teaching);
	/*
	Factors:
		- spread (climbable)
		- period assignments (climbable)
		- class assignments
	*/
}

void print_diagnostics(vvi &count, vvi &classes_teaching) {
	ofstream fout("schedule_diagnostics.txt");
	ld spread_score = evaluate_spread(count);
	fout << "Raw Spread: " << spread_score << '\n';
	fout << "Adj. Spread: " << spread_weight*spread_score << '\n';
	ld period_score = evaluate_period_assignments();
	fout << "Raw Period Score: " << period_score << '\n';
	fout << "Adj. Period Score: " << period_weight*period_score << '\n';
	ld class_score = evaluate_class_assignments(classes_teaching);
	fout << "Raw Class Score: " << class_score << '\n';
	fout << "Adj. Class Score: " << class_weight*class_score << '\n';
	fout << "\nTotal (sum of adj): " << evaluate_schedule(count, classes_teaching) << '\n';
	fout.close();
}

int hsh(int per, int cla) {
 	return NUM_PERIODS*cla + per;
}

pii dehsh(int v) {
	return pii(v%NUM_PERIODS, v/NUM_PERIODS);
}

int get_count(vvi &count, int i, int j) {
	return j == -1 ? 0 : count[i][j];
}

bool hill_climb(vector<bool_arr_num_per> &preps, vvi &count) {
	bool found = 0;
	for (int i = 0; i < m; i++) {
		bool f = 1;
		while (f) {
			f = 0;
			for (int j = 0; j < NUM_PERIODS && !f; j++) {
				if (preps[i][j] || teachers.get(i, "per"+to_string(j+1)) == 0) continue;
				for (int k = j+1; k < NUM_PERIODS && !f; k++) {
					if (preps[i][k] || teachers.get(i, "per"+to_string(k+1)) == 0 ||
						schedule[i][j] == schedule[i][k]) continue;

					int a = decode(schedule[i][j]), b = decode(schedule[i][k]);
					ld cur_cost = spread_weight*(evaluate_spread_single(a, get_count(count, j, a))+evaluate_spread_single(b, get_count(count, k, b))+
						evaluate_spread_single(a, get_count(count, k, a))+evaluate_spread_single(b, get_count(count, j, b)))
						+ period_weight*((schedule[i][j] ? get_point_period_cost(i, j) : 0) + (schedule[i][k] ? get_point_period_cost(i, k) : 0));
					ld next_cost = spread_weight*(evaluate_spread_single(a, get_count(count, j, a)-1)+evaluate_spread_single(b, get_count(count, k, b)-1)+
						evaluate_spread_single(a, get_count(count, k, a)+1)+evaluate_spread_single(b, get_count(count, j, b)+1))
						+ period_weight*((schedule[i][k] ? get_point_period_cost(i, j) : 0) + (schedule[i][j] ? get_point_period_cost(i, k) : 0));
					if (cur_cost > eps + next_cost) {

						f = 1;
						if (a != -1) {
							count[j][a]--; count[k][a]++;
						}
						if (b != -1) {
							count[j][b]++; count[k][b]--;
						}
						swap(schedule[i][j], schedule[i][k]);
						found = 1;
					}
				}
			}
		}
	}

	return found;
}

vvi iter(vector<bool_arr_num_per> &preps, vvi &classes_teaching) {
	vector<bool_arr_num_per> avail(m, {0,0,0,0,0,0,0});
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < NUM_PERIODS; j++) {
			avail[i][j] = (teachers.get(i, "per"+to_string(j+1)) > 0 && !preps[i][j]);
			if (!preps[i][j]) schedule[i][j] = 0;
		}
	}
	vvi count(NUM_PERIODS, vi(n, 0));
	vector<pii> to_assign;
	for (int i = 0; i < m; i++) {
		for (int v: classes_teaching[i]) to_assign.push_back(pii(i, v));
	}
	for (int i = 1; i < to_assign.size(); i++)
		swap(to_assign[i], to_assign[gen()%(i+1)]);
	for (pii p: to_assign) {
		int bst = -1;
		ld cost = 0;
		for (int t = 0; t < NUM_PERIODS; t++) {
			if (!avail[p.first][t]) continue;
			ld cur_cost = evaluate_spread_single(p.second, count[t][p.second]+1)
				-evaluate_spread_single(p.second, count[t][p.second]);
			if (bst == -1 || cur_cost < cost) {
				bst = t;
				cost = cur_cost;
			}
		}
		assert(bst != -1);
		avail[p.first][bst] = 0;
		schedule[p.first][bst] = encode(p.second, 0);
		count[bst][p.second]++;
	}
	while (hill_climb(preps, count)) {}
	return count;
}

void optimize_schedule(vector<bool_arr_num_per> &preps, vvi &classes_teaching) {
	for (int i = 0; i < num_hill_climbs; i++) {
		vvi count = iter(preps, classes_teaching);
		ld cur_cost = evaluate_schedule(count, classes_teaching);
		ifstream fin("schedule.txt");
		ld best_cost;
		if (!(fin >> best_cost)) best_cost = INF;
		if (cur_cost + eps < best_cost) {
			best_cost = cur_cost;
			cerr << "found " << i << ' ' << fixed << setprecision(10) << cur_cost << '\n';
			print_schedule(cur_cost);
			print_diagnostics(count, classes_teaching);
		}
	}
}

void make_matching(vector<bool_arr_num_per> &preps, vvi &is_teaching) {
	int sz = 2+m+n;
	MCMF class_matchings(sz+2);
	int src = sz-2, snk = sz-1;
	for (int i = 0; i < m; i++) {
		class_matchings.addBounded(src, i, teachers.get(i, "sections"),
			teachers.get(i, "sections"), 0, 0);
		for (int v: is_teaching[i]) {
			class_matchings.addBounded(i, m+v, 1,
				classes.get(v, "isCollegePrep") ? 3 : 4, 0, 0);
		}
	}
	for (int i = 0; i < n; i++) {
		int numSections = classes.get(i, "sections");
		class_matchings.addBounded(m+i, snk, numSections, numSections, 0, 0);
	}
	auto res = class_matchings.bounded_flow(src, snk);
	assert(res.first);
	vvi classes_teaching(m);//, periods_teaching(m);
	for (int i = 0; i < m; i++) {
		for (int v: is_teaching[i]) {
			for (int _ = 0; _ < class_matchings.get_flow(i, m+v); _++)
				classes_teaching[i].push_back(v);
		}
	}
	optimize_schedule(preps, classes_teaching/*, periods_teaching*/);
}

void evolve(ld &temp, ld amt) {
	temp = max_temp-(max_temp-temp)*(1-amt); // can change
}

int transform(int wt) {
	return (wt == 0 ? 0 : (int)(1+100*log2(wt)));
}

int main() {
	classes = csv_file(n, "classes.csv");
	teachers = csv_file(m, "teachers.csv");
	num_avail.resize(m, 0);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			teachers.set(i, classes.entry_names[j],
				transform(teachers.get(i, classes.entry_names[j])));
		}
		for (int j = 0; j < NUM_PERIODS; j++) {
			num_avail[i] += (teachers.get(i, "per"+to_string(1+j))>0);
		}
	}
	schedule.resize(m);
	for (int _ = 0; 1; _++) {
		cerr << "iter " << _ << '\n';
		bool f = 0;
		vvi is_teaching;
		ld temp = 0;
		int num = 0;
		while (!f) {
			if (!make_teacher_assignments(is_teaching, temp)) continue;
			f = check_section_feasable(is_teaching);
			// could potentially add in more checks (such as all CS teachers teaching a CS class)
			evolve(temp, 0.1);
		}
		print_matching(is_teaching);
		cerr << "matched\n";
		people_teaching = vvi(n, vi());
		for (int i = 0; i < m; i++) {
			for (int j: is_teaching[i])
				people_teaching[j].push_back(i);
		}
		f = 0;
		vector<bool_arr_num_per> preps;
		for (int __ = 0; __ < num_sub_iters; __++) {
			cerr << "subiter " << __ << '\n';
			preps = assign_preps();
			make_matching(preps, is_teaching);
		}
	}
}