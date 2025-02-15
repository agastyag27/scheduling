import csv
import math
import random
import heapq
import networkx as nx

# Constants from includes_and_constants.h
class Constants:
    NUM_PERIODS = 7
    eps = 1e-5
    max_temp = 20
    INF = 10**18  # a very large number
    SEED = 1
    fudge = True
    spread_weight = 5
    class_weight = 20
    period_weight = 1
    num_sub_iters = 10
    num_hill_climbs = 100
    zero_cost_weight = 170
    classes_out_of_range_penalty = 100

random.seed(Constants.SEED)

def get_fudge(temp):
    if temp < Constants.eps:
        return 0
    a = 0
    g = random.randint(0, 2**32 - 1)
    while g - (temp+1) * (g // (temp+1)) < temp:
        a += 1
        g = random.randint(0, 2**32 - 1)
    return a

# CSVFile class – conversion of csvreader.h
class CSVFile:
    def __init__(self, filename):
        self.entry_names = []
        self.field_names = []
        self.field_to_index = {}
        self.name_to_index = {}
        self.data_entries = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            # remove trailing carriage returns if any
            header = [h.rstrip('\r') for h in header]
            assert header[0] == "name", "First field must be 'name'"
            self.field_names = header[1:]
            for idx, field in enumerate(self.field_names):
                self.field_to_index[field] = idx
            for row in reader:
                if not row: 
                    continue
                row = [x.rstrip('\r') for x in row]
                name = row[0]
                self.entry_names.append(name)
                self.name_to_index[name] = len(self.entry_names) - 1
                # convert remaining fields to int
                self.data_entries.append([int(x) for x in row[1:]])
        self.sz = len(self.data_entries)

    def print(self):
        print("name," + ",".join(self.field_names))
        for i, name in enumerate(self.entry_names):
            print(name + "," + ",".join(map(str, self.data_entries[i])))

    def get(self, i, field):
        assert field in self.field_to_index, f"Field {field} not found"
        return self.data_entries[i][self.field_to_index[field]]

    def set(self, i, field, value):
        assert field in self.field_to_index, f"Field {field} not found"
        self.data_entries[i][self.field_to_index[field]] = value

    def check_name(self, s):
        assert s in self.name_to_index, f"Name {s} not found"

    def print_names(self):
        print(",".join(self.entry_names))

# MCMF class – conversion of mcmf.h
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

# Scheduler class – combines the teacher–assignment (from make_teacher_assignments.h)
# and the scheduling/hill–climb optimization (from schedule.cpp)
class Scheduler:
    def __init__(self, classes_file, teachers_file):
        self.classes = CSVFile(classes_file)
        self.teachers = CSVFile(teachers_file)
        self.m = len(self.teachers.entry_names)  # number of teachers
        self.n = len(self.classes.entry_names)   # number of classes
        self.schedule = [[0]*Constants.NUM_PERIODS for _ in range(self.m)]
        self.num_avail = [0]*self.m
        for i in range(self.m):
            # Transform teacher’s scores for each class.
            for j in range(self.n):
                score = self.teachers.get(i, self.classes.entry_names[j])
                self.teachers.set(i, self.classes.entry_names[j], self.transform(score))
            for j in range(Constants.NUM_PERIODS):
                if self.teachers.get(i, f"per{j+1}") > 0:
                    self.num_avail[i] += 1
        # people_teaching: for each class, list teacher indices teaching that class.
        self.people_teaching = [[] for _ in range(self.n)]
    
    def transform(self, wt):
        if wt == 0:
            return 0
        return int(1 + 100 * math.log2(wt))
    
    def encode(self, v, is_prep):
        # Returns a coded value for class v. In C++:
        # return is_prep ? -v-1 : v+1;
        return -v-1 if is_prep else v+1
    
    def make_teacher_assignments(self, temp=0):
        is_teaching = [[] for _ in range(self.m)]
        total_nodes = 3*self.m + self.n + 4
        teacher_class_matching = MCMF(total_nodes, temp)
        source = 3*self.m + self.n
        sink = 3*self.m + self.n + 1
        for i in range(self.m):
            extra_teaching = self.teachers.get(i, "sections")
            if extra_teaching == 0:
                continue
            lim = min(self.num_avail[i], 2)
            num_teaching = 0
            if (extra_teaching <= 2 or 
                (extra_teaching == 3 and random.choice([True, False])) or 
                (extra_teaching == 4 and random.randint(0, 3) == 0)):
                teacher_class_matching.add_bounded(source, i, 1, 1, 0)
                num_teaching = 1
            else:
                teacher_class_matching.add_bounded(source, i, lim, lim, 0)
                num_teaching = 2
            if num_teaching == 1 and extra_teaching == 4:
                teacher_class_matching.add_edge(i, 2*self.m + i, 1, 0)
            else:
                teacher_class_matching.add_edge(i, self.m + i, 1, 0)
                teacher_class_matching.add_edge(i, 2*self.m + i, 2, 0)
            for v in range(self.n):
                class_name = self.classes.entry_names[v]
                if self.teachers.get(i, class_name) == 0:
                    continue
                if self.classes.get(v, "isCollegePrep"):
                    teacher_class_matching.add_bounded(
                        self.m + i, 3*self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
                else:
                    teacher_class_matching.add_bounded(
                        2*self.m + i, 3*self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
        for v in range(self.n):
            teacher_class_matching.add_bounded(
                3*self.m + v, sink,
                self.classes.get(v, "minTeachers"), self.classes.get(v, "maxTeachers"), 0
            )
        feasible, flowCost = teacher_class_matching.bounded_flow(source, sink)
        if not feasible:
            #print("Teacher-class matching infeasible. Aborting assignment for this iteration.")
            return False, is_teaching
        for i in range(self.m):
            max_secs = 0
            for j in range(self.n):
                try:
                    flow1 = teacher_class_matching.get_flow(self.m + i, 3*self.m + j)
                    flow2 = teacher_class_matching.get_flow(2*self.m + i, 3*self.m + j)
                except ValueError as e:
                    #print(f"Error getting flow for teacher {i}, class {j}: {e}")
                    return False, is_teaching
                if flow1 or flow2:
                    is_teaching[i].append(j)
                    max_secs += self.classes.get(j, "sections")
            if max_secs < self.teachers.get(i, "sections"):
                #print(f"Teacher {i} does not meet section requirements.")
                return False, is_teaching
        return True, is_teaching

    def check_section_feasible(self, is_teaching):
        total_nodes = self.m + self.n + 4
        section_matching = MCMF(total_nodes)
        src = self.m + self.n
        snk = self.m + self.n + 1
        for i in range(self.m):
            numSections = self.teachers.get(i, "sections")
            section_matching.add_bounded(src, i, numSections, numSections, 0, to_fudge=False)
            for v in is_teaching[i]:
                cap = 3 if self.classes.get(v, "isCollegePrep") else 4
                section_matching.add_bounded(i, self.m + v, 1, cap, 0, to_fudge=False)
        for v in range(self.n):
            numSections = self.classes.get(v, "sections")
            section_matching.add_bounded(self.m + v, snk, numSections, numSections, 0, to_fudge=False)
        feasible, _ = section_matching.bounded_flow(src, snk)
        return feasible

    def assign_preps(self):
        # Build a preps table: one list per teacher (length = NUM_PERIODS) indicating if that period is used for prep.
        preps = [[False] * Constants.NUM_PERIODS for _ in range(self.m)]
        # Build list of classes that require common prep.
        requires_prep = []
        for v in range(self.n):
            # If class v needs common prep and more than one teacher is teaching it.
            if self.classes.get(v, "needsCommonPrep") and len(self.people_teaching[v]) > 1:
                requires_prep.append(v)
        # Shuffle the list randomly.
        random.shuffle(requires_prep)
        # For each class that requires prep...
        for v in requires_prep:
            # For each period, check if all teachers teaching class v are available and not already prepping.
            avail = [True] * Constants.NUM_PERIODS
            for i in self.people_teaching[v]:
                for j in range(Constants.NUM_PERIODS):
                    # A teacher is available in period j if his "per" score is positive and he isn’t already assigned a prep.
                    avail[j] = avail[j] and (self.teachers.get(i, f"per{j+1}") > 0 and not preps[i][j])
            # Collect available periods.
            possible_periods = [j for j, available in enumerate(avail) if available]
            if not possible_periods:
                print(self.classes.entry_names[v])
                raise Exception("No available period for class " + self.classes.entry_names[v])
            prep = random.choice(possible_periods)
            # Assign this period as a prep period for every teacher teaching class v.
            for i in self.people_teaching[v]:
                preps[i][prep] = True
                self.schedule[i][prep] = self.encode(v, True)
        return preps

    def optimize_schedule(self, preps, classes_teaching):
        # [This method is assumed to perform hill–climb optimization over the schedule.]
        # For brevity, we assume its implementation is similar to the earlier translation.
        best_cost = float('inf')
        for _ in range(Constants.num_hill_climbs):
            count = self.iterate_schedule(preps, classes_teaching)
            cur_cost = self.evaluate_schedule(count, classes_teaching)
            if cur_cost + Constants.eps < best_cost:
                best_cost = cur_cost
                self.print_schedule(cur_cost)
                # Optionally, print diagnostics.
    
    def iterate_schedule(self, preps, classes_teaching):
        # [A simplified translation of the iter(...) function from C++]
        avail = [[(self.teachers.get(i, f"per{j+1}") > 0 and not preps[i][j])
                  for j in range(Constants.NUM_PERIODS)]
                 for i in range(self.m)]
        # Reset schedule for non-prep periods.
        for i in range(self.m):
            for j in range(Constants.NUM_PERIODS):
                if not preps[i][j]:
                    self.schedule[i][j] = 0
        # Initialize count matrix: periods x classes.
        count = [[0]*self.n for _ in range(Constants.NUM_PERIODS)]
        to_assign = []
        for i in range(self.m):
            for v in classes_teaching[i]:
                to_assign.append((i, v))
        random.shuffle(to_assign)
        for i, v in to_assign:
            best_period = None
            best_cost = float('inf')
            for t in range(Constants.NUM_PERIODS):
                if not avail[i][t]:
                    continue
                cur_cost = self.evaluate_spread_single(v, count[t][v]+1) - self.evaluate_spread_single(v, count[t][v])
                if best_period is None or cur_cost < best_cost:
                    best_period = t
                    best_cost = cur_cost
            if best_period is None:
                raise Exception("No available period found")
            avail[i][best_period] = False
            self.schedule[i][best_period] = self.encode(v, False)
            count[best_period][v] += 1
        while self.hill_climb(preps, count):
            pass
        return count

    def evaluate_spread_single(self, cla, cnt):
        if cla == -1:
            return 0
        return (cnt*cnt / self.classes.get(cla, "sections")) + (3 if cnt == 0 else 0)

    def evaluate_schedule(self, count, classes_teaching):
        return (Constants.spread_weight * self.evaluate_spread(count) +
                Constants.period_weight * self.evaluate_period_assignments() +
                Constants.class_weight * self.evaluate_class_assignments(classes_teaching))

    def evaluate_spread(self, count):
        total = 0
        for t in range(Constants.NUM_PERIODS):
            for v in range(self.n):
                total += self.evaluate_spread_single(v, count[t][v])
        return total

    def evaluate_period_assignments(self):
        total = 0
        for i in range(self.m):
            for j in range(Constants.NUM_PERIODS):
                if self.schedule[i][j]:
                    total += self.get_point_period_cost(i, j)
        return total

    def get_point_period_cost(self, i, j):
        return 5 - self.teachers.get(i, f"per{j+1}")

    def evaluate_class_assignments(self, classes_teaching):
        total = 0
        for i in range(self.m):
            for v in classes_teaching[i]:
                total += Constants.zero_cost_weight - self.teachers.get(i, self.classes.entry_names[v])
        return total

    def hill_climb(self, preps, count):
        found = False
        for i in range(self.m):
            improved = True
            while improved:
                improved = False
                for j in range(Constants.NUM_PERIODS):
                    if preps[i][j] or self.teachers.get(i, f"per{j+1}") == 0:
                        continue
                    for k in range(j+1, Constants.NUM_PERIODS):
                        if preps[i][k] or self.teachers.get(i, f"per{k+1}") == 0 or self.schedule[i][j] == self.schedule[i][k]:
                            continue
                        a = self.schedule[i][j]
                        b = self.schedule[i][k]
                        # Decode the assigned classes.
                        a_dec = -a-1 if a < 0 else a-1
                        b_dec = -b-1 if b < 0 else b-1
                        cur_cost = (Constants.spread_weight * (self.evaluate_spread_single(a_dec, count[j][a_dec]) +
                                    self.evaluate_spread_single(b_dec, count[k][b_dec]) +
                                    self.evaluate_spread_single(a_dec, count[k][a_dec]) +
                                    self.evaluate_spread_single(b_dec, count[j][b_dec])) +
                                    Constants.period_weight * ((self.get_point_period_cost(i, j) if self.schedule[i][j] else 0) +
                                                               (self.get_point_period_cost(i, k) if self.schedule[i][k] else 0)))
                        next_cost = (Constants.spread_weight * (self.evaluate_spread_single(a_dec, count[j][a_dec]-1) +
                                     self.evaluate_spread_single(b_dec, count[k][b_dec]-1) +
                                     self.evaluate_spread_single(a_dec, count[k][a_dec]+1) +
                                     self.evaluate_spread_single(b_dec, count[j][b_dec]+1)) +
                                     Constants.period_weight * ((self.get_point_period_cost(i, j) if self.schedule[i][k] else 0) +
                                                                (self.get_point_period_cost(i, k) if self.schedule[i][j] else 0)))
                        if cur_cost > next_cost + Constants.eps:
                            if a_dec != -1:
                                count[j][a_dec] -= 1
                                count[k][a_dec] += 1
                            if b_dec != -1:
                                count[j][b_dec] += 1
                                count[k][b_dec] -= 1
                            self.schedule[i][j], self.schedule[i][k] = self.schedule[i][k], self.schedule[i][j]
                            improved = True
                            found = True
        return found

    def make_matching(self, preps, is_teaching):
        sz = 2 + self.m + self.n
        class_matchings = MCMF(sz + 2, temp=0)
        src = sz - 2
        snk = sz - 1
        for i in range(self.m):
            sections = self.teachers.get(i, "sections")
            class_matchings.add_bounded(src, i, sections, sections, 0, to_fudge=False)
            for v in is_teaching[i]:
                cap = 3 if self.classes.get(v, "isCollegePrep") else 4
                class_matchings.add_bounded(i, self.m + v, 1, cap, 0, to_fudge=False)
        for i in range(self.n):
            numSections = self.classes.get(i, "sections")
            class_matchings.add_bounded(self.m + i, snk, numSections, numSections, 0, to_fudge=False)
        feasible, _ = class_matchings.bounded_flow(src, snk)
        if not feasible:
            raise Exception("Class matching flow infeasible.")
        classes_teaching = [[] for _ in range(self.m)]
        for i in range(self.m):
            for v in is_teaching[i]:
                flow_val = int(class_matchings.get_flow(i, self.m + v))
                for _ in range(flow_val):
                    classes_teaching[i].append(v)
        self.optimize_schedule(preps, classes_teaching)

    def print_schedule(self, cost):
        print(f"Schedule cost: {cost}")
        header = "name," + ",".join([f"per{j+1}" for j in range(Constants.NUM_PERIODS)])
        print(header)
        for i in range(self.m):
            line = self.teachers.entry_names[i]
            for j in range(Constants.NUM_PERIODS):
                line += f", {self.get_class(self.schedule[i][j])}"
            print(line)

    def get_class(self, v):
        if v == 0:
            return ""
        return self.classes.entry_names[-v-1] + " prep" if v < 0 else self.classes.entry_names[v-1]

    def run(self):
        iteration = 0
        temp = 0
        while True:
            print(f"Iteration {iteration}")
            success = False
            is_teaching = None
            while not success:
                res, assignments = self.make_teacher_assignments(temp)
                if not res:
                    #print("Teacher assignment failed. Retrying...")
                    continue
                if not self.check_section_feasible(assignments):
                    #print("Section feasibility check failed. Retrying teacher assignments...")
                    continue
                success = True
                is_teaching = assignments
                temp = Constants.max_temp - (Constants.max_temp - temp) * (1 - 0.1)
            self.print_matching(is_teaching)
            self.people_teaching = [[] for _ in range(self.n)]
            for i in range(self.m):
                for v in is_teaching[i]:
                    self.people_teaching[v].append(i)
            for _ in range(Constants.num_sub_iters):
                preps = self.assign_preps()
                self.make_matching(preps, is_teaching)
            iteration += 1
            # For demonstration, break after one iteration.
            break

    def print_matching(self, is_teaching):
        for i in range(self.m):
            line = self.teachers.entry_names[i] + "\t" + " ".join(self.classes.entry_names[v] for v in is_teaching[i])
            print(line)

# Example usage:
scheduler = Scheduler("classes.csv", "teachers.csv")
scheduler.run()
# mcmf = MCMF(total_nodes, temp)
# mcmf.add_edge(0, 1, 10, 5)
# mcmf.add_bounded(1, 2, 2, 8, 3)
# feasible, cost = mcmf.bounded_flow(source, sink)
# if not feasible:
#     print("Flow network is infeasible; please check your constraints.")
# else:
#     print("Total cost:", cost)
#     print("Flow from 0 to 1:", mcmf.get_flow(0, 1))
"""  
In this Python version:
• The CSVFile class (from csvreader.h citeturn0file0) reads a CSV with a header beginning with “name.”
• The Constants class (from includes_and_constants.h citeturn0file1) holds parameters and type–equivalent values.
• The MCMF class (from mcmf.h citeturn0file3) implements a min–cost max–flow algorithm.
• The Scheduler class (combining make_teacher_assignments.h citeturn0file2 and schedule.cpp citeturn0file4) contains methods for teacher–assignment, checking feasibility, scheduling (including hill–climb optimization), and running the full process.

This “translation” preserves much of the structure and logic of the original C++ code while using Python’s classes and standard libraries. (Note that real–world use would require further testing and debugging.)
"""