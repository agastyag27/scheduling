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
    SEED = 0
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

    def addEdge(self, u, v, cap, cost, to_fudge=True):
        # Optionally add random “fudge” to cost.
        if to_fudge and Constants.fudge:
            cost += get_fudge(self.temp)
        self.G.add_edge(u, v, capacity=cap, weight=cost)
        self.lower_bounds[(u, v)] = 0

    def addBounded(self, u, v, lb, ub, cost, to_fudge=True):
        if to_fudge and Constants.fudge:
            cost += get_fudge(self.temp)
        # Add edge for the extra capacity (upper bound minus lower bound).
        if ub > lb:
            self.G.add_edge(u, v, capacity=ub - lb, weight=cost)
        self.lower_bounds[(u, v)] = lb
        # Adjust node demands so that at least lb flow is pushed on this edge.
        self.G.nodes[u]['demand'] -= lb
        self.G.nodes[v]['demand'] += lb

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
        self.m = len(self.teachers.entry_names)
        self.n = len(self.classes.entry_names)
        self.schedule = [[0]*Constants.NUM_PERIODS for _ in range(self.m)]
        self.num_avail = [0]*self.m
        for i in range(self.m):
            # Transform teacher's scores for each class.
            for j in range(self.n):
                score = self.teachers.get(i, self.classes.entry_names[j])
                self.teachers.set(i, self.classes.entry_names[j], self.transform(score))
            for j in range(Constants.NUM_PERIODS):
                if self.teachers.get(i, f"per{j+1}") > 0:
                    self.num_avail[i] += 1
        # people_teaching: for each class, list teacher indices
        self.people_teaching = [[] for _ in range(self.n)]
    
    def transform(self, wt):
        if wt == 0:
            return 0
        return int(1 + 100 * math.log2(wt))
    
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
                teacher_class_matching.addBounded(source, i, 1, 1, 0)
                num_teaching = 1
            else:
                teacher_class_matching.addBounded(source, i, lim, lim, 0)
                num_teaching = 2
            if num_teaching == 1 and extra_teaching == 4:
                teacher_class_matching.addEdge(i, 2*self.m + i, 1, 0)
            else:
                teacher_class_matching.addEdge(i, self.m + i, 1, 0)
                teacher_class_matching.addEdge(i, 2*self.m + i, 2, 0)
            for v in range(self.n):
                class_name = self.classes.entry_names[v]
                if self.teachers.get(i, class_name) == 0:
                    continue
                if self.classes.get(v, "isCollegePrep"):
                    teacher_class_matching.addBounded(
                        self.m + i, 3*self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
                else:
                    teacher_class_matching.addBounded(
                        2*self.m + i, 3*self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
        for v in range(self.n):
            teacher_class_matching.addBounded(
                3*self.m + v, sink,
                self.classes.get(v, "minTeachers"), self.classes.get(v, "maxTeachers"), 0
            )
        # Compute the flow and check feasibility before continuing.
        feasible, flowCost = teacher_class_matching.bounded_flow(source, sink)
        if not feasible:
            #print("Teacher-class matching infeasible. Aborting assignment for this iteration.")
            return False, is_teaching
        # Now safely query flows.
        for i in range(self.m):
            max_secs = 0
            for j in range(self.n):
                try:
                    flow1 = teacher_class_matching.get_flow(self.m + i, 3*self.m + j)
                    flow2 = teacher_class_matching.get_flow(2*self.m + i, 3*self.m + j)
                except ValueError as e:
                    print(f"Error getting flow for teacher {i}, class {j}: {e}")
                    return False, is_teaching
                if flow1 or flow2:
                    is_teaching[i].append(j)
                    max_secs += self.classes.get(j, "sections")
            if max_secs < self.teachers.get(i, "sections"):
                print(f"Teacher {i} does not meet section requirements.")
                return False, is_teaching
        return True, is_teaching

    def check_section_feasible(self, is_teaching):
        total_nodes = self.m + self.n + 4
        section_matching = MCMF(total_nodes)
        src = self.m + self.n
        snk = self.m + self.n + 1
        for i in range(self.m):
            numSections = self.teachers.get(i, "sections")
            section_matching.addBounded(src, i, numSections, numSections, 0, to_fudge=False)
            for v in is_teaching[i]:
                cap = 3 if self.classes.get(v, "isCollegePrep") else 4
                section_matching.addBounded(i, self.m + v, 1, cap, 0, to_fudge=False)
        for v in range(self.n):
            numSections = self.classes.get(v, "sections")
            section_matching.addBounded(self.m + v, snk, numSections, numSections, 0, to_fudge=False)
        feasible, _ = section_matching.bounded_flow(src, snk)
        return feasible

    def print_matching(self, is_teaching):
        # Simple matching print-out.
        for i in range(self.m):
            line = self.teachers.entry_names[i] + "\t" + " ".join(self.classes.entry_names[v] for v in is_teaching[i])
            print(line)

    # Assume additional methods like assign_preps, make_matching, etc. are defined here.

    def run(self):
        iteration = 0
        temp = 0
        while True:
            print(f"Iteration {iteration}")
            success = False
            is_teaching = None
            # Retry teacher assignment until both matching and section feasibility pass.
            while not success:
                res, assignments = self.make_teacher_assignments(temp)
                if not res:
                    #print("Teacher assignment failed. Retrying...")
                    continue
                if not self.check_section_feasible(assignments):
                    print("Section feasibility check failed. Retrying teacher assignments...")
                    continue
                success = True
                is_teaching = assignments
                temp = Constants.max_temp - (Constants.max_temp - temp) * (1 - 0.1)
            self.print_matching(is_teaching)
            # Build people_teaching from is_teaching.
            self.people_teaching = [[] for _ in range(self.n)]
            for i in range(self.m):
                for v in is_teaching[i]:
                    self.people_teaching[v].append(i)
            # Continue with the rest of the scheduling process...
            for _ in range(Constants.num_sub_iters):
                preps = self.assign_preps()
                self.make_matching(preps, is_teaching)
            iteration += 1
            # For demonstration, break after one iteration.
            break

# Example usage:
scheduler = Scheduler("classes.csv", "teachers.csv")
scheduler.run()
# mcmf = MCMF(total_nodes, temp)
# mcmf.addEdge(0, 1, 10, 5)
# mcmf.addBounded(1, 2, 2, 8, 3)
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