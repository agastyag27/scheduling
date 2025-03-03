import random
import math
import time
import os
import multiprocessing
import json
import copy

from constants import Constants
from csvfile import CSVFile
from mcmf import MCMF

class Scheduler:
    def __init__(self, classes_file, teachers_file, shared_hashes, shared_top_schedules, is_master=False):
        random.seed(time.time() * os.getpid())
        self.classes = CSVFile(classes_file)
        self.teachers = CSVFile(teachers_file)
        self.m = len(self.teachers.entry_names)
        self.n = len(self.classes.entry_names)
        self.schedule = [[0] * Constants.NUM_PERIODS for _ in range(self.m)]
        self.num_avail = [0] * self.m
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
        # Shared objects for caching assignments and schedules.
        self.shared_hashes = shared_hashes       # a Manager.dict() used as a set (keys are the hashes)
        self.shared_top_schedules = shared_top_schedules  # a Manager.list() of schedule dictionaries
        self.is_master = is_master

    def transform(self, wt):
        if wt == 0:
            return 0
        return int(1 + 100 * math.log2(wt))

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

    def encode(self, v, is_prep):
        return -v - 1 if is_prep else v + 1

    def decode(self, encoded): 
        if encoded < 0:
            return -encoded - 1, True
        else:
            return encoded - 1, False

    def make_matching(self, preps, is_teaching):
        sz = 2 + self.m + self.n
        class_matchings = MCMF(sz + 2, temp=Constants.max_temp)
        src = sz - 2
        snk = sz - 1
        for i in range(self.m):
            sections = self.teachers.get(i, "sections")
            class_matchings.add_bounded(src, i, sections, sections, 0)
            for v in is_teaching[i]:
                cap = 3 if self.classes.get(v, "isCollegePrep") else 4
                class_matchings.add_bounded(i, self.m + v, 1, cap, 0)
        for i in range(self.n):
            numSections = self.classes.get(i, "sections")
            class_matchings.add_bounded(self.m + i, snk, numSections, numSections, 0)
        feasible, _ = class_matchings.bounded_flow(src, snk)
        if not feasible:
            raise Exception("Class matching flow infeasible.")
        classes_teaching = [[] for _ in range(self.m)]
        for i in range(self.m):
            for v in is_teaching[i]:
                flow_val = int(class_matchings.get_flow(i, self.m + v))
                for _ in range(flow_val):
                    classes_teaching[i].append(v)
        return self.optimize_schedule(preps, classes_teaching)

    def get_best_cost(self):
        # Use the shared_top_schedules instead of reading a file.
        schedules = list(self.shared_top_schedules)
        if not schedules or len(schedules) < Constants.NUM_SCHEDULES:
            return float('inf')
        return max(s['cost'] for s in schedules)  # Return the worst (highest) cost

    def optimize_schedule(self, preps, classes_teaching):
        best_schedule = []
        best_cost = float('inf')
        for i in range(Constants.num_hill_climbs):
            count = self.iterate_schedule(preps, classes_teaching)
            cur_cost = self.evaluate_schedule(count, classes_teaching)
            if cur_cost + Constants.eps < best_cost:
                best_cost = cur_cost
                best_schedule = copy.deepcopy(self.schedule)
        return best_schedule, best_cost

    def print_diagnostics(self, count, classes_teaching):
        spread_score = self.evaluate_spread(count)
        period_score = self.evaluate_period_assignments()
        class_score = self.evaluate_class_assignments(classes_teaching)
        diagnostics = (
            f"Raw Spread: {spread_score}\n"
            f"Adj. Spread: {Constants.spread_weight * spread_score}\n"
            f"Raw Period Score: {period_score}\n"
            f"Adj. Period Score: {Constants.period_weight * period_score}\n"
            f"Raw Class Score: {class_score}\n"
            f"Adj. Class Score: {Constants.class_weight * class_score}\n"
            f"\nTotal Cost: {self.evaluate_schedule(count, classes_teaching)}\n"
        )
        print(diagnostics)
        with open("schedule_diagnostics.txt", "w") as f:
            f.write(diagnostics)

    def iterate_schedule(self, preps, classes_teaching):
        avail = [
            [
                self.teachers.get(i, f"per{j+1}") > 0 and not preps[i][j]
                for j in range(Constants.NUM_PERIODS)
            ]
            for i in range(self.m)
        ]
    
        for i in range(self.m):
            for j in range(Constants.NUM_PERIODS):
                if not preps[i][j]:
                    self.schedule[i][j] = 0
    
        count = [[0] * self.n for _ in range(Constants.NUM_PERIODS)]
    
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
                current_cost = self.evaluate_spread_single(v, count[t][v] + 1) - self.evaluate_spread_single(v, count[t][v])
                if best_period is None or current_cost < best_cost:
                    best_period = t
                    best_cost = current_cost
            if best_period is None:
                raise Exception("No available period found for teacher assignment")
            avail[i][best_period] = False
            self.schedule[i][best_period] = self.encode(v, False)
            count[best_period][v] += 1
    
        while self.hill_climb(preps, count):
            pass
        return count
    
    def evaluate_spread_single(self, cla, cnt):
        if cla == -1:
            return 0
        sections = self.classes.get(cla, "sections")
        penalty = 3 if cnt == 0 else 0
        return (cnt * cnt / sections) + penalty

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
                        a, _ = self.decode(self.schedule[i][j])
                        b, _ = self.decode(self.schedule[i][k])
                        current_cost = (
                            Constants.spread_weight * (
                                self.evaluate_spread_single(a, count[j][a] if a != -1 else 0) +
                                self.evaluate_spread_single(b, count[k][b] if b != -1 else 0) +
                                self.evaluate_spread_single(a, count[k][a] if a != -1 else 0) +
                                self.evaluate_spread_single(b, count[j][b] if b != -1 else 0)
                            )
                            + Constants.period_weight * (
                                (self.get_point_period_cost(i, j) if self.schedule[i][j] != 0 else 0) +
                                (self.get_point_period_cost(i, k) if self.schedule[i][k] != 0 else 0)
                            )
                        )
                        next_cost = (
                            Constants.spread_weight * (
                                self.evaluate_spread_single(a, (count[j][a] if a != -1 else 0) - 1) +
                                self.evaluate_spread_single(b, (count[k][b] if b != -1 else 0) - 1) +
                                self.evaluate_spread_single(a, (count[k][a] if a != -1 else 0) + 1) +
                                self.evaluate_spread_single(b, (count[j][b] if b != -1 else 0) + 1)
                            )
                            + Constants.period_weight * (
                                (self.get_point_period_cost(i, j) if self.schedule[i][k] != 0 else 0) +
                                (self.get_point_period_cost(i, k) if self.schedule[i][j] != 0 else 0)
                            )
                        )
                        if current_cost > Constants.eps + next_cost:
                            if a != -1:
                                count[j][a] -= 1
                                count[k][a] += 1
                            if b != -1:
                                count[j][b] += 1
                                count[k][b] -= 1
                            self.schedule[i][j], self.schedule[i][k] = self.schedule[i][k], self.schedule[i][j]
                            improved = True
                            found = True
                            break
                    if improved:
                        break
        return found
    
    def get_point_period_cost(self, i, j):
        return 5 - self.teachers.get(i, f"per{j+1}")
    
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

    def evaluate_class_assignments(self, classes_teaching):
        total = 0
        for i in range(self.m):
            for v in classes_teaching[i]:
                total += Constants.zero_cost_weight - self.teachers.get(i, self.classes.entry_names[v])
        return total
    
    def print_schedule(self, next_schedule, cost):
        # Build schedule data using the shared_top_schedules.
        headers = ["Name"] + [f"Period {j+1}" for j in range(Constants.NUM_PERIODS)]
        schedule_with_names = [[self.get_class(entry) for entry in row] for row in next_schedule]
        new_schedule = {
            "cost": cost,
            "headers": headers,
            "schedule": schedule_with_names,
            "names": self.teachers.entry_names
        }
        # Update the shared top schedules.
        schedules = list(self.shared_top_schedules)
        schedules.append(new_schedule)
        schedules = sorted(schedules, key=lambda x: x["cost"])[:Constants.NUM_SCHEDULES]
        # Replace the shared list with the new sorted schedules.
        self.shared_top_schedules[:] = schedules
        if self.is_master:
            with open("top_schedules.json", "w") as f:
                json.dump(schedules, f)

    def get_class(self, v):
        if v == 0:
            return ""
        return self.classes.entry_names[-v-1] + " prep" if v < 0 else self.classes.entry_names[v-1]

    def hash_teacher_class_assignments(self, is_teaching):
        return hash(str(is_teaching))
    
    def make_teacher_assignments(self, temp=0):
        is_teaching = [[] for _ in range(self.m)]
        total_nodes = 3 * self.m + self.n + 2
        teacher_class_matching = MCMF(total_nodes, temp)
        source = 3 * self.m + self.n
        sink = 3 * self.m + self.n + 1
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
                teacher_class_matching.add_edge(i, 2 * self.m + i, 1, 0)
            else:
                teacher_class_matching.add_edge(i, self.m + i, 1, 0)
                teacher_class_matching.add_edge(i, 2 * self.m + i, 2, 0)
            for v in range(self.n):
                class_name = self.classes.entry_names[v]
                if self.teachers.get(i, class_name) == 0:
                    continue
                if self.classes.get(v, "isCollegePrep"):
                    teacher_class_matching.add_bounded(
                        self.m + i, 3 * self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
                else:
                    teacher_class_matching.add_bounded(
                        2 * self.m + i, 3 * self.m + v, 0, 1,
                        Constants.zero_cost_weight - self.teachers.get(i, class_name)
                    )
        for v in range(self.n):
            teacher_class_matching.add_bounded(
                3 * self.m + v, sink,
                self.classes.get(v, "minTeachers"), self.classes.get(v, "maxTeachers"), 0
            )
        feasible, flowCost = teacher_class_matching.bounded_flow(source, sink)
        if not feasible:
            return False, is_teaching
        for i in range(self.m):
            max_secs = 0
            for j in range(self.n):
                try:
                    flow1 = teacher_class_matching.get_flow(self.m + i, 3 * self.m + j)
                    flow2 = teacher_class_matching.get_flow(2 * self.m + i, 3 * self.m + j)
                except ValueError as e:
                    return False, is_teaching
                if flow1 or flow2:
                    is_teaching[i].append(j)
                    max_secs += self.classes.get(j, "sections")
            if max_secs < self.teachers.get(i, "sections"):
                return False, is_teaching
        return True, is_teaching

    def check_section_feasible(self, is_teaching):
        total_nodes = self.m + self.n + 2
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

    def print_matching(self, is_teaching):
        for i in range(self.m):
            line = self.teachers.entry_names[i] + "\t" + " ".join(self.classes.entry_names[v] for v in is_teaching[i])
            print(line)

    def get_wait_time(self, cache_size):
        return 1 + (cache_size // 200)

    def run(self):
        iteration = 0
        temp = 0
        # Local temporary set for assignment hashes before updating the shared cache.
        temp_hashes = set()
        # Instead of reading from a file, initialize current_hashes from the shared dictionary.
        current_hashes = set(self.shared_hashes.keys())
        delay = self.get_wait_time(len(current_hashes))
        while True:
            print(f"Iteration {iteration}")
            success = False
            is_teaching = None

            while not success:
                if delay == 0:
                    # Merge the temporary hashes into the shared cache.
                    for h in temp_hashes:
                        self.shared_hashes[h] = True
                    temp_hashes.clear()
                    # If this process is master, update the external file.
                    if self.is_master:
                        with open("hashed_assignments.txt", "w") as f:
                            f.write(" ".join([str(x) for x in self.shared_hashes.keys()]))
                    delay = self.get_wait_time(len(self.shared_hashes))
                    # Refresh current_hashes.
                    current_hashes = set(self.shared_hashes.keys())

                res, assignments = self.make_teacher_assignments(temp)
                assignment_hash = self.hash_teacher_class_assignments(assignments)
                delay -= 1

                if not res or assignment_hash in current_hashes:
                    continue
                temp_hashes.add(assignment_hash)
                if not self.check_section_feasible(assignments):
                    temp = Constants.max_temp - (Constants.max_temp - temp) * (1 - 0.1)
                    continue
                success = True
                is_teaching = assignments

            # Build people_teaching from is_teaching.
            self.people_teaching = [[] for _ in range(self.n)]
            for i in range(self.m):
                for v in is_teaching[i]:
                    self.people_teaching[v].append(i)
                    
            best_schedule = []
            best_cost = float('inf')

            for _ in range(Constants.num_sub_iters):
                preps = self.assign_preps()
                final_schedule, final_cost = self.make_matching(preps, is_teaching)
                if final_cost + Constants.eps < best_cost:
                    best_schedule = final_schedule
                    best_cost = final_cost

            cutoff_cost = self.get_best_cost()
            if best_cost + Constants.eps < cutoff_cost:
                self.print_schedule(best_schedule, best_cost)

            iteration += 1

def run_scheduler(shared_hashes, shared_top_schedules, is_master=False):
    scheduler = Scheduler("classes.csv", "teachers.csv", shared_hashes, shared_top_schedules, is_master)
    scheduler.run()

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    # Shared dictionary: keys are assignment hashes.
    shared_hashes = manager.dict()
    # Shared list of top schedules.
    shared_top_schedules = manager.list()

    processes = []
    num_procs = multiprocessing.cpu_count()
    for idx in range(num_procs):
        # Designate the first process as master.
        is_master = (idx == 0)
        p = multiprocessing.Process(target=run_scheduler, args=(shared_hashes, shared_top_schedules, is_master))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
