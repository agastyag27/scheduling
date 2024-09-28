bool make_teacher_assignments(vvi &is_teaching, ld temp = 0) {
    MCMF teacher_class_matching(3*m+n+4, temp); // +2 because of ranges
    int source = 3*m+n, sink = 3*m+n+1;
    int ex = 0;
    is_teaching.clear();
    // --> WHAT IS N? M? THESE THINGS SHOULD HAVE NAMES THAT CLEARLY INDICATE THEIR PURPOSE IN THIS CODE
    is_teaching.resize(m);
    // --> THIS FOR LOOP IS TOO LARGE. SEPARATE INTO A SPEPARATE FUNCTION THAT HAS A MEANINGFUL NAME
    // --> ALSO THIS IS WHY ENCAPSULATION INTO A CLASS IS USEFUL SO YOU DON'T HAVE TO PASS STATE
    for (int i = 0; i < m; i++) {
        //is_teaching[i].clear();
        // --> WHY 2? THIS SHOULD BE SOME KIND OF A DEFINED CONSTANT. UNLESS 2 IS CRUCIAL FOR YOUR ALGORITHM.
        // --> IF TWO IS CRUCIAL, CALL THAT OUT IN THE NAME OF THE FUNCTION -- EG. MAKE_DYADIC_TEACHER_ASSIGNMENTS
        int lim = min(num_avail[i]-teachers.get(i, "sections"), 2);

        // --> CASE? SWITCH?
        if (teachers.get(i, "sections") <= 2) {
            teacher_class_matching.addBounded(source, i, 1, 1, 0);
            ex++;
        }
        else if (teachers.get(i, "sections") <= 3) {
            teacher_class_matching.addBounded(source, i, 1, lim, 0); // incentivize one class
            ex += 2;
        }
        else {
            teacher_class_matching.addBounded(source, i, lim, lim, 0);
            ex += 2;
        }
        teacher_class_matching.addEdge(i, m+i, 1, 0);
        teacher_class_matching.addBounded(i, 2*m+i, 1, 2, 0);
        for (int v = 0; v < n; v++) {
            string class_name = classes.entry_names[v];
            if (teachers.get(i, class_name) == 0) continue;
            // bool has_to = (teachers[i].must_teach.find(v) !=
            //  teachers[i].must_teach.end());
            if (classes.get(v, "isCollegePrep")) {
                teacher_class_matching.addBounded(m+i, 3*m+v, 0, 1,
                    11 - teachers.get(i, class_name));
                //else teacher_class_matching.addEdge(m+i, 3*m+n+v, has_to, 1, 0);
            }
            else {
                teacher_class_matching.addBounded(2*m+i, 3*m+v, 0, 1, 
                    11 - teachers.get(i, class_name));
            }
        }
    }
    for (int i = 0; i < n; i++) {
        // if (!(classes.get(i, "isCollegePrep") ||
        //      classes.get(i, "sections") <= 3)) {
        //  teacher_class_matching.addEdge(3*m+i, 3*m+n+i, 3, 0);
        //      // add cost instead of max rely?
        // }
        teacher_class_matching.addBounded(3*m+i, sink,
            classes.get(i, "minTeachers"), classes.get(i, "maxTeachers"), 0);
    }
    teacher_class_matching.bounded_flow(source, sink);
    // --> THE NEXT FOR LOOP IS TOO BIG. SEPARATE FUNCTION.
    for (int i = 0; i < m; i++) {
        // cout << teachers.entry_names[i] << ':';
        int max_secs = 0;
        for (int j = 0; j < n; j++) {
            if (teacher_class_matching.get_flow(m+i, 3*m+j) ||
                teacher_class_matching.get_flow(2*m+i, 3*m+j)) {
                is_teaching[i].push_back(j);
                // cout << ' ' << classes.entry_names[j];
                max_secs += classes.get(j, "sections");
            }
        }
        //cout << endl;
        if (max_secs < teachers.get(i, "sections")) return 0;
        // assert(max_secs >= teachers.get(i, "sections"));
    }
    return 1;
    // cout << teacher_class_matching.flow[sink][source] << '\n';
    // cout << ex << '\n';
}
