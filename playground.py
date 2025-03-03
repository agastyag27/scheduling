import csv

def csv_to_list_of_lists(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data

def process(row):
    for index, value in enumerate(row):
        if value == "Top choice":
            row[index] = 1
        elif value == "Second choice":
            row[index] = 2
        elif value == "Third choice":
            row[index] = 3
        elif value == "Fourth choice":
            row[index] = 4
        elif value == "Fifth choice":
            row[index] = 5
        elif value == "Sixth choice":
            row[index] = 6
        elif value == "Seventh choice":
            row[index] = 7
    final = []
    name = row[1]
    final.append(name)
    sections = int(row[2])
    final.append(sections)
    periods = row[3].split(", ")
    for x in range(7):
        if str(x+1) in periods:
            final.append(1)
        else:
            final.append(0)
    preferred_number_of_classes = int(row[4])
    final.append(preferred_number_of_classes)
    preference_value = int(row[5])
    final.append(preference_value)
    for classes in row[6:-2]:
        if classes == "":
            final.append(0)
        else:
            final.append(8-classes)
    college_prep=int(row[-2])
    final.append(college_prep)
    return(",".join(str(x) for x in final))
filename = "raw.csv"
data = csv_to_list_of_lists(filename)[1:]
print("name,sections,per1,per2,per3,per4,per5,per6,per7,preferred_number_of_classes,preference_value,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12,class13,class14,class15,class16,class17,class18,class19,class20,max_college_prep")
for row in data:
    row = process(row)
    print(row)


