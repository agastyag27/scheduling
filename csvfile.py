import csv

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
            print(name + "," + (",".join(map(str, self.data_entries[i])) if self.data_entries[i] else "prep"))

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