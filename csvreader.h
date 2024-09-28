#include "includes_and_constants.h"

class csv_file {
public:
	vector<string> entry_names; // name of each teacher/class
	vector<string> field_names; // name of each attribute
	map<string, int> field_to_index; 
	map<string, int> name_to_index;
	vvi data_entries;
	int sz;
	csv_file() {}
	csv_file(int &n, string filename) {
		ifstream fin(filename);
		string s;
		getline(fin, s);
		if (s.back() == '\r') s.pop_back();
		stringstream fields(s);
		string field;
		getline(fields, field, ','); assert(field == "name");
		while (getline(fields, field, ',')) {
			field_to_index[field] = field_names.size();
			field_names.push_back(field);
		}
		string dat;
		while (getline(fin, dat)) {
			stringstream cur_line(dat);
			string entry;
			getline(cur_line, entry, ',');
			name_to_index[entry] = entry_names.size();
			entry_names.push_back(entry);
			data_entries.push_back(vi());
			while (getline(cur_line, entry, ',')) {
				data_entries.back().push_back(stoi(entry));
			}
		}
		n = data_entries.size();
		fin.close();
		sz = n;
	}
	void print() {
		cout << "name";
		for (string s: field_names) cout << ',' << s;
		cout << '\n';
		for (int i = 0; i < sz; i++) {
			cout << entry_names[i];
			for (int v: data_entries[i]) cout << ',' << v;
			cout << '\n';
		}
	}
	int get(int i, string field) {
		assert(field_to_index.find(field) != field_to_index.end());
		return data_entries[i][field_to_index[field]];
	}
	void set(int i, string field, int v) {
		assert(field_to_index.find(field) != field_to_index.end());
		data_entries[i][field_to_index[field]] = v;
	}
	void check_name(string s) {
		assert(name_to_index.find(s) != name_to_index.end());
	}
	void print_names() {
		for (int i = 0; i < entry_names.size(); i++) {
			if (i) cout << ',';
			cout << entry_names[i];
		}
		cout << '\n';
	}
};

