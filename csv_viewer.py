import tkinter as tk
import json
import colorsys
import csv
from tkinter import filedialog

NUM_SCHEDULES = 5  # Number of top schedules stored

# Read the JSON data from the top_schedules.json file.
try:
    with open("top_schedules.json", "r") as f:
        schedules = json.load(f)
except FileNotFoundError:
    schedules = []

current_index = 0  # Index of currently displayed schedule

# Extract schedule data
if schedules:
    cost = schedules[current_index]["cost"]
    headers = schedules[current_index]["headers"]
    schedule = schedules[current_index]["schedule"]
    names = schedules[current_index]["names"]
else:
    cost, headers, schedule, names = 0, [], [], []

def create_table():
    # Clear existing widgets in the frame
    for widget in frame.winfo_children():
        widget.destroy()
    global data
    data = []
    # Rebuild table based on current headers and names
    for i in range(len(names) + 1):
        row_cells = []
        for j in range(len(headers)):
            label = tk.Label(frame, text="", borderwidth=1, relief="solid", width=15, height=2)
            label.grid(row=i, column=j, sticky="nsew")
            frame.grid_columnconfigure(j, weight=1)
            row_cells.append(label)
        data.append(row_cells)

def update_schedule(index):
    """ Updates the display to show the selected schedule. """
    global cost, headers, schedule, names, current_index, schedules

    try:
        with open("top_schedules.json", "r") as f:
            schedules = json.load(f)
    except FileNotFoundError:
        schedules = []

    if not schedules:
        return
    current_index = (index if index < len(schedules) else len(schedule)-1)
    cost = schedules[current_index]["cost"]
    headers = schedules[current_index]["headers"]
    schedule = schedules[current_index]["schedule"]
    names = schedules[current_index]["names"]

    create_table()
    title_label.config(text=f"Schedule {current_index + 1}")

    global class_base_hues
    unique_classes = set()
    for sched in schedules:
        for row in sched["schedule"]:
            for cell in row:
                if cell.strip():
                    unique_classes.add(cell.strip())
    unique_classes = sorted(unique_classes)
    class_base_hues = {cls: i / len(unique_classes) for i, cls in enumerate(unique_classes)}

    # Update table contents
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            text_value = ""
            if i == 0:
                text_value = headers[j] if j < len(headers) else ""
                cell.config(text=text_value, bg="lightgray", font=("Helvetica", 10, "bold"))
            elif j == 0:
                text_value = names[i - 1] if i - 1 < len(names) else ""
                cell.config(text=text_value, bg="lightgray", font=("Helvetica", 10, "bold"))
            else:
                text_value = schedule[i - 1][j - 1] if i - 1 < len(schedule) and j - 1 < len(schedule[i - 1]) else ""
                cell.config(text=text_value, bg=get_dynamic_color_for_class(text_value), font=("Helvetica", 10))

def download_csv():
    """ Prompts the user to save the current schedule as a CSV file (without the cost). """
    # Build the table data: first row is headers; subsequent rows are teacher names + schedule row.
    table = []
    table.append(headers)
    for i, teacher_name in enumerate(names):
        row = [teacher_name] + schedule[i]
        table.append(row)
    # Ask user where to save the CSV file.
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save schedule as CSV"
    )
    if file_path:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)

# Compute unique class names from non-header cells.


# Map each class name to a base hue (evenly spaced on the hue wheel).
class_base_hues = {}

def get_dynamic_color_for_class(class_name):
    """ Generates a dynamic color based on the class name. """
    if not class_name or class_name.strip() == "":
        return "white"
    hue = class_base_hues.get(class_name.strip(), -1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0 if hue == -1 else 0.2, 0.9 if hue == -1 else 0.99)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

# Create the main application window
root = tk.Tk()
root.title("Top 5 Schedules Viewer")

# Title label
title_label = tk.Label(root, text=(f"Schedule {current_index + 1}" if schedules else "No Schedules Generated Yet"), font=("Helvetica", 14, "bold"))
title_label.pack(pady=5)

# Create a frame to hold the table.
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# List to hold references to table cells
data = []

# Create an empty table
for i in range(len(names) + 1):
    row_cells = []
    for j in range(len(headers)):
        label = tk.Label(frame, text="", borderwidth=1, relief="solid", width=15, height=2)
        label.grid(row=i, column=j, sticky="nsew")
        frame.grid_columnconfigure(j, weight=1)
        row_cells.append(label)
    data.append(row_cells)

# Update the table with current schedule
update_schedule(0)

# Navigation buttons and download button
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

def prev_schedule():
    """ Show the previous schedule. """
    update_schedule((current_index - 1) % len(schedules) if schedules else 0)

def next_schedule():
    """ Show the next schedule. """
    print(current_index)
    update_schedule((current_index + 1) % len(schedules) if schedules else 0)

# Using grid to neatly arrange buttons
prev_button = tk.Button(button_frame, text="Previous", command=prev_schedule)
prev_button.grid(row=0, column=0, padx=10)

download_button = tk.Button(button_frame, text="Download CSV", command=download_csv)
download_button.grid(row=0, column=1, padx=10)

next_button = tk.Button(button_frame, text="Next", command=next_schedule)
next_button.grid(row=0, column=2, padx=10)

root.mainloop()
