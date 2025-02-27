import tkinter as tk
import json
import colorsys

# Read the JSON data from the schedule.json file.
with open("schedule.json", "r") as f:
    data_json = json.load(f)

# Extract the cost, headers, schedule, and teacher names.
cost = data_json["cost"]
headers = data_json["headers"]
schedule = data_json["schedule"]
names = data_json["names"]

# Combine the teacher names with their schedule rows.
# The resulting table's first row is the header.
data = [headers]
for teacher_name, sched_row in zip(names, schedule):
    data.append([teacher_name] + sched_row)

# Compute unique class names from non-header cells.
unique_classes = set()
for i, row in enumerate(data):
    for j, cell in enumerate(row):
        # Skip header row and teacher names column.
        if i != 0 and j != 0 and cell.strip():
            unique_classes.add(cell.strip())
unique_classes = sorted(unique_classes)

# Map each class name to a base hue (evenly spaced on the hue wheel).
class_base_hues = {}
N = len(unique_classes)
for index, cls in enumerate(unique_classes):
    base_hue = index / N
    class_base_hues[cls] = base_hue

def get_dynamic_color_for_class(class_name):
    # Return white if the cell is empty.
    if not class_name or class_name.strip() == "":
        return "white"
    # Look up the base hue for this class.
    hue = class_base_hues.get(class_name.strip(), 0)
    saturation = 0.2
    brightness = 0.99
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

# Create the main application window and include the cost in the title.
root = tk.Tk()
root.title(f"Schedule Table - Cost: {cost}")

# Create a frame to hold the table.
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# List to hold references to animated (non-header) labels.
animated_cells = []

# Loop through the data to create and grid each label.
for i, row in enumerate(data):
    for j, cell in enumerate(row):
        if i == 0 or j == 0:
            bg_color = "lightgray"
            font = ("Helvetica", 10, "bold")
        else:
            bg_color = get_dynamic_color_for_class(cell)
            font = ("Helvetica", 10)
        label = tk.Label(frame,
                         text=cell,
                         borderwidth=1,
                         relief="solid",
                         bg=bg_color,
                         fg="black",
                         font=font,
                         width=15,
                         height=2)
        label.grid(row=i, column=j, sticky="nsew")
        frame.grid_columnconfigure(j, weight=1)
        # Save non-header labels for animation.
        if i != 0 and j != 0:
            animated_cells.append((label, cell))

def update_colors():
    # Update the colors of the animated cells.
    for label, cell in animated_cells:
        new_color = get_dynamic_color_for_class(cell)
        label.config(bg=new_color)
    # Schedule the next update after 50 milliseconds.
    root.after(50, update_colors)

update_colors()  # Start the animation loop.
root.mainloop()
