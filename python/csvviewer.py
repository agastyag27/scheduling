import tkinter as tk
import csv
import time
import colorsys

# Read the CSV data from the schedule.txt file.
data = []
with open("schedule.txt", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
    # If the first row is a header row, you may remove it:
    data.pop(0)

# Compute the unique class names from non-header cells.
unique_classes = set()
# Assume the first row and first column are headers.
for i, row in enumerate(data):
    for j, cell in enumerate(row):
        if i != 0 and j != 0 and cell.strip():
            unique_classes.add(cell.strip())
unique_classes = sorted(list(unique_classes))

# Map each class name to a base hue (equally spaced on the hue wheel).
class_base_hues = {}
N = len(unique_classes)
for index, cls in enumerate(unique_classes):
    base_hue = index / N  # Equally spaced in the range [0,1)
    class_base_hues[cls] = base_hue

def get_dynamic_color_for_class(class_name, offset):
    # Return white if the cell is empty.
    if not class_name or class_name.strip() == "":
        return "white"
    # Look up the base hue for this class.
    base_hue = class_base_hues.get(class_name.strip(), 0)
    # Add a time-based offset to animate the color.
    hue = (base_hue + offset) % 1.0
    saturation = 0.2
    brightness = 0.99
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

# Create the main application window.
root = tk.Tk()
root.title("Schedule Table")

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
            bg_color = get_dynamic_color_for_class(cell, 0)
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
    # Define a period (in seconds) for a full hue rotation.
    offset = (time.time()/100) % 1.0
    for label, cell in animated_cells:
        new_color = get_dynamic_color_for_class(cell, offset)
        label.config(bg=new_color)
    # Schedule the next update after 50 milliseconds.
    root.after(50, update_colors)

update_colors()  # Start the animation loop.
root.mainloop()
