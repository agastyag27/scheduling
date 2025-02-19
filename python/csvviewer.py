import tkinter as tk
import csv
import time

# A dictionary to store the assigned colors for each class name.
class_colors = {}
import hashlib
import colorsys

def get_color_for_class(class_name):
    # Return white if the cell is empty.
    if not class_name or class_name.strip() == "":
        return "white"
    # If the class already has an assigned color, return it.
    if class_name in class_colors:
        return class_colors[class_name]
    
    # Create a stable hash of the class name.
    hash_object = hashlib.md5(class_name.encode())
    hash_int = int(hash_object.hexdigest(), 16)*time.time()
    
    # Map the hash to a hue value between 0 and 1.
    hue = (hash_int % 360) / 360.0
    # Use fixed saturation and brightness for consistency.
    saturation = 0.2
    brightness = 0.99
    
    # Convert HSV to RGB.
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    # Convert the RGB values (0-1) to a hex string.
    color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    
    # Save and return the color.
    class_colors[class_name] = color
    return color


# Create the main application window.
root = tk.Tk()
root.title("Schedule Table")

# Read the CSV data from the schedule.txt file.
data = []
with open("schedule.txt", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
    data.pop(0)

# Create a frame to hold the table.
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Loop through the data to create and grid each label.
for i, row in enumerate(data):
    for j, cell in enumerate(row):
        # For header row, use white background and bold font.
        if i == 0 or j == 0:
            bg_color = "lightgray"
            font = ("Helvetica", 10, "bold")
        else:
            bg_color = get_color_for_class(cell)
            font = ("Helvetica", 10)
        # Create a label for this cell.
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
        # Ensure the column expands evenly.
        frame.grid_columnconfigure(j, weight=1)

root.mainloop()
