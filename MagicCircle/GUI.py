import tkinter as tk
from tkinter import ttk
from tkinter import *
import json
import draw
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(MODULE_DIR, "settings.json"), "r") as f:
    settings:dict = json.load(f)
    lvl_nodes = 1 if settings.get("max_lvl") else 0
    form_nodes = len(settings.get("forms")) if settings.get("forms") else 0
    effects = 0
    for school in settings.get("effects").keys():
        print(school, len(settings.get("effects").get(school)))
        if len(settings.get("effects").get(school)) > effects:
            effects = (len(settings.get("effects").get(school)))
    
    # n is the highest number of nodes that the graph can have
    n = lvl_nodes if (lvl_nodes > form_nodes) else (form_nodes if (form_nodes > effects) else effects)
    draw.n = n
    print(f"lvl_nodes: {lvl_nodes}, form_nodes: {form_nodes}, effects: {effects}, n: {n}")


root = tk.Tk()
root.minsize(400, 500)

draw.draw(-1, "", "", "", "", 0, "circular", True, True, 300)

# Create the labels for each field.
level_label = ttk.Label(root, text="Level:")
school_label = ttk.Label(root, text="School:")
dmg_type_label = ttk.Label(root, text="Damage Type:")
area_type_label = ttk.Label(root, text="Area Type:")
range_label = ttk.Label(root, text="Range:")
radius_label = ttk.Label(root, text="Line Radius:")
graph_label = ttk.Label(root, text="Graph Type:")
image = tk.PhotoImage(file=os.path.join(MODULE_DIR, "fig.png"))
node_label:Checkbutton = ttk.Checkbutton(root, text="Node Label")
edge_color:Checkbutton = ttk.Checkbutton(root, text="Edge Color")
node_size_label = ttk.Label(root, text="Node Size:")

# set checkboxs to selected by default
node_label.selection_clear()
edge_color.selection_clear()

lbls = [level_label, school_label, dmg_type_label,
                  area_type_label, range_label]
for lbl in lbls:
    
    # add ■ to the left of the label in red color
    c = ["black", "blue", "green", "red", "#FDDA0D", "orange", "purple", "pink"].__getitem__(lbls.index(lbl))
    lbl.config(foreground=c, text="■ " + lbl.cget("text"))


schools = ["abjuration", "conjuration", "divination", "enchantment", "evocation", "illusion", "necromancy", "transmutation"]

dmg_types = ["acid", "bludgeoning", "cold", "damage", "extra", "fire", "force", "lightning", "necrotic", "nonmagical", "piercing", "poison", "psychic", "radiant", "slashing", "thunder"]

area_types = ["circle", "cone/sphere", "cone",
                  "cube", "cylinder", "line", "multiple targets/sphere",
                  "multiple targets", "none", "single target/cone",
                  "single target/cube", "single target/multiple targets", "single target/sphere",
                  "single target/wall", "single target", "sphere/cylinder",
                  "sphere", "square", "wall"]

ranges = ["10ft radius", "100ft line", "15ft cone", "15ft cube",
            "15ft radius", "30ft cone", "30ft line", "30ft radius",
            "5ft radius", "60ft cone", "60ft line", "1mi point",
            "10ft point", "1000ft point", "120ft point", "150ft point",
            "30ft point", "300ft point", "5ft point", "500ft point",
            "60ft point", "90ft point", "self", "sight",
            "special", "touch"]

figures = ["circular","kamada_kawai","random","shell","planar","spring"]

# Create the entry boxes for each field and pack them and pin to left side
level_entry = ttk.Entry(root)
school_entry = ttk.Combobox(root, values=schools)
dmg_type_entry = ttk.Combobox(root, values=dmg_types)
area_type_entry = ttk.Combobox(root, values=area_types)
range_entry = ttk.Combobox(root, values=ranges)
radius_entry = ttk.Entry(root)
graph_entry = ttk.Combobox(root, values=figures)
node_size_entry = ttk.Entry(root)

# Create the buttons to submit and clear the form.
submit_button = ttk.Button(root, text="Submit")
clear_button = ttk.Button(root, text="Clear", command=lambda: [
    level_entry.delete(0, tk.END),
    school_entry.delete(0, tk.END),
    dmg_type_entry.delete(0, tk.END),
    area_type_entry.delete(0, tk.END),
    range_entry.delete(0, tk.END),
    radius_entry.delete(0, tk.END),
    graph_entry.delete(0, tk.END),
    node_label.selection_clear(),
    edge_color.selection_clear()
])

def validate_level(widget):
    # Get the text from the entry box.
    text = widget.get()

    # Check if the text is a number from 1 to 10.
    if text.isdigit() and int(text) in range(1, 11):
        return True
    else:
        widget.bell()
        return False

    # Set the focus back to the entry box.
    widget.focus_set()

def validate_radius(widget):
    # Get the text from the entry box.
    text = widget.get()

    # Check if the text is a number from 1 to 10.
    if text.isdigit():
        return True
    else:
        widget.bell()
        return False

    # Set the focus back to the entry box.
    widget.focus_set()

def submit_button_clicked():
    # Get the values from the entry boxes.
    level = int(level_entry.get())
    school = school_entry.get()
    dmg_type = dmg_type_entry.get()
    area_type = area_type_entry.get()
    range = range_entry.get()
    radius = float(radius_entry.get()) if radius_entry.get() != "" else 0
    graph = graph_entry.get()

    # Draw the spell effect.
    draw.draw(level, school, dmg_type, area_type, range, radius, graph, node_label.instate(["selected"]), edge_color.instate(["selected"]),
              float(node_size_entry.get())*100 if node_size_entry.get() != "" else 300)

    image.configure(file=os.path.join(MODULE_DIR, "fig.png"))

# Validate the entry box.
level_entry.config(validate="key", validatecommand=validate_level(level_entry))
radius_entry.config(validate="key", validatecommand=validate_radius(radius_entry))
submit_button.config(command=submit_button_clicked)

try:
    image_label = tk.Label(root, image=image)
    image_label.pack(side="bottom")
except:
    pass

# Pack the labels and entry boxes.
level_label.pack()
level_entry.pack()
school_label.pack()
school_entry.pack()
dmg_type_label.pack()
dmg_type_entry.pack()
area_type_label.pack()
area_type_entry.pack()
range_label.pack()
range_entry.pack()
radius_label.pack()
radius_entry.pack()
graph_label.pack()
graph_entry.pack()
node_label.pack()
edge_color.pack()
node_size_label.pack()
node_size_entry.pack()
submit_button.pack()
clear_button.pack()

root.mainloop()