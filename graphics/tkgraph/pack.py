import tkinter as tk
from tkinter import ttk


#############################################
#side=tk.RIGHT/LEFT
#before/after=objeto
#ipadx/ipady son tamaños maximos relativos al tamaño de la pantalla
#expand
#fill, rellena el objeto hacia una direccion

ventana = tk.Tk()
ventana.title("GRID")

entry = ttk.Entry()
entry.pack()

b1 = ttk.Button(text="Button")
b1.pack(padx=30, pady=30, ipadx=50, ipady=80)

label = ttk.Label(text="Hello world")
label.pack(before=entry)

entry = ttk.Entry()
entry.pack(after=b1, expand=True, fill=tk.BOTH)

label = ttk.Label(text="Hello world2")
label.pack(after=b1)

ventana.mainloop()