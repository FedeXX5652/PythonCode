import tkinter as tk
from tkinter import ttk


#############################################
#con el metodo grid no podes saltear columnas ni filas, tenes que declarar los espacios vacios

ventana = tk.Tk()
ventana.title("GRID")

entry = ttk.Entry()
entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

#sticky ancla un objeto a una direccion (NSEW, coordenadas cartesianas en ingles)
#pad ancla la distancia entre objetos

b1 = ttk.Button(text="press")
b1.grid(row=0, column=1, rowspan=2)

label = ttk.Label(text="Hello world")
label.grid(row=2, column=0, columnspan=2)

ventana.columnconfigure(0, weight =1)
ventana.rowconfigure(0, weight =1)

ventana.mainloop()