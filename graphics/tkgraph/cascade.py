import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def nuevo():
    print("menu/archivo/nuevo")

def abrir():
    print("menu/archivo/abrir")

def guardar():
    print("menu/archivo/guardar")

def ayuda():
    print("menu/ayuda/ayuda")

def acerca():
    print("menu/ayuda/acerca")

def contacto():
    print("menu/ayuda/contacto")

#######################################
ventana = tk.Tk()
ventana.title("CASCADE")
menu_bar = tk.Menu()
ventana.config(width=300, height=200, menu=menu_bar)

#menu archivo
menu_archivo = tk.Menu(menu_bar, tearoff=0)
menu_archivo.add_command(label="Nuevo", command=nuevo)
menu_archivo.add_command(label="Abrir", command=abrir)
menu_archivo.add_command(label="Guardar", command=guardar)

#menu ayuda
menu_ayuda = tk.Menu(menu_bar, tearoff=0)
menu_ayuda.add_command(label="Ayuda", command=ayuda)
menu_ayuda.add_command(label="Acerca de", command=acerca)
menu_ayuda.add_command(label="Contacto", command=contacto)

#cascada
menu_bar.add_cascade(label="Archivo", menu=menu_archivo)
menu_bar.add_cascade(label="Ayuda", menu=menu_ayuda)

#pop ups
messagebox.showinfo(title="Info", message="pop up info")
messagebox.showerror(title="ERROR", message="pop up error")
messagebox.showwarning(title="Warning", message="pop up warning")

messagebox.askokcancel(title="Pregunta", message="Desea salir?")
messagebox.askretrycancel(title="Operacion fallida", message="Desea reintentar?")

ventana.mainloop()