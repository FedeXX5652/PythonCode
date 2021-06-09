import tkinter as tk
from tkinter import ttk

def print_text():
    print("boton 1")

def print_text2():
    print("boton 2")

def print_select():
    print("Lenguaje selecionado: ",lista.get(lista.curselection()))

def print_combo():
    print("Combo: ",combo.get())

def print_entry():
    nombre = entry.get()
    print("Entry: ", nombre)

###########################################################

ventana = tk.Tk()
ventana.title("PLACE")
ventana.config(width=800, height=600)
ventana.resizable(0, 0)
ventana.iconbitmap("ico.ico")

#boton con dimension adaptativa
boton = ttk.Button(text="Puto el que lee", command=print_text)
boton.place(x=50,y=10)

#boton con dimension fija
boton = ttk.Button(text="Puto el que lee", command=print_text2)
boton.place(x=150,y=70, width=100, height=50)
boton.config(text="Puto el que lee")

#labels
label = ttk.Label(text="haha tkinter goes brrrrrrr")
label.place(x=160, y=120)

#imagen en label/posicionamiento relativo
imagen = tk.PhotoImage(file="png.png")
label = ttk.Label(image=imagen)
label.place(relx=0.25, rely=0.25, relwidth=0.5, relheight=0.5)

#lista (listbox)
lista = tk.Listbox()
lista.insert(0, "Python", "C++", "Java")
lista.place(x=10, y=170)
boton = ttk.Button(text="Puto el que lee", command=print_select)
boton.place(x=10,y=350)

#lista (combobox)
combo = ttk.Combobox(state="readonly", values=[
                                            "Visual Studio",
                                            "Genie",
                                            "Text Doc",
                                            "Visual Studio Code",
                                            ])
combo.place(x=300, y=300)
boton = ttk.Button(text="Puto el que lee", command=print_combo)
boton.place(x=300,y=350)

#text box
entry = ttk.Entry()
entry.place(x=300, y=10)
entry.insert(0, "Insert name")
boton = ttk.Button(text="Save Name", command=print_entry)
boton.place(x=300, y=40)
boton.config()

#check button
check_state = tk.BooleanVar()
check = ttk.Checkbutton(text="option", variable=check_state)
check.place(x=300, y=100)
check_state.set("False")


ventana.mainloop()