# import tkinter as tk
# from tkinter import filedialog

# from numpy import left_shift
# import func

# class Application(tk.Frame):
#     def __init__(self, master=None):
#         super().__init__(master)
#         self.master = master
#         self.pack()
#         self.create_widgets()

#     def create_widgets(self):

#         self.label = tk.Label(self, text="Hello world")
#         self.label.pack(side="top")

#         self.entry = tk.Entry()
#         self.entry.pack(side="top", expand=True, fill=tk.BOTH)

#         self.hi_there = tk.Button(self)
#         self.hi_there["text"] = "Seleccione un archivo"
#         self.hi_there["command"] = self.say_hi
#         self.hi_there.pack(side="right")

#         self.start = tk.Button(self)
#         self.start["text"] = "Seleccione un archivo"
#         self.start["command"] = self.startFunc
#         self.start.pack(side="left")

#         self.quit = tk.Button(self, text="QUIT", fg="red",
#                               command=self.master.destroy)
#         self.quit.pack(side="bottom", after=self.start)

#     def say_hi(self):
#         file = filedialog.askopenfile(title="Seleccione un archivo")
   
#     def startFunc(self):
#         print("funcion acá")

# root = tk.Tk()
# app = Application(master=root)
# app.mainloop()

from tkinter import *
from tkinter import filedialog
  
root = Tk()
root.geometry("600x250")
root.title(" Converter ")
  
def selFile(File):
    root.path = filedialog.askopenfilename(title="Seleccione un archivo")
    my_string_var.set(root.path)

def start(path):
    print(path)

      
l = Label(text = "Armador de pilones de 10000 v1.0").pack()

my_string_var = StringVar()

my_string_var.set("File Path")

File = Label(root,
                bg = "light yellow", textvariable = my_string_var).pack()
  
Display = Button(root, height = 2,
                 width = 20, 
                 text ="Seleccione un archivo",
                 command = lambda:selFile(File)).pack()

Display2 = Button(root, height = 2,
                 width = 20, 
                 text ="Start",
                 command = lambda:start(my_string_var.get())).pack()
  
root.mainloop()