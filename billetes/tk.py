from tkinter import *
from tkinter import filedialog, messagebox
import func
  
root = Tk()
root.geometry("350x200")
root.title(" Converter ")

frame = Frame(root)
frame.pack(side=TOP)

midframe = Frame(root)
midframe.pack()


bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM )

  
def selFile():
    root.path = filedialog.askopenfilename(title="Seleccione un archivo")
    File.delete("1.0", "end-1c")
    File.insert(END, root.path)

def start(path, amount):
    print(path)
    try:
        f = open(str(path), "r")
    except:
        messagebox.showerror(title="Error", message="Error al abrir el archivo")
    lines = f.readlines()
    l = []

    for line in lines:
        if line.strip() != "" and line.strip().isnumeric():
            l.append(int(line.strip(),  base=10))
    
    try:
        func.func(l, amount)
        messagebox.showinfo(title="Archivo creado", message="El resulrado fue pueso en el archivo"+"\nCOMBINACIÓN_BILLETES.txt")
    except:
        messagebox.showerror(title="Error", message="Error al operar")
    


      
l = Label(frame, text = "Armador de pilones de ").pack(side=LEFT)

Enter = Entry(frame, 
              width = 10, 
              bg = "light green")

Enter.pack(side=LEFT)

l2 = Label(frame, text = " v1.0").pack(side=LEFT)

File = Text(midframe, height = 5, 
              width = 35, 
              bg = "light cyan")

File.pack(side=RIGHT)
  
Display = Button(bottomframe, height = 2,
                 width = 20, 
                 text ="Seleccione un archivo",
                 command = lambda:selFile()).pack(side=RIGHT, padx=5, pady=5)

Display2 = Button(bottomframe, height = 2,
                 width = 20, 
                 text ="Start",
                 command = lambda:start(File.get("1.0", "end-1c"), int(Enter.get()))).pack(side=LEFT, padx=5, pady=5)
  
root.mainloop()