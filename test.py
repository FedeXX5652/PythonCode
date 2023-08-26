from tkinter import *
task_list=["Call","Work","Help"]
root=Tk()
Label(root,text="My Tasks").place(x=5,y=0)
placement=20
for tasks in task_list:
    Checkbutton(root,text=str(tasks)).place(x=5,y=placement)
    placement+=20
root.mainloop()