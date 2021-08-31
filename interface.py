from tkinter import *

root = Tk()


# root.geometry("900x600")

# defining the functions
def exit_program():
    root.quit()


# defining the elements of the GUI
label_1 = Label(root, text="Lower cutoff frequency")
label_2 = Label(root, text="Higher cutoff frequency")
label_3 = Label(root, text="Volume")

s1 = Scale(root, variable=DoubleVar,
           from_=20, to=20000,
           orient=HORIZONTAL)

s2 = Scale(root, variable=DoubleVar,
           from_=10000, to=70000,
           orient=HORIZONTAL)

s3 = Scale(root, variable=DoubleVar,
           from_=1, to=100,
           orient=HORIZONTAL)

button = Button(root, text="Exit Program", padx=40, command=exit_program)


# Placing th options on the interface
label_1.grid(row=0, column=0)
label_2.grid(row=0, column=1)
label_3.grid(row=0, column=2)

s1.grid(row=1, column=0)
s2.grid(row=1, column=1)
s3.grid(row=1, column=2)

button.grid(row=2, column=0, columnspan=3)

root.mainloop()
