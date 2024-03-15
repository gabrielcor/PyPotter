import tkinter as tk
from tkinter import ttk

def slide():
    value = slider.get()
    label.config(text=f"Slider Value: {value}")

# Create the main window
root = tk.Tk()
root.title("Slider Example")

# Create a label to display the slider value
label = tk.Label(root, text="Slider Value: ")
label.pack()

# Create the slider
slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", command=slide)
slider.pack()

# Run the Tkinter event loop
root.mainloop()
