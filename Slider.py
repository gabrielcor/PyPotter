import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk

def slide():
    value = slider.get()
    label.config(text=f"Slider Value: {value}")

# Create the main window
root = tk.Tk()
root.title("Examples")


if False:
    # SLIDER
    # Create a label to display the slider value
    label = tk.Label(root, text="Slider Value: ")
    label.pack()

    # Create the slider
    slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", command=slide)
    slider.pack()


#CIRCLE

# Make the window full-screen and remove the title bar
root.attributes('-fullscreen', True)
root.overrideredirect(True)  # This removes the title bar and makes the window borderless

# Create a canvas widget with a black background
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), bg='black')
canvas.pack()

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

# Initial setup
image_size = (250, 250)
oval_position = [100, 100, 150, 150]
outline_color = "#D2B48C"  # Light brown
outline_rgb = hex_to_rgb(outline_color)  # Convert hex to RGB
fill_color = "black"
glow_width = 10  # Adjust the glow size

# Create a new PIL image with a specific size and background color
image = Image.new("RGBA", image_size, (0, 0, 0, 0 ))
draw = ImageDraw.Draw(image, "RGBA")

# Function to add glow by drawing multiple outlines
def add_glow(position, glow_width, outline_color):
    r, g, b = outline_rgb
    for i in range(1, glow_width + 1):
        # Calculate the new size
        new_position = [position[0] - i, position[1] - i, position[2] + i, position[3] + i]
        # Decrease opacity as the glow gets larger
        alpha = int(255 * (1 - i / glow_width))
        draw.ellipse(new_position, outline=(r, g, b, alpha))

# Draw the glow
add_glow(oval_position, glow_width, outline_color)

# Draw the original oval on top of the glow
draw.ellipse(oval_position, outline=outline_color, fill=fill_color, width=4)


# Convert the PIL image to a format Tkinter canvas can use
tk_image = ImageTk.PhotoImage(image)

# Put the PIL image on the Tkinter canvas
canvas.create_image(0, 0, anchor="nw", image=tk_image)

# Place a "?" mark in the middle of the canvas
canvas.create_text(125, 125, text="?", font=("Papyrus", 35), fill=outline_color)




# Run the Tkinter event loop
root.mainloop()
