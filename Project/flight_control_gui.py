import tkinter as tk
from tkinter import font

root = tk.Tk()
root.title("Command Based Flight Control")
root.configure(bg='#27315C')

# Custom font
btn_font = font.Font(size=10, weight='bold')

# Main Frame
frame = tk.Frame(root, bg='#27315C')
frame.pack(padx=16, pady=16)

title = tk.Label(frame, text="Command Based Flight Control", bg='#27315C', fg='white', font=('Arial', 12, 'bold'))
title.grid(row=0, column=0, columnspan=4, pady=(0,12))

# Take off and Land
btn_take_off = tk.Button(frame, text="Take off", width=12, font=btn_font)
btn_take_off.grid(row=1, column=0, rowspan=2, sticky='ew', padx=(0,13), pady=2)

btn_land = tk.Button(frame, text="Land", width=12, font=btn_font)
btn_land.grid(row=3, column=0, rowspan=2, sticky='ew', padx=(0,13), pady=2)

# Arrow and movement buttons
arrow_up = tk.Button(frame, text="↑", width=5, height=1, font=btn_font)
arrow_up.grid(row=1, column=2, pady=1)

arrow_left = tk.Button(frame, text="←", width=5, height=1, font=btn_font)
arrow_left.grid(row=2, column=1, pady=1)

arrow_right = tk.Button(frame, text="→", width=5, height=1, font=btn_font)
arrow_right.grid(row=2, column=3, pady=1)

arrow_down = tk.Button(frame, text="↓", width=5, height=1, font=btn_font)
arrow_down.grid(row=3, column=2, pady=1)

btn_up = tk.Button(frame, text="Up", width=12, font=btn_font)
btn_up.grid(row=4, column=1, columnspan=2, sticky='ew', pady=(8,2))

btn_down = tk.Button(frame, text="Down", width=12, font=btn_font)
btn_down.grid(row=4, column=2, columnspan=2, sticky='ew', padx=(8,0), pady=(8,2))

root.mainloop()
