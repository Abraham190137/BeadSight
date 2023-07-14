import tkinter as tk
import time

def update_time():
    current_time = time.strftime('%H:%M:%S.') + str(int(time.time() % 1 * 1000)).zfill(3)
    time_label.config(text=current_time)
    time_label.after(1, update_time)

# Create the main window
window = tk.Tk()
window.title("Current Time")
window.geometry("200x50")

# Create a label to display the time
time_label = tk.Label(window, font=("Helvetica", 24))
time_label.pack(pady=10)

# Start updating the time
update_time()

# Start the main loop
window.mainloop()