import importlib
import tkinter as tk
import traceback
from tkinter import ttk

import command as cmd

def reimport():
    importlib.reload(cmd)

def error_decorator(command_func):
    def decorated_command():
        try:
            command_func()
        except Exception as e:
            traceback.print_exception(e)

    return decorated_command

def show_panel():
    def _quit():
        root.quit()
        root.destroy()

    # Create the main Tkinter window
    root = tk.Tk()
    root.geometry("300x400")
    root.protocol("WM_DELETE_WINDOW", _quit)
    root.title("Control panel")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)  # Рядки розтягуються по вертикалі
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(3, weight=1)
    root.rowconfigure(4, weight=1)
    root.rowconfigure(5, weight=1)
    root.rowconfigure(6, weight=1)
    root.rowconfigure(7, weight=1)
    root.rowconfigure(8, weight=1)
    root.rowconfigure(9, weight=1)

    command1_button = ttk.Button(root, text="Chart for current index", command=error_decorator(cmd.command1))
    command1_button.grid(row=0, column=0, sticky="nsew")

    command2_button = ttk.Button(root, text="Simulator info", command=error_decorator(cmd.command2))
    command2_button.grid(row=1, column=0, sticky="nsew")

    command3_button = ttk.Button(root, text="Observe", command=error_decorator(cmd.command3))
    command3_button.grid(row=2, column=0, sticky="nsew")

    command4_button = ttk.Button(root, text="Perform", command=error_decorator(cmd.command4))
    command4_button.grid(row=3, column=0, sticky="nsew")

    command5_button = ttk.Button(root, text="Skip", command=error_decorator(cmd.command5))
    command5_button.grid(row=4, column=0,sticky="nsew")

    command6_button = ttk.Button(root, text="Analyze", command=error_decorator(cmd.command6))
    command6_button.grid(row=5, column=0, sticky="nsew")

    command7_button = ttk.Button(root, text="Skip 200", command=error_decorator(cmd.command7))
    command7_button.grid(row=6, column=0, sticky="nsew")

    command8_button = ttk.Button(root, text="Reset simulator", command=error_decorator(cmd.command8))
    command8_button.grid(row=7, column=0, sticky="nsew")

    command9_button = ttk.Button(root, text="Update data", command=error_decorator(cmd.command9))
    command9_button.grid(row=8, column=0, sticky="nsew")

    reimport_button = ttk.Button(root, text="Reimport", command=error_decorator(reimport))
    reimport_button.grid(row=9, column=0, sticky="nsew")

    root.mainloop()


if __name__ == "__main__":
    show_panel()