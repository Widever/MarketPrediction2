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
    root.geometry("300x600")
    root.protocol("WM_DELETE_WINDOW", _quit)
    root.title("Control panel")

    root.columnconfigure(0, weight=1)
    for n_row in range(19):
        root.rowconfigure(n_row, weight=1)  # Рядки розтягуються по вертикалі

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

    command10_button = ttk.Button(root, text="Benchmark small", command=error_decorator(cmd.command10))
    command10_button.grid(row=9, column=0, sticky="nsew")

    command11_button = ttk.Button(root, text="Benchmark big", command=error_decorator(cmd.command11))
    command11_button.grid(row=10, column=0, sticky="nsew")

    command12_button = ttk.Button(root, text="Go to next sl", command=error_decorator(cmd.command12))
    command12_button.grid(row=11, column=0, sticky="nsew")

    command13_button = ttk.Button(root, text="Go to prev sl", command=error_decorator(cmd.command13))
    command13_button.grid(row=12, column=0, sticky="nsew")

    command14_button = ttk.Button(root, text="Go to next perform", command=error_decorator(cmd.command14))
    command14_button.grid(row=13, column=0, sticky="nsew")

    command15_button = ttk.Button(root, text="Go to prev perform", command=error_decorator(cmd.command15))
    command15_button.grid(row=14, column=0, sticky="nsew")

    command16_button = ttk.Button(root, text="Go to start", command=error_decorator(cmd.command16))
    command16_button.grid(row=15, column=0, sticky="nsew")

    command17_button = ttk.Button(root, text="Jump n", command=error_decorator(cmd.command17))
    command17_button.grid(row=16, column=0, sticky="nsew")

    command18_button = ttk.Button(root, text="Event stats", command=error_decorator(cmd.command18))
    command18_button.grid(row=17, column=0, sticky="nsew")

    command19_button = ttk.Button(root, text="Optimize", command=error_decorator(cmd.command19))
    command19_button.grid(row=18, column=0, sticky="nsew")

    reimport_button = ttk.Button(root, text="Reimport", command=error_decorator(reimport))
    reimport_button.grid(row=19, column=0, sticky="nsew")

    root.mainloop()


if __name__ == "__main__":
    show_panel()