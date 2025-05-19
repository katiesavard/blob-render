#from dart import *
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from pathlib import Path
import os

class gDART(tk.Frame):
    """
    Create a tkinter frame for handling GUI presentation of DART functionality.
    """
    def __init__(self, mode="dark"):
        if "dark" in mode.lower(): # convert to kwarg?
            self.txt_col = "white"
            self.bg_col = "black"
        elif "light" in mode.lower():
            self.txt_col = "black"
            self.bg_col = "white"
        else:
            raise Exception("unable to parse mode in gDART constructor.")
        
        # define statics
        self.blank_str = os.path.join(Path(__file__).resolve().parent,"blank.png") # todo: protect this

        # generate window and call widget construction
        self.root = tk.Tk()
        self.root.geometry("500x500")
        self.root.title("blob-render: gDART")
        self.root.configure(bg=self.bg_col)
        self.root.resizable(False, False)
        tk.Frame.__init__(self, self.root)
        self.create_widgets()

    def create_widgets(self):

        # contstruct user input panel
        self.user_input = tk.Frame(master=self.root, bg=self.bg_col)
        self.def_args = {"master": self.user_input, "bg": self.bg_col, "fg": self.txt_col}

        self.load_lbl = tk.Label(text="Load Path", **self.def_args)
        self.load_lbl.grid(row=0, column=0)
        self.load_ent = tk.Entry(**self.def_args)
        self.load_ent.grid(row=0, column=1, columnspan=2, sticky=tk.W+tk.E)
        self.pos_lbl = tk.Label(text="(theta, phi)",**self.def_args)
        self.pos_lbl.grid(row=1, column=0)
        self.theta_ent = tk.Entry(**self.def_args)
        self.theta_ent.grid(row=1, column=1)
        self.phi_ent = tk.Entry(**self.def_args)
        self.phi_ent.grid(row=1, column=2)
        self.res_lbl = tk.Label(text="(nx, ny)", **self.def_args)
        self.res_lbl.grid(row=2, column=0)
        self.nx_ent = tk.Entry(**self.def_args)
        self.nx_ent.grid(row=2, column=1)
        self.ny_ent = tk.Entry(**self.def_args)
        self.ny_ent.grid(row=2, column=2)
        self.save_lbl = tk.Label(text="Save path", **self.def_args)
        self.save_lbl.grid(row=3, column=0)
        self.save_ent = tk.Entry(**self.def_args)
        self.save_ent.grid(row=3, column=1, columnspan=2,sticky=tk.W+tk.E)

        # construct output display
        img = ImageTk.PhotoImage(Image.open(self.blank_str))
        self.output_pnl = tk.Label(master=self.root, image=img, bg=self.bg_col)
        self.output_pnl.image = img # redundant?
        self.output_pnl.grid(row=0, column=0)

        # place frame
        self.user_input.grid(row=0, column=0)
        self.output_pnl.grid(row=1, column=0)

        # add call functionality
        self.root.bind("<Return>", self.call_renderer)
        self.user_input.bind("<Return>", self.call_renderer)
        self.output_pnl.bind("<Return>", self.call_renderer)

    def call_renderer(self, event=None):
        load_str = self.laod_ent.get()
        if load_str.replace(" ","") == "":
            return # do not call renderer without real input
        print("called")

    def boot_gui(self):
        self.root.mainloop()

if __name__ == "__main__":

    gDART().boot_gui()