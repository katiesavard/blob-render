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
        self.prog_bar = tk.Progressbar(**self.def_args, orient=tk.HORIZONTAL, length=100, mode="determinate")
        self.prog_bar.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E)

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
        # load data from entry
        load_str = self.load_ent.get()
        theta = self.theta_ent.get()
        phi = self.phi_ent.get()
        nx = self.nx_ent.get()
        ny = self.ny_ent.get()
        save_str = self.save_ent.get()

        # parse data TODO: safe aginst large input
        if load_str.replace(" ","") == "":
            return # do not call renderer without real input
        elif not os.path.exists(load_str):
            raise Exception("unable to locate .npy file")
        
        try:
            theta = float(theta) * np.pi / 180.0
            phi = float(phi) * np.pi / 180.0
            nx = int(nx)
            ny = int(ny)
        except:
            print("Failure to parse user input, check types")

        # load data, setup scene
        mesh = dt.Mesh()
        mesh.import_pdata(load_str)

        pdim = [nx, ny]
        sdim = [0.5,0.5*ny/nx] # TODO: update with autoroutine 
        screen = dt.Screen(R=2, theta=theta, phi=phi, pdim=pdim, sdim=sdim, tilt=0)
        render = screen.render(mesh, use_bake=False, verbose=False) # TODO: add prog bar pass
        image = Image.fromarray(render).convert("L")
        draw = ImageTk.PhotoImage(image)
        self.output_pnl.configure(image=draw)
        self.output_pnl.image = drawn

    def boot_gui(self):
        self.root.mainloop()

if __name__ == "__main__":

    gDART().boot_gui()