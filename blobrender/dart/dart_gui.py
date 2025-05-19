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
    def __init__(self, mode="light"):
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
        self.text_col="black"
        self.back_col="white"

        # generate window and call widget construction
        self.root = tk.Tk()
        self.root.geometry("1600x500")
        self.root.title("blob-render: gDART")
        self.root.configure(bg="white")
        #self.root.resizable(False, False)
        tk.Frame.__init__(self, self.root)
        self.create_widgets()

    def create_widgets(self):
        #self.frm_entry = tk.Frame(master=self.root, bg="white")


        # contstruct user input panel
        self.user_input = tk.Frame(master=self.root, bg=self.bg_col)
        self.def_args = {"master": self.user_input, "bg": self.bg_col, "fg": self.txt_col}
        hidden_labels = ["load_str", "theta", "phi", "pdim"]
        input_labels = ["Path to .npy file", "theta", "phi", "Resolution"]
        for i, hidden_label in enumerate(hidden_labels):
            setattr(self, hidden_label + "_lbl", tk.Label(text=input_labels[i], **self.def_args))
            setattr(self, hidden_label + "_ent", tk.Entry(**self.def_args))
            getattr(self, hidden_label + "_lbl").grid(row=i, column=0)
            getattr(self, hidden_label + "_ent").grid(row=i, column=1)

        # construct render button
        self.render_btn = tk.Button(text="Render", command=self.call_renderer, **self.def_args)
        self.render_btn.grid(row=len(hidden_labels), column=0)
        self.root.bind("<Return>", self.call_renderer)

        # construct output display
        img = ImageTk.PhotoImage(Image.open(self.blank_str))
        self.output_pnl = tk.Label(master=self.root, image=img, bg=self.bg_col)
        self.output_pnl.image = img # redundant?
        self.output_pnl.grid(row=0, column=5, rowspan=6,columnspan=6)

        # place frame
        self.user_input.grid(row=0, column=0) # revert to root usage? partition inp window?
        self.output_pnl.grid(row=0, column=1)

    def call_renderer(self, event=None):
        load_str = self.load_str_ent.get()
        if load_str.replace(" ","") == "":
            return # do not call renderer without real input

    def boot_gui(self):
        self.root.mainloop()

if __name__ == "__main__":

    gDART().boot_gui()