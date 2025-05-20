import dart as dt
import numpy as np
import tkinter as tk
# from tkinter.tkk import Progressbar
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
        
        # define initials
        self.blank_str = os.path.join(Path(__file__).resolve().parent,"blank.png") # todo: depreciate this
        self.render = None
        self.cur_zoom = 1
        self.cur_exp = 1

        # generate window and call widget construction
        self.root = tk.Tk()
        self.root.geometry("650x200")
        self.root.title("blob-render: gDART")
        self.root.configure(bg=self.bg_col)
        #self.root.resizable(False, False)
        tk.Frame.__init__(self, self.root)
        self.create_widgets()

    def create_widgets(self):

        # contstruct user input panel
        self.dart_input = tk.Frame(master=self.root, bg=self.bg_col)
        self.dart_args = {"master": self.dart_input, "bg": self.bg_col, "fg": self.txt_col}

        self.user_lbl = tk.Label(text="Render Properties", **self.dart_args)
        self.user_lbl.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E)
        self.load_lbl = tk.Label(text="Load Path", **self.dart_args)
        self.load_lbl.grid(row=1, column=0)
        self.load_ent = tk.Entry(**self.dart_args)
        self.load_ent.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E)
        self.pos_lbl = tk.Label(text="(theta, phi)",**self.dart_args)
        self.pos_lbl.grid(row=2, column=0)
        self.theta_ent = tk.Entry(**self.dart_args)
        self.theta_ent.grid(row=2, column=1)
        self.phi_ent = tk.Entry(**self.dart_args)
        self.phi_ent.grid(row=2, column=2)
        self.res_lbl = tk.Label(text="(nx, ny)", **self.dart_args)
        self.res_lbl.grid(row=3, column=0)
        self.nx_ent = tk.Entry(**self.dart_args)
        self.nx_ent.grid(row=3, column=1)
        self.ny_ent = tk.Entry(**self.dart_args)
        self.ny_ent.grid(row=3, column=2)
        self.rdr_btn = tk.Button(**self.dart_args, text="Render", command=self.call_renderer)
        self.rdr_btn.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E)

        self.post_input = tk.Frame(master=self.root, bg=self.bg_col)
        self.post_args = {"master": self.post_input, "bg": self.bg_col, "fg": self.txt_col}

        self.post_lbl = tk.Label(text="Post-Processing Properties", **self.post_args)
        self.post_lbl.grid(row=0, column=0, columnspan=2)
        self.zoom_lbl = tk.Label(text="Zoom", **self.post_args)
        self.zoom_lbl.grid(row=1, column=0)
        self.zoom_ent = tk.Entry(**self.post_args)
        self.zoom_ent.grid(row=1,column=1)
        self.exp_lbl = tk.Label(text="Exposure", **self.post_args)
        self.exp_lbl.grid(row=2, column=0)
        self.exp_ent = tk.Entry(**self.post_args)
        self.exp_ent.grid(row=2,column=1)
        self.save_lbl = tk.Label(text="Save path", **self.post_args)
        self.save_lbl.grid(row=3, column=0)
        self.save_ent = tk.Entry(**self.post_args)
        self.save_ent.grid(row=3, column=1, columnspan=2,sticky=tk.W+tk.E)
        self.post_btn = tk.Button(**self.post_args, text="Post-Process", command=self.call_postproc)
        self.post_btn.grid(row=4,column=0, columnspan=2, stick=tk.W+tk.E)

        # self.prog_bar = Progressbar(**self.def_args, orient=tk.HORIZONTAL, length=100, mode="determinate")
        # self.prog_bar.grid(row=4, column=1, columnspan=2, sticky=tk.W+tk.E)

        # set default values
        self.load_ent.insert(0, "/scratch/render/interpolated_frame_gamm7_early_287.npy")
        self.theta_ent.insert(0,"90")
        self.phi_ent.insert(0,"45")
        self.nx_ent.insert(0,"250")
        self.ny_ent.insert(0,"250")
        #self.save_ent.insert(0, "/scratch/render/output.png")
        self.zoom_ent.insert(0, "1")
        self.exp_ent.insert(0,"1")

        # place frame
        self.dart_input.grid(row=0, column=0)
        self.post_input.grid(row=0, column=1)

        # add call functionality
        self.root.bind("<Return>", self.call_renderer)
        self.dart_input.bind("<Return>", self.call_renderer)
        self.post_input.bind("<Return>", self.call_postproc)

    def resize(self):
        dart_width = self.dart_input.winfo_width()
        dart_height = self.dart_input.winfo_height()
        post_width = self.post_input.winfo_width()
        post_height = self.post_input.winfo_height()

        output_width = self.render_dims[1] * self.cur_zoom
        output_height = self.render_dims[0] * self.cur_zoom
        total_width = np.max([dart_width + post_width, output_width])
        total_height = np.max([dart_height,post_height]) + output_height
        self.root.geometry(str(int(total_width)) + "x" + str(int(total_height)))
        self.root.update()

    def call_postproc(self):
        if self.render is None: return

        user_exp = float(self.exp_ent.get())
        if user_exp != self.cur_exp:
            base = 255 * self.render / np.max(self.render)
            scaled = base * user_exp
            scaled[scaled > 255] = 255
            self.image = Image.fromarray(scaled).convert("L")
            draw = ImageTk.PhotoImage(self.image)
            self.output_pnl.configure(image=draw)
            self.output_pnl.image = draw
            self.cur_exp = user_exp
            self.cur_zoom = 1

        user_zoom = float(self.zoom_ent.get())
        if user_zoom != self.cur_zoom:
            img_width = int(self.render_dims[1] * user_zoom)
            img_height = int(self.render_dims[0] * user_zoom)
            self.image = self.image.resize((img_width, img_height))
            draw = ImageTk.PhotoImage(self.image)
            self.output_pnl.configure(image=draw)
            self.output_pnl.image = draw
            self.cur_zoom = user_zoom

        self.resize()
        self.root.update()

    def call_renderer(self, event=None):
        # load data from entry
        load_str = self.load_ent.get()
        theta = self.theta_ent.get()
        phi = self.phi_ent.get()
        nx = self.nx_ent.get()
        ny = self.ny_ent.get()
        self.cur_zoom = 1
        self.cur_exp = 1

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
        sdim = [0.866,0.866*ny/nx] # TODO: update with autoroutine
        screen = dt.Screen(R=2, theta=theta, phi=phi, pdim=pdim, sdim=sdim, tilt=0)
        self.render = screen.render(mesh, use_bake=False, verbose=True) # TODO: add prog bar pass
        self.render_dims = np.shape(self.render)
        self.image = Image.fromarray(255 * self.render / np.max(self.render)).convert("L")
        draw = ImageTk.PhotoImage(self.image)
        if not hasattr(self, "output_pnl"):
            self.output_pnl = tk.Label(master=self.root, image=draw, bg=self.bg_col)
            self.output_pnl.image = draw # redundant?
            self.output_pnl.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E)
            self.output_pnl.bind("<Return>", self.call_renderer)
        else:
            self.output_pnl.configure(image=draw)
            self.output_pnl.image = draw
        self.resize()
        self.call_postproc()


    def boot_gui(self):
        self.root.mainloop()

if __name__ == "__main__":

    gDART().boot_gui()