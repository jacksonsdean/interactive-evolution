from cProfile import label
import copy
import json
import time
from tkinter import Tk, filedialog

from matplotlib import pyplot as plt
import numpy as np
from config import Config
from cppn import CPPN
from PIL import Image
from image_utils import make_weight_video, save_image
from tkvideo import tkvideo
from visualize import visualize_network
from PIL import ImageTk
import tkinter as tk

def load_from_file(filename, res_h=None, res_w=None) -> CPPN:
    with open(filename, 'r') as f:
        data = json.load(f)
        f.close()
        config = Config.create_from_json(data['config'])
        if res_h is not None:
            config.res_h = res_h
        if res_w is not None:
            config.res_w = res_w
        
        if 'cxs' in data:
            # OLD WAY:
            nodes = data['nodes']
            connections = data['cxs']
            cppn = CPPN(config)
            cppn.construct_from_lists(nodes, connections) 
            i = cppn 
        else:
            i = CPPN.create_from_json(data, config)
    return i 

class GUI:
    def __init__(self, root):
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.individual = CPPN(Config())

    def run(self):
        self.main_window()
        self.update_title()
        try:
            self.root.mainloop()
        except MemoryError:
            print("Memory Error")
            self.root.mainloop()
        except np.core._exceptions._ArrayMemoryError:
            print("Memory Error")
            self.root.mainloop()
   
    def weight_gif (self):
        ind = self.individual
        win = tk.Toplevel(self.root)
        if hasattr(ind, 'animated_cxs'):
            cxs = ind.animated_cxs
        else:
            cxs = list(np.random.choice(range(len(ind.connection_genome)), size=min(len(ind.connection_genome), 10), replace=False))
        
        cxs_input = tk.Text(win, height=4, width=30)
        cxs_input.delete(1.0, tk.END)
        cxs_input.insert("end-1c", f"{cxs}")
        
        iterations_input = tk.Text(win, height=2, width= 10)
        iterations_input.delete(1.0, tk.END)
        iterations_input.insert("end-1c", f"30")
        
        res_input = tk.Text(win, height=2, width= 10)
        res_input.delete(1.0, tk.END)
        res_input.insert("end-1c", f"1024")
        
        preview_res_input = tk.Text(win, height=2, width= 10)
        preview_res_input.delete(1.0, tk.END)
        preview_res_input.insert("end-1c", f"300")
        
        n_c_input = tk.Text(win, height=3, width=6)
        n_c_input.delete(1.0, tk.END)
        n_c_input.insert("end-1c", f"10")
        
        mutate_input = tk.Text(win, height=2, width= 5)
        mutate_input.delete(1.0, tk.END)
        mutate_input.insert("end-1c", f".1")
        
        rand_weight  = tk.BooleanVar()
        random_mutate = tk.Checkbutton(win, height=2, width= 2, text="Random",variable = rand_weight  )

        
        gif_win = [None]
        def new_cxs():
            C = int(n_c_input.get("1.0", "end-1c"))
            val = f"{ list(np.random.choice(range(len(ind.connection_genome)), size=min(len(ind.connection_genome), C), replace=False))}"
            cxs_input.delete(1.0, tk.END)
            cxs_input.insert("end-1c", val)

        wait = tk.Label(win, text="Generating, please wait...")
        
        def save_gif(ind):
            cxs = cxs_input.get("1.0", "end-1c").replace("[","").replace("]","").split(",")
            cxs = [int(x) for x in cxs]
            mutate = float(mutate_input.get("1.0", "end-1c"))
            iterations = int(iterations_input.get("1.0", "end-1c") )
            use_random_weights = rand_weight.get()
            res = int(res_input.get("1.0", "end-1c"))
            wait.grid(row=9, column=0, columnspan=10)
            wait.update()
            save.config(state=tk.DISABLED)
            save.update()
            make_weight_video(ind, cxs=cxs, res=res,iterations=iterations, mutate=mutate, random_weights=use_random_weights)
            wait.grid_forget()
            save.config(state=tk.NORMAL)
            save.update()
            
            
        def preview(ind):
            name = f"tmp/{time.time()}.mp4"
            mutate = float(mutate_input.get("1.0", "end-1c") )
            res = int(preview_res_input.get("1.0", "end-1c") )
            iterations = int(iterations_input.get("1.0", "end-1c") )
            cxs = cxs_input.get("1.0", "end-1c").replace("[","").replace("]","").split(",")
            cxs = [int(x) for x in cxs]
            use_random_weights = rand_weight.get()
            make_weight_video(ind, name=name, res=res, cxs=cxs, iterations=iterations, mutate=mutate, random_weights=use_random_weights)
            if gif_win[0] is not None: gif_win[0].destroy()
            gif_win[0] = tk.Toplevel(win)
            gif_label = tk.Label(gif_win[0])
            gif_label.pack()
            
            player = tkvideo.tkvideo(None, loop = 1, label=gif_label, size = (res,res), hz=10)
            player.path = name
            player.play()
        
        # weight = tk.Button(win, text="Weight GIF", command=lambda ind=ind: weight_gif(ind),height=3,font=('Arial', 15, 'bold'))
        preview_btn = tk.Button(win, text="Preview", command=lambda ind=ind: preview(ind),height=3,font=('Arial', 15, 'bold'))
        new_cxs_btn = tk.Button(win, text="New CXS", command=new_cxs,height=3,font=('Arial', 15, 'bold'))
        save = tk.Button(win, text="Save", command=lambda ind=ind: save_gif(ind),height=3,font=('Arial', 15, 'bold'))
        
        cxs_label = tk.Label(win, text="Weights:")
        iterations_label = tk.Label(win, text="Iterations:")
        preview_res_label = tk.Label(win, text="Preview Resolution:")
        res_label = tk.Label(win, text="Resolution:")
        mutate_label = tk.Label(win, text="Mutation Rate:")
        
        save.grid(row=0, column=0,sticky='NEWS', columnspan=3, padx=1)
        preview_btn.grid(row=1, column=0, sticky='NEWS', padx=1)
        new_cxs_btn.grid(row=1, column=1, sticky='NEWS', padx=1)
        n_c_input.grid(row=1, column=2, sticky='NEWS', padx=1)
        cxs_input.grid(row = 2, column = 1, sticky='NEWS', padx=1, columnspan=2)
        mutate_input.grid(row = 5, column = 1, sticky='NEWS', padx=1, columnspan=1)
        mutate_label.grid(row = 5, column = 0, sticky='NEWS', padx=1)
        random_mutate.grid(row = 5, column = 2, sticky='NEWS', padx=1)
        iterations_input.grid(row = 3, column = 1, sticky='NEWS', padx=1, columnspan=2)
        res_input.grid(row = 4, column = 1, sticky='NEWS', padx=1, columnspan=2)
        res_label.grid(row = 4, column = 0, sticky='NEWS', padx=1)
        preview_res_input.grid(row = 6, column = 1, sticky='NEWS', padx=1, columnspan=2)
        preview_res_label.grid(row = 6, column = 0, sticky='NEWS', padx=1)
        cxs_label.grid(row = 2, column = 0, sticky='NEWS', padx=1)
        iterations_label.grid(row = 3, column = 0, sticky='NEWS', padx=1)
   
    def main_window(self):
        self.command_panel = tk.Frame(self.root)
        self.command_panel.grid(row = 0, column = 0, sticky = "nsew", columnspan=100)
        self.images_panel = tk.Frame(self.root)
        self.img_panels = None
        self.save_w_input = tk.Entry(self.command_panel, width=5, font=('Arial', 15, 'bold'))
        self.save_h_input = tk.Entry(self.command_panel, width=5, font=('Arial', 15, 'bold'))
        self.save_w_input.insert(0, 1024)
        self.save_h_input.insert(0, 1024)
        
        def show_genome():
            tmp_ind = copy.deepcopy(self.individual)
            for n in tmp_ind.node_genome:
                n.outputs = None
                n.sum_inputs = None
            visualize_network(tmp_ind)

        def save_genome():
            save_image(self.individual, int(self.save_w_input.get()), int(self.save_h_input.get()))
                 
        def load_genome():
            file = filedialog.askopenfile(mode='r', filetypes=[('JSON', '*.json')], initialdir="./saved_genomes")
            self.individual = load_from_file(file.name)
            image = self.individual.get_image(800, 800, force_recalculate=True)
            img = Image.fromarray((image*255.0).astype(np.uint8))
            img_tk = ImageTk.PhotoImage(img)
            image_preview.image = img_tk
            image_preview.update()
            image_preview.config(image=img_tk)
            self.update_title()
        load_genome_btn = tk.Button(self.command_panel, text='Load genome', command=load_genome, height=3,font=('Arial', 15, 'bold'))
        save_genome_btn = tk.Button(self.command_panel, text='Save image', command=save_genome, height=3,font=('Arial', 15, 'bold'))
        weight_btn = tk.Button(self.command_panel, text='GIF', command=self.weight_gif, height=3,font=('Arial', 15, 'bold'))
       
        image = np.zeros((1024, 1024, 3))
        img = Image.fromarray((image*255.0).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(img)
        image_preview = tk.Label(self.root, image=img_tk)
        # image_preview.image = img_tk
        image_preview.grid(row=1, column=0, sticky="nsew")
        load_genome_btn.grid(row=0, column=0, sticky="nsew", columnspan=1)
        
        show_genome_btn = tk.Button(self.command_panel, text='Show genome', command=show_genome, height=3,font=('Arial', 15, 'bold'))
        save_genome_btn.grid(row=0, column=1, sticky="nsew", columnspan=1)
        show_genome_btn.grid(row=0, column=2, sticky="nsew", columnspan=1)
        weight_btn.grid(row=0, column=3, sticky="nsew", columnspan=1)
        
        save_label = tk.Label(self.command_panel, text='Save resolution:', font=('Arial', 15, 'bold'))
        
        save_label.grid(row=0, column=4, sticky="nsew", columnspan=1)
        self.save_w_input.grid(row=0, column=5, sticky="nsew", columnspan=1)
        self.save_h_input.grid(row=0, column=6, sticky="nsew", columnspan=1)
       
        load_genome()
    def update_title(self, percent=None):
        self.root.title(f"Nodes: {np.mean([len(self.individual.node_genome) for self.individual in [self.individual]]):.2f} | Connections: {np.mean([len(list(self.individual.enabled_connections())) for self.individual in [self.individual]]):.2f}{f' | {percent:.2f}%' if percent is not None else ''}")

if __name__ == "__main__":
    gui = GUI(None)
    gui.run()
    