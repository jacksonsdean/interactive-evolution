from cProfile import label
import copy
import json
from tkinter import Tk, filedialog

from matplotlib import pyplot as plt
import numpy as np
from config import Config
from cppn import CPPN
from PIL import Image
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
   
    def main_window(self):
        self.command_panel = tk.Frame(self.root)
        self.command_panel.grid(row = 0, column = 0, sticky = "nsew", columnspan=100)
        self.images_panel = tk.Frame(self.root)
        self.img_panels = None
        self.save_w_input = tk.Entry(self.command_panel, width=5, font=('Arial', 15, 'bold'))
        self.save_h_input = tk.Entry(self.command_panel, width=5, font=('Arial', 15, 'bold'))
        self.save_w_input.insert(0, self.individual.config.res_w)
        self.save_h_input.insert(0, self.individual.config.res_h)
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
            
        def show_genome():
            tmp_ind = copy.deepcopy(self.individual)
            for n in tmp_ind.node_genome:
                n.outputs = None
                n.sum_inputs = None
            visualize_network(tmp_ind)

        def save_genome():
            w = int(self.save_w_input.get())
            h = int(self.save_h_input.get())
            img = self.individual.get_image(h, w, force_recalculate=True)
            img = Image.fromarray((img*255.0).astype(np.uint8))
            file_name = filedialog.asksaveasfilename(filetypes=[('PNG', '*.png')], defaultextension=".png", initialdir="./saved_imgs")
            print(file_name)
            if file_name != '':
                img.save(file_name)
        load_genome_btn = tk.Button(self.command_panel, text='Load genome', command=load_genome, height=3,font=('Arial', 15, 'bold'))
        save_genome_btn = tk.Button(self.command_panel, text='Save image', command=save_genome, height=3,font=('Arial', 15, 'bold'))
       
        image = np.zeros((1024, 1024, 3))
        img = Image.fromarray((image*255.0).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(img)
        image_preview = tk.Label(self.root, image=img_tk)
        image_preview.image = img_tk
        image_preview.grid(row=1, column=0, sticky="nsew")
        load_genome_btn.grid(row=0, column=0, sticky="nsew", columnspan=1)
        
        show_genome_btn = tk.Button(self.command_panel, text='Show genome', command=show_genome, height=3,font=('Arial', 15, 'bold'))
        save_genome_btn.grid(row=0, column=1, sticky="nsew", columnspan=1)
        show_genome_btn.grid(row=0, column=2, sticky="nsew", columnspan=1)
        
        save_label = tk.Label(self.command_panel, text='Save resolution:', font=('Arial', 15, 'bold'))
        
        save_label.grid(row=0, column=3, sticky="nsew", columnspan=1)
        self.save_w_input.grid(row=0, column=4, sticky="nsew", columnspan=1)
        self.save_h_input.grid(row=0, column=5, sticky="nsew", columnspan=1)
        

    def update_title(self, percent=None):
        self.root.title(f"Nodes: {np.mean([len(self.individual.node_genome) for self.individual in [self.individual]]):.2f} | Connections: {np.mean([len(list(self.individual.enabled_connections())) for self.individual in [self.individual]]):.2f}{f' | {percent:.2f}%' if percent is not None else ''}")

if __name__ == "__main__":
    gui = GUI(None)
    gui.run()
    