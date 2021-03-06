
from logging import warning
from math import e
import platform
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

from visualize import visualize_network
plt.style.use('seaborn')

from PIL import Image
import copy
import time
from network_util import *
from cppn import *
from image_utils import *
# from species import *
# from evolution import *
from activation_functions import *
from cppn import *
# from lib.classification import classification_fitness
# from lib.autoencoder import initialize_encoders
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
from tqdm.notebook import tqdm, trange
from multiprocessing.pool import ThreadPool as Pool
import os

import tkinter as tk
from PIL import ImageTk
from functools import partial

from tkvideo import tkvideo
from tkinter.ttk import *

CONFIG_SAVE_FILE = "iec_config.json"


def mutate(child, config):
    child.image = None # image will be different after mutation
    child.fitness, child.adjusted_fitness = -math.inf, -math.inf # new fitnesses after mutation
    if hasattr(child,"animation_name"): delattr(child,"animation_name")
    
    if(np.random.uniform(0,1) < config.prob_random_restart):
        child = CPPN(config)
    if(np.random.uniform(0,1) < config.prob_add_node):
        child.add_node()
    if(np.random.uniform(0,1) < config.prob_remove_node):
        child.remove_node()
    if(np.random.uniform(0,1) < config.prob_add_connection):
        child.add_connection()
    if(np.random.uniform(0,1) < config.prob_disable_connection):
        child.disable_connection()
    
    child.mutate_activations()
    child.mutate_weights()
    child.update_node_layers()
    child.disable_invalid_connections()
    return child



class GUI:
    def __init__(self, root=None):
        
        config = Config()
        config.load_saved(CONFIG_SAVE_FILE)
        self.config = config
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.selected = []
        self.history = []
        self.gen = 0
        self.pop = None
        self.last_selected = None
        self.seed_index = 0
        config.num_inputs = 2
        if config.use_radial_distance:
            config.num_inputs += 1
        if config.use_input_bias:
            config.num_inputs += 1
        
    def run(self):
        dir = 'tmp/'
        for f in os.listdir(dir):
            try:
                os.remove(os.path.join(dir, f))
            except OSError:
                continue
        self.pop = self.initial_population()
        self.main_window()
        self.next_gen(False)
        self.update_title()
        try:
            self.root.mainloop()
        except MemoryError:
            print("Memory Error")
            self.root.mainloop()
        except np.core._exceptions._ArrayMemoryError:
            print("Memory Error")
            self.root.mainloop()
    
    def settings_window(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        
        def save_settings():
            needs_reset = False
            self.config.population_image_size = [int(res_input.get("1.0", "end-1c"))]*2
            self.config.detail_image_size = [int(detail_res_input_h.get("1.0", "end-1c")), int(detail_res_input_w.get("1.0", "end-1c"))]
            self.config.save_size = [int(save_res_input_h.get("1.0", "end-1c")), int(save_res_input_w.get("1.0", "end-1c"))]
            self.config.prob_mutate_weight = float(prob_weight_input.get("1.0", "end-1c"))
            self.config.prob_add_node = float(prob_add_node_input.get("1.0", "end-1c"))
            self.config.prob_remove_node = float(prob_remove_node_input.get("1.0", "end-1c"))
            self.config.prob_add_connection = float(prob_add_connection_input.get("1.0", "end-1c"))
            self.config.prob_disable_connection = float(prob_disable_connection_input.get("1.0", "end-1c"))
            self.config.hidden_nodes_at_start = int(hidden_at_start_input.get("1.0", "end-1c"))
            self.config.max_weight = float(max_weight_input.get("1.0", "end-1c"))
            self.config.weight_mutation_max = float(mutate_weight_input.get("1.0", "end-1c"))
            self.config.wildcard = wildcard_var.get()
            self.config.allow_recurrent = recurrent_var.get()
            self.config.allow_input_activation_mutation = input_mutation_var.get()
            self.config.population_size = int(pop_input.get("1.0", "end-1c"))
            new_animate = animate_var.get()
            if new_animate != self.config.animate:
                self.config.animate = new_animate
                # needs_reset = True
            
            new_radial_dist = radial_dist_var.get()
            if new_radial_dist != self.config.use_radial_distance:
                needs_reset = True
                self.config.use_radial_distance = new_radial_dist
            
            
            new_color_mode = color_mode_var.get()
            if new_color_mode != self.config.color_mode:
                if new_color_mode in ["RGB", "HSL", "L"]:
                    self.config.color_mode = new_color_mode
                    self.config.num_outputs = len(self.config.color_mode)
                    needs_reset = True
            self.config.save_json(CONFIG_SAVE_FILE)
            for individual in self.pop:
                individual.config = self.config
                
            if needs_reset:
                self.reset_button_clicked()
            else:
                self.next_gen(False)

        save_btn = tk.Button(win, text="Save", command=save_settings,height=3,font=('Arial', 15, 'bold'))
        
        res_input = tk.Text(win, height=2, width= 10)
        res_input.delete(1.0, tk.END)
        res_input.insert("end-1c", f"{self.config.population_image_size[0]}")
        res_label = tk.Label(win, text="Preview size:", font=('Arial', 10))
        
        detail_res_input_h = tk.Text(win, height=2, width= 10)
        detail_res_input_h.delete(1.0, tk.END)
        detail_res_input_h.insert("end-1c", f"{self.config.detail_image_size[0]}")
        detail_res_input_w = tk.Text(win, height=2, width= 10)
        detail_res_input_w.delete(1.0, tk.END)
        detail_res_input_w.insert("end-1c", f"{self.config.detail_image_size[1]}")
        detail_res_label = tk.Label(win, text="Detail size:", font=('Arial', 10))
       
        save_res_input_h = tk.Text(win, height=2, width= 10)
        save_res_input_h.delete(1.0, tk.END)
        save_res_input_h.insert("end-1c", f"{self.config.save_size[0]}")
        save_res_input_w = tk.Text(win, height=2, width= 10)
        save_res_input_w.delete(1.0, tk.END)
        save_res_input_w.insert("end-1c", f"{self.config.save_size[1]}")
        save_res_label = tk.Label(win, text="Save size:", font=('Arial', 10))
        
        prob_weight_input = tk.Text(win, height=2, width= 10)
        prob_weight_input.delete(1.0, tk.END)
        prob_weight_input.insert("end-1c", f"{self.config.prob_mutate_weight}")
        prob_weight_label = tk.Label(win, text="Prob. mutate weight:", font=('Arial', 10))
        
        max_weight_input = tk.Text(win, height=2, width= 10)
        max_weight_input.delete(1.0, tk.END)
        max_weight_input.insert("end-1c", f"{self.config.max_weight}")
        max_weight_label = tk.Label(win, text="Max weight:", font=('Arial', 10))
        
        mutate_weight_input = tk.Text(win, height=2, width= 10)
        mutate_weight_input.delete(1.0, tk.END)
        mutate_weight_input.insert("end-1c", f"{self.config.max_weight}")
        mutate_weight_label = tk.Label(win, text="Weight mutation:", font=('Arial', 10))
        
        pop_input = tk.Text(win, height=2, width= 10)
        pop_input.delete(1.0, tk.END)
        pop_input.insert("end-1c", f"{self.config.population_size}")
        pop_label = tk.Label(win, text="Population size:", font=('Arial', 10))
    
         
        prob_add_node_input = tk.Text(win, height=2, width= 10)
        prob_add_node_input.delete(1.0, tk.END)
        prob_add_node_input.insert("end-1c", f"{self.config.prob_add_node}")
        prob_add_node_label = tk.Label(win, text="Prob. add node:", font=('Arial', 10))
         
        prob_add_connection_input = tk.Text(win, height=2, width= 10)
        prob_add_connection_input.delete(1.0, tk.END)
        prob_add_connection_input.insert("end-1c", f"{self.config.prob_add_connection}")
        prob_add_connection_label = tk.Label(win, text="Prob. add cx:", font=('Arial', 10))
    
         
        prob_remove_node_input = tk.Text(win, height=2, width= 10)
        prob_remove_node_input.delete(1.0, tk.END)
        prob_remove_node_input.insert("end-1c", f"{self.config.prob_remove_node}")
        prob_remove_node_label = tk.Label(win, text="Prob. remove node:", font=('Arial', 10))
         
        prob_disable_connection_input = tk.Text(win, height=2, width= 10)
        prob_disable_connection_input.delete(1.0, tk.END)
        prob_disable_connection_input.insert("end-1c", f"{self.config.prob_disable_connection}")
        prob_disable_connection_label = tk.Label(win, text="Prob. disable cx:", font=('Arial', 10))
        
        hidden_at_start_input = tk.Text(win, height=2, width= 10)
        hidden_at_start_input.delete(1.0, tk.END)
        hidden_at_start_input.insert("end-1c", f"{self.config.hidden_nodes_at_start}")
        hidden_at_start_label = tk.Label(win, text="Nodes at start:", font=('Arial', 10))
    
        color_mode_var = tk.StringVar(win)
        color_mode_var.set(self.config.color_mode)
        color_label = tk.Label(win, text="Color mode:", font=('Arial', 10))
        color_rgb = tk.Radiobutton(win, text="RGB", variable=color_mode_var,  value="RGB")
        color_hsl = tk.Radiobutton(win, text="HSL", variable=color_mode_var,  value="HSL")
        color_l = tk.Radiobutton(win, text="L", variable=color_mode_var,  value="L")
        

        n_c_input = tk.Text(win, height=3, width=6)
        n_c_input.delete(1.0, tk.END)
        n_c_input.insert("end-1c", f"10")
        
        mutate_input_add_connection = tk.Text(win, height=2, width= 5)
        mutate_input_add_connection.delete(1.0, tk.END)
        mutate_input_add_connection.insert("end-1c", f".1")
        
        animate_var  = tk.BooleanVar(value=self.config.animate)
        animate_var.set(self.config.animate)
        animate_check = tk.Checkbutton(win, variable=animate_var, text="Animate", font=('Arial', 10))
        
        wildcard_var  = tk.BooleanVar(value=self.config.wildcard)
        wildcard_var.set(self.config.wildcard)
        wildcard_check = tk.Checkbutton(win, variable=wildcard_var, text="Wildcard", font=('Arial', 10))
        
        recurrent_var  = tk.BooleanVar(value=self.config.allow_recurrent)
        recurrent_var.set(self.config.allow_recurrent)
        recurrent_check = tk.Checkbutton(win, variable=recurrent_var, text="Recurrent cxs", font=('Arial', 10))
        
        input_mutation_var  = tk.BooleanVar(value=self.config.allow_input_activation_mutation)
        input_mutation_var.set(self.config.allow_input_activation_mutation)
        input_mutation_check = tk.Checkbutton(win, variable=input_mutation_var, text="Mutate input activation", font=('Arial', 10))
        
        radial_dist_var  = tk.BooleanVar(value=self.config.use_radial_distance)
        radial_dist_var.set(self.config.use_radial_distance)
        radial_dist_check = tk.Checkbutton(win, variable=radial_dist_var, text="Radial distance (bias for circles)", font=('Arial', 10))
        
        
        row, col = 0, 0
        
        res_label.grid(row=row, column=col, sticky="w")
        col+=1
        res_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        pop_label.grid(row=row, column=col, sticky="w")
        col+=1
        pop_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        prob_weight_label.grid(row=row, column=col, sticky="w")
        col+=1
        prob_weight_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        max_weight_label.grid(row=row, column=col, sticky="w")
        col+=1
        max_weight_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        mutate_weight_label.grid(row=row, column=col, sticky="w")
        col+=1
        mutate_weight_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        prob_add_node_label.grid(row=row, column=col, sticky="w")
        col+=1
        prob_add_node_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        prob_remove_node_label.grid(row=row, column=col, sticky="w")
        col+=1
        prob_remove_node_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        prob_add_connection_label.grid(row=row, column=col, sticky="w")
        col+=1
        prob_add_connection_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        prob_disable_connection_label.grid(row=row, column=col, sticky="w")
        col+=1
        prob_disable_connection_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        hidden_at_start_label.grid(row=row, column=col, sticky="w")
        col+=1
        hidden_at_start_input.grid(row=row, column=col, columnspan=8)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        
        
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        
        
        animate_check.grid(row=row, column=col, sticky="w", columnspan=8, rowspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        wildcard_check.grid(row=row, column=col, sticky="w", columnspan=8, rowspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        recurrent_check.grid(row=row, column=col, sticky="w", columnspan=8, rowspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        save_btn.grid(row=row, column=col, columnspan=8, sticky="ew")
        
        row+=1; col = 0
        input_mutation_check.grid(row=row, column=col, sticky="w", columnspan=8, rowspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
       
        row+=1; col = 0
        detail_res_label.grid(row=row, column=col, sticky="w")
        col+=1
        detail_res_input_w.grid(row=row, column=col, columnspan=1)
        col+=1
        detail_res_input_h.grid(row=row, column=col, columnspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        
        row+=1; col = 0
        save_res_label.grid(row=row, column=col, sticky="w")
        col+=1
        save_res_input_w.grid(row=row, column=col, columnspan=1)
        col+=1
        save_res_input_h.grid(row=row, column=col, columnspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        
       
        row+=1; col = 0
        warning_label = Label(win, text="Requires reset:", font=("Helvetica", 12, "bold"))
        warning_label.grid(row=row, column=col, sticky="w"); 
        row+=1; col = 0
        radial_dist_check.grid(row=row, column=col, sticky="w", columnspan=8, rowspan=1)
        Separator(win, orient='horizontal').grid(row=row+1, column=0, sticky="ew", columnspan=8); row+=1
        row+=1; col = 0
        color_label.grid(row=row, column=col, sticky="w")
        row+=1; col = 0
        color_rgb.grid(row=row, column=col)
        col+=1
        color_hsl.grid(row=row, column=col)
        col+=1
        color_l.grid(row=row, column=col)
        row+=1; col = 0
        save_btn.grid(row=row, column=col, columnspan=8, sticky="ew")
        
        
    def do_popup(self, event):
        try:
            if not event.widget in self.img_panels:
                return
            x = event.x_root 
            y = event.y_root
            clicked_index = self.img_panels.index(event.widget)
            self.popup_menu.entryconfig("Save image", command=lambda:self.save_image_popup_clicked(clicked_index))
            self.popup_menu.entryconfig("Save genome", command=lambda:self.save_genome_popup_clicked(clicked_index))
            self.popup_menu.entryconfig("Show genome", command=lambda:self.show_genome_popup_clicked(clicked_index))
            self.popup_menu.entryconfig("Details", command=lambda:self.details_popup_clicked(clicked_index))
            self.popup_menu.tk_popup(x, y)
        finally:
            self.popup_menu.grab_release()
  
    def save_image_popup_clicked(self, index):
        ind = self.pop[index]
        save_image(ind, self.config.save_size[1], self.config.save_size[0])
        
    def show_genome_popup_clicked(self, index):
        ind = self.pop[index]
        self.show_genome(ind)
    def save_genome_popup_clicked(self, index):
        ind = self.pop[index]
        self.save_genome_window(ind)
    def details_popup_clicked(self, index):
        ind = self.pop[index]
        self.details([ind])

    def show_genome(self, ind):
        ind.update_node_layers()
        visualize_network(ind)

    def seed(self):
        loaded_genomes = filedialog.askopenfilenames(initialdir="./saved_genomes", title="Select genomes to seed", filetypes=[("Genome files", "*.json")])    
        if loaded_genomes == "" or loaded_genomes == () or loaded_genomes is None:
            return
        self.selected = []
        self.root.config(cursor="wait")
        self.root.update()
        for g in loaded_genomes:
            ind = CPPN.load_static(g)
            ind.selected = False
            if ind.n_outputs!= len(self.config.color_mode):
                tk.messagebox.showerror("Error", "Genome {} has {} outputs, but the color mode is {}".format(g, ind.n_inputs, self.config.color_mode))
                continue
            if ind.n_inputs!= self.pop[0].n_inputs:
                tk.messagebox.showerror("Error", "Genome {} has {} inputs, but the population has {} inputs".format(g, ind.n_inputs, self.pop[0].n_inputs))
                continue
            self.pop[-(self.seed_index+1)] = ind
            self.seed_index+=1
        self.root.config(cursor="")
        self.render_pop()
    
    
    def main_window(self):
        self.command_panel = tk.Frame(self.root)
        self.command_panel.pack(side=tk.TOP, fill=tk.X)
        self.images_panel = tk.Frame(self.root)
        self.img_panels = None
                 
        self.popup_menu = tk.Menu(self.root, tearoff = 0)
        self.popup_menu.add_command(label ="Save image")
        self.popup_menu.add_command(label ="Save genome")
        self.popup_menu.add_command(label ="Show genome")
        self.popup_menu.add_command(label ="Details")
        # if windows
        if platform.system() == "Windows":
            self.root.bind("<Button-3>", self.do_popup)
        # if mac
        elif platform.system() == "Darwin":
            self.root.bind("<Button-2>", self.do_popup)
        
        next_gen = tk.Button(self.command_panel, text='Next Generation', command=lambda: self.next_gen_button_clicked(), height=3,font=('Arial', 15, 'bold'))
        prev_btn = tk.Button(self.command_panel, text='Previous Generation', command=lambda: self.previous_gen_clicked(), height=3,font=('Arial', 15, 'bold'))
        save_btn = tk.Button(self.command_panel, text='Save Selected', command=lambda: self.save_button_clicked(), height=3,font=('Arial', 15, 'bold'))
        retry_btn = tk.Button(self.command_panel, text='Retry', command=self.retry_clicked, height=3,font=('Arial', 15, 'bold'))
        reset_btn = tk.Button(self.command_panel, text='Reset', command=lambda: self.reset_button_clicked(), height=3,font=('Arial', 15, 'bold'))
        # details_btn = tk.Button(self.command_panel, text='Details', command=lambda: self.details(self.selected), height=3,font=('Arial', 15, 'bold'))
        seed_btn = tk.Button(self.command_panel, text='Seed', command=self.seed, height=3,font=('Arial', 15, 'bold'))
        settings_btn = tk.Button(self.command_panel, text='Settings', command=self.settings_window, height=3,font=('Arial', 15, 'bold'))

        next_gen.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        retry_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        # save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        # details_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        seed_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)
        settings_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=4)

    def previous_gen_clicked(self):
        self.prev_gen()
        if self.gen>1:self.gen-=1
        self.update_title()
        
    def next_gen_button_clicked(self):
        if len(self.selected) == 0: return
        self.next_gen()
        self.gen+=1
        self.update_title()
        
    def retry_clicked(self):
        self.history.append(self.pop)
        self.previous_gen_clicked()
        if self.last_selected is not None:
            self.selected = self.last_selected
        self.next_gen_button_clicked()
        self.update_title()
        
    def save_genome_window(self, ind):
        file_name = filedialog.asksaveasfilename(filetypes=[('Genome', '*.json')], defaultextension=".json", initialdir="./save_genomes")
        if file_name is not None and file_name != "":
            ind.save(file_name)
            
    def weight_gif (self,ind):
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
        
    def details(self, selected):
        wins = [None]*len(selected)
        for i, ind in enumerate(selected):
            image = ind.get_image(self.config.detail_image_size[0], self.config.detail_image_size[1])
            win = tk.Toplevel(self.root)
            wins[i] = win
            weight = tk.Button(wins[i], text="Weight GIF", command=lambda ind=ind: self.weight_gif(ind),height=3,font=('Arial', 15, 'bold'))
            ok = tk.Button(wins[i], text="Close", command=wins[i].destroy,height=3,font=('Arial', 15, 'bold'))
            save = tk.Button(wins[i], text="Save", command=lambda ind=ind: save_image(ind, self.config.save_size[1], self.config.save_size[0]),height=3,font=('Arial', 15, 'bold'))
            show_genome = tk.Button(wins[i], text="Show Genome", command=lambda ind=ind: self.show_genome(ind),height=3,font=('Arial', 15, 'bold'))
            save_genome = tk.Button(wins[i], text="Save genome", command=lambda ind=ind:self.save_genome_window(ind), height=3,font=('Arial', 15, 'bold'))
            wins[i].title(f"Nodes:{len(ind.node_genome)} | Connections:{len(list(ind.enabled_connections()))}")
            img = Image.fromarray((image*255.0).astype(np.uint8))
            img_tk = ImageTk.PhotoImage(img)
            p = tk.Label(wins[i], image=img_tk)
            p.image = img_tk
            
            
            p.grid(row=1, column=0, columnspan=6)
            save.grid(row=0, column=0,sticky='ew', columnspan=2, padx=1)
            save_genome.grid(row=0, column=2,sticky='ew', columnspan=1, padx=1)
            show_genome.grid(row=0, column=3,sticky='ew', columnspan=1, padx=1)
            weight.grid(row=0, column=4, sticky='ew', padx=1)
            ok.grid(row=0, column=5, sticky='ew', padx=1)

    def save_button_clicked(self):
        for ind in self.selected:
            save_image
            (ind, 4096, 4096) 
    
    def update_title(self):
        self.root.title(f"Generation: {self.gen} | Nodes: {np.mean([len(ind.node_genome) for ind in self.pop]):.2f} | Connections: {np.mean([len(list(ind.enabled_connections())) for ind in self.pop]):.2f}")
     
    def reset_button_clicked(self):
        self.history.append(self.pop)
        self.gen_before_reset = self.gen
        self.gen = 0
        self.pop = self.initial_population()
        self.next_gen(False)
        
    def prev_gen(self):
        if len(self.history) > 0:
            self.pop = self.history.pop()
            self.next_gen(False)
            if self.gen == 0 and hasattr(self, "gen_before_reset"):
                self.gen = self.gen_before_reset
                delattr(self, "gen_before_reset")
    
            
    def next_gen(self, new_pop=True):
        if new_pop:
            self.history.append(copy.deepcopy(self.pop))
            self.pop = self.next_population()
        if len(self.selected)>0:
            self.last_selected = self.selected
        self.seed_index = 0
        self.selected = []
        self.render_pop()
        
    def render_pop(self):
        self.root.config(cursor="wait")
        self.root.update()
         # delete all from images_panel
        for widget in self.images_panel.winfo_children():
            if widget!=self.images_panel:
                widget.destroy()
        self.img_panels = [None] * len(self.pop)

        def select(index):
            if self.pop[index] in self.selected:
                self.selected.remove(self.pop[index])
                self.img_panels[index].configure(borderwidth=6, relief="sunken", highlightbackground="white", highlightcolor="white", bg="white", highlightthickness=2)
            else:
                self.selected.append(self.pop[index])
                self.img_panels[index].configure(borderwidth=6, relief="raised", highlightbackground="red", highlightcolor="red", bg="red", highlightthickness=2)
                
        for i, individual in enumerate(self.pop):
            row = i // 7
            column = i % 7
            if self.config.animate:
                cxs = list(np.random.choice(range(len(individual.connection_genome)), size=min(len(individual.connection_genome), 10), replace=False))
                individual.animated_cxs = cxs
                self.img_panels[i] = tk.Label(self.images_panel)
                
                if not hasattr(individual, "animation_name"):
                    name = f"tmp/{time.time()}.mp4"
                    individual.animation_name = name
                    make_weight_video(individual, name=individual.animation_name, cxs=cxs, mutate=0.1, iterations=30)
                    
                player = tkvideo.tkvideo(individual.animation_name, loop = 1, label=self.img_panels[i], size = self.config.population_image_size, hz=10)
                player.play()
                    
                self.img_panels[i].bind("<Button-1>", lambda event, index=i: select(index))
                self.img_panels[i].update()
            else:
                img = individual.get_image(self.config.population_image_size[0], self.config.population_image_size[1])
                img = Image.fromarray((img*255.0).astype(np.uint8))
                img_tk = ImageTk.PhotoImage(img)
                self.img_panels[i] = tk.Button(self.images_panel, image=img_tk, command=partial(select, i))
                self.img_panels[i].image = img_tk
                
            self.img_panels[i].grid(row=row, column=column, padx= 2, pady= 2)
            self.img_panels[i].configure(borderwidth=6, relief="sunken", highlightbackground="white", highlightcolor="white", bg="white", highlightthickness=2)
            
            
        self.images_panel.pack()
        self.root.config(cursor="")
        
        
    def initial_population(self):
        pop = []
        for _ in range(self.config.population_size):
            ind = CPPN(self.config)
            pop.append(ind)
        return pop

    def next_population(self):
        last_gen = self.pop
        next_gen = copy.copy(self.selected)
        num_crossover = 0  if len(self.selected) < 1 else self.config.population_size//2
        if num_crossover > 0:
            for _ in range(num_crossover):
                choice1 = np.random.choice(self.selected)
                choice2 = np.random.choice(self.selected)
                parent1 = copy.deepcopy(choice1)
                parent2 = copy.deepcopy(choice2)
                child = parent1.crossover(parent2)
                child.get_image(self.config.population_image_size[0], self.config.population_image_size[0])
                next_gen.append(child)
        
        num = self.config.population_size
        if self.config.wildcard:
            num-=1
        while len(next_gen) < num:
            choice = np.random.choice(self.selected)
            if hasattr(choice, "animated_player"): delattr(choice, "animated_player")
            next_gen.append(mutate(copy.deepcopy(choice), self.config))
        
        if self.config.wildcard:
            next_gen.append(CPPN(self.config)) # for spicy
        
        for i, ind in enumerate(next_gen):
            if ind not in self.selected:
                next_gen[i] = mutate(ind, self.config)
        
        return next_gen
    