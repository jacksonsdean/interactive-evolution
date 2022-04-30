
from math import e
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
plt.style.use('seaborn')

from PIL import Image
import copy
from IPython.display import display # to display images
import time
from network_util import *
from cppn import *
from image_utils import *
from activation_functions import *
from config import *
from cppn import *
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
from multiprocessing.pool import ThreadPool as Pool
import asyncio
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt



import tkinter as tk
from PIL import ImageTk
from functools import partial
from tkinter import filedialog

from tkvideo import tkvideo
from tkinter.ttk import *


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
        child.add_connection(config.prob_reenable_connection, config.allow_recurrent)
    if(np.random.uniform(0,1) < config.prob_disable_connection):
        child.disable_connection()
    # if(np.random.uniform(0,1)< prob_mutate_activation):
    
    child.mutate_activations(config.prob_mutate_activation)
    child.mutate_weights(config.weight_mutation_max, config.prob_mutate_weight)
    return child



class GUI:
    def __init__(self, config, root):
        self.config = config
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.history = []
        self.gen = 0
        self.individual = CPPN(config)
        self.new_config()
        self.music_file = ""
        self.sample_rate = 0
        self.spectrogram = None
        self.time_series = None
        self.num_frequencies = 5
        
    
    def new_config(self):
        self.config.num_inputs = 2
        if self.config.use_radial_distance:
            self.config.num_inputs += 1
        if self.config.use_input_bias:
            self.config.num_inputs += 1
        self.config.num_outputs = len(self.config.color_mode)
    
    def run(self):
        dir = 'tmp/'
        for f in os.listdir(dir):
            try:
                os.remove(os.path.join(dir, f))
            except OSError:
                continue
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
    
        win = tk.Toplevel(self.root)
        win.title("Settings")
        
        res_input = tk.Text(win, height=2, width= 10)
        res_input.delete(1.0, tk.END)
        res_input.insert("end-1c", f"{self.config.population_image_size[0]}")
        res_label = tk.Label(win, text="Preview size:", font=('Arial', 10))
        
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
        
        color_label.grid(row=row, column=col, sticky="w")
        row+=1; col = 0
        color_rgb.grid(row=row, column=col)
        col+=1
        color_hsl.grid(row=row, column=col)
        col+=1
        color_l.grid(row=row, column=col)
        
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
        save_btn.grid(row=row, column=col, columnspan=8, sticky="ew")
        
    def reset_button_clicked(self):
        pass
    def main_window(self):
        self.command_panel = tk.Frame(self.root)
        self.command_panel.grid(row = 0, column = 0, sticky = "nsew", columnspan=100)
        self.images_panel = tk.Frame(self.root)
        self.img_panels = None
        
        def load_music():
            file = filedialog.askopenfile(mode='r', filetypes=[('MP3', '*.mp3'), ('WAV', '*.wav'), ('All', "*.*")], initialdir="./music")
            music_label.config(text=f"Music: loading...")
            music_label.update()
            self.music_file = file.name
             # getting information from the file
            try:
                self.time_series, self.sample_rate = librosa.load(self.music_file)# getting a matrix which contains amplitude values according to frequency and time indexes
            except:
                # show tk error and abort 
                tk.messagebox.showerror("Error", "Could not load the file, try again")
                music_label.config(text=f"Please load music")
                self.time_series, self.sample_rate = None, None
                self.music_file = ""
                return 
            stft = np.abs(librosa.stft(self.time_series, hop_length=512, n_fft=2048*4))# converting the matrix to decibel matrix
            self.spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
            self.spectrogram = np.abs(self.spectrogram)
            music_label.config(text=f"Music: {os.path.basename(file.name)}")
        
        def load_genome():
            file = filedialog.askopenfile(mode='r', filetypes=[('JSON', '*.json')], initialdir="./saved_genomes")
            self.individual = CPPN(self.config)
            self.individual.load(file.name)
            self.config = self.individual.config
            self.new_config()
            image = self.individual.get_image(800, 800, force_recalculate=True)
            print(image.shape)
            # img = Image.fromarray((image*255.0).astype(np.uint8))
            img = Image.fromarray((image*255.0).astype(np.uint8))
            img_tk = ImageTk.PhotoImage(img)
            image_preview.image = img_tk
            image_preview.update()
            image_preview.config(image=img_tk)
            if hasattr(self.individual, 'animated_cxs'):
                cxs = self.individual.animated_cxs
            else:
                cxs = list(np.random.choice(range(len(self.individual.connection_genome)), size=num_to_mutate, replace=False))
                self.individual.animated_cxs = cxs
            update_cxs()
            self.update_title()
            
        save_btn = tk.Button(self.command_panel, text='Save', command=lambda: self.save_button_clicked(), height=3,font=('Arial', 15, 'bold'))
        details_btn = tk.Button(self.command_panel, text='Details', command=lambda: self.details(self.selected), height=3,font=('Arial', 15, 'bold'))
        load_genome_btn = tk.Button(self.command_panel, text='Load genome', command=load_genome, height=3,font=('Arial', 15, 'bold'))
        load_music_btn = tk.Button(self.command_panel, text='Load music', command=load_music, height=3,font=('Arial', 15, 'bold'))
        music_label = tk.Label(self.command_panel, text='Please load music', font=('Arial', 12, 'bold'))
        
        image = self.individual.get_image(800, 800, self.config.color_mode)
        img = Image.fromarray((image*255.0).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(img)
        image_preview = tk.Label(self.root, image=img_tk)
        image_preview.image = img_tk
        
        
        num_to_mutate = 1
        
        if hasattr(self.individual, 'animated_cxs'):
            cxs = self.individual.animated_cxs
        else:
            cxs = list(np.random.choice(range(len(self.individual.connection_genome)), size=num_to_mutate, replace=False))
            self.individual.animated_cxs = cxs
            
        self.cxs_input = tk.Text(self.root, height=4, width=30)
        self.cxs_input.delete(1.0, tk.END)
        self.cxs_input.insert("end-1c", f"{cxs}")
        
        iterations_input = tk.Text(self.root, height=2, width= 10)
        iterations_input.delete(1.0, tk.END)
        iterations_input.insert("end-1c", f"30")
        
        self.res_h_input = tk.Text(self.root, height=2, width= 10)
        self.res_h_input.delete(1.0, tk.END)
        self.res_h_input.insert("end-1c", f"64")
        
        self.res_w_input = tk.Text(self.root, height=2, width= 10)
        self.res_w_input.delete(1.0, tk.END)
        self.res_w_input.insert("end-1c", f"64")

        preview_res_input = tk.Text(self.root, height=2, width= 10)
        preview_res_input.delete(1.0, tk.END)
        preview_res_input.insert("end-1c", f"300")
        
        n_c_input = tk.Text(self.root, height=3, width=6)
        n_c_input.delete(1.0, tk.END)
        n_c_input.insert("end-1c", f"10")
        
        mutate_input = tk.Text(self.root, height=2, width= 5)
        mutate_input.delete(1.0, tk.END)
        mutate_input.insert("end-1c", f".1")
        
        self.mutate_var_input = tk.Text(self.root, height=2, width= 5)
        self.mutate_var_input.delete(1.0, tk.END)
        self.mutate_var_input.insert("end-1c", f".8")
        
        self.mutate_exp_input = tk.Text(self.root, height=2, width= 5)
        self.mutate_exp_input.delete(1.0, tk.END)
        self.mutate_exp_input.insert("end-1c", f"2")
        
        self.complexify_threshold_input = tk.Text(self.root, height=2, width= 5)
        self.complexify_threshold_input.delete(1.0, tk.END)
        self.complexify_threshold_input.insert("end-1c", f"999")
        
        rand_weight  = tk.BooleanVar()
        random_mutate = tk.Checkbutton(self.root, height=2, width= 2, text="Random",variable = rand_weight  )
        

        
        def new_cxs():
            C = int(n_c_input.get("1.0", "end-1c"))
            cxs =  list(np.random.choice(range(len(self.individual.connection_genome)), size=min(len(self.individual.connection_genome), C), replace=False))
            self.individual.animated_cxs = cxs
            update_cxs()
        def update_cxs():
            val = f"{self.individual.animated_cxs}"
            self.cxs_input.delete(1.0, tk.END)
            self.cxs_input.insert("end-1c", val)
        
        wait = tk.Label(self.root, text="Generating, please wait...")
        
        def save_gif():
            cxs = self.cxs_input.get("1.0", "end-1c").replace("[","").replace("]","").split(",")
            cxs = [int(x) for x in cxs]
            mutate = float(mutate_input.get("1.0", "end-1c"))
            iterations = int(iterations_input.get("1.0", "end-1c") )
            use_random_weights = rand_weight.get()
            res_h = int(self.res_h_input.get("1.0", "end-1c"))
            wait.grid(row=9, column=0, columnspan=10)
            wait.update()
            save.config(state=tk.DISABLED)
            save.update()
            make_weight_video(self.individual, cxs=cxs, res=res_h,iterations=iterations, mutate=mutate, random_weights=use_random_weights)
            wait.grid_forget()
            save.config(state=tk.NORMAL)
            save.update()
            
        gif_win = [None]
        def preview():
            tmp_exists = os.path.exists(f"tmp/")
            if not tmp_exists:
                os.makedirs("tmp/")
            name = f"tmp/{time.time()}.mp4"
            mutate = float(mutate_input.get("1.0", "end-1c") )
            res = int(preview_res_input.get("1.0", "end-1c") )
            iterations = int(iterations_input.get("1.0", "end-1c") )
            cxs = self.cxs_input.get("1.0", "end-1c").replace("[","").replace("]","").split(",")
            cxs = [int(x) for x in cxs]
            use_random_weights = rand_weight.get()
            make_weight_video(self.individual, name=name, res=res, cxs=cxs, iterations=iterations, mutate=mutate, random_weights=use_random_weights)
            if gif_win[0] is not None: gif_win[0].destroy()
            gif_win[0] = tk.Toplevel(self.root)
            gif_label = tk.Label(gif_win[0])
            gif_label.pack()
            
            player = tkvideo.tkvideo(None, loop = 1, label=gif_label, size = (res,res), hz=10)
            player.path = name
            player.play()
        
        # weight = tk.Button(win, text="Weight GIF", command=lambda self.individual=self.individual: weight_gif(self.individual),height=3,font=('Arial', 15, 'bold'))
        preview_btn = tk.Button(self.root, text="Preview", command=preview,height=3,font=('Arial', 8, 'bold'))
        new_cxs_btn = tk.Button(self.root, text="New CXS", command=new_cxs,height=3,font=('Arial', 8, 'bold'))
        save = tk.Button(self.root, text="Save", command=save_gif,height=3,font=('Arial', 8, 'bold'))
        
        cxs_label = tk.Label(self.root, text="Weights:")
        iterations_label = tk.Label(self.root, text="Iterations:")
        preview_res_label = tk.Label(self.root, text="Preview resolution:")
        res_label = tk.Label(self.root, text="Res (h,w):")
        mutate_label = tk.Label(self.root, text="Preview mutation rate:")
        mutate_var_label = tk.Label(self.root, text="Music weight variation:")
        mutate_exp_label = tk.Label(self.root, text="Exponent:")
        complexify_threshold_label = tk.Label(self.root, text="Complexify threshold:")
        
        
        row = 0
        col = 0
        
        save_btn.grid(row=row, column=col,sticky='NEWS')
        col+=1
        details_btn.grid(row=row, column=col,sticky='NEWS')

        row+=1
        col = 0
        load_genome_btn.grid(row=row, column=col,sticky='NEWS')
        col += 1
        load_music_btn.grid(row=row, column=col,sticky='NEWS')
        row+=1
        col = 0
        music_label.grid(row=row, column=col,sticky='NEWS', columnspan=2)
        col=0
        row+=1
        
        # save.grid(row=row, column=col, sticky='NEWS', columnspan=3, padx=1)
        # col+=1
        preview_btn.grid(row=row, column=col,sticky='NEWS', padx=1)
        col=0
        row +=1
        new_cxs_btn.grid(row=row, column=col,sticky='NEWS', padx=1)
        col+=1
        n_c_input.grid(row=row, column=col,sticky='NEWS', padx=1)
        col=0
        row+=1
        
        cxs_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        self.cxs_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=2)
        col=0
        row+=1
        
        mutate_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        mutate_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        col+=1
        random_mutate.grid(row=row, column=col, sticky='NEWS', padx=1)
        col=0
        row+=1
        
        mutate_var_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        self.mutate_var_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        row+=1
        col = 0
        
        mutate_exp_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        self.mutate_exp_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        row+=1
        col=0
        
        complexify_threshold_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        self.complexify_threshold_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        row+=1
        
        col = 0
        iterations_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        iterations_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=2)
        col = 0
        row+=1
        
        res_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        self.res_h_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        col+=1
        self.res_w_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=1)
        row +=1
        col = 0
        
        preview_res_label.grid(row=row, column=col, sticky='NEWS', padx=1)
        col+=1
        preview_res_input.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=2)
        row +=1
        col = 0

        row = 0
        col = 30
        image_preview.grid(row=row, column=col, sticky='NEWS', padx=1, columnspan=20, rowspan=20)

    def make_vis(self):
        # len_in_seconds = librosa.get_duration(self.time_series, sr=self.sample_rate)
        mutate_exponent = float(self.mutate_exp_input.get("1.0", "end-1c"))
        res = [int(self.res_h_input.get("1.0", "end-1c")), int(self.res_w_input.get("1.0", "end-1c"))]
        weight_variation = float(self.mutate_var_input.get("1.0", "end-1c"))
        comp_thresh = float(self.complexify_threshold_input.get("1.0", "end-1c"))
        cxs = self.cxs_input.get("1.0", "end-1c").replace("[","").replace("]","").split(",")
        cxs = [int(x) for x in cxs]
        self.individual.animated_connections = cxs
        self.num_frequencies = len(self.individual.animated_connections)
        
        frequencies = librosa.core.fft_frequencies(n_fft=2048*4)  # getting an array of frequencies
        times = librosa.core.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=self.sample_rate, hop_length=512, n_fft=2048*4)
        time_index_ratio = len(times)/times[len(times) - 1]
        frequencies_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]

        def get_decibel(target_time, freq):
            index0 = int(freq*frequencies_index_ratio)
            index0 = min(index0, len(self.spectrogram)-1)
            index1 = int(target_time*time_index_ratio)
            index1 = min(index1, len(self.spectrogram[0])-1)
            return self.spectrogram[index0][index1]

        len_in_seconds = max(times)
        max_decibel = np.max(self.spectrogram)
        num_per_second = 24
        total_samples = int(num_per_second * len_in_seconds)
        # total_samples = int(max(times) * time_index_ratio)
        print(total_samples)
        time_between_samples = 1/num_per_second

        num_bars = self.num_frequencies
        self.frequencies = np.arange(100, 8000, int(8000//num_bars))
        originals = []
        for cx in range(len(self.individual.connection_genome)):
            originals.append(self.individual.connection_genome[cx].weight)
        
        prog_bar = Progressbar(self.root, orient="horizontal", mode="determinate", maximum=total_samples)
        r=self.root.grid_size()[0]
        prog_bar.grid(row=0, column=0, sticky='NEWS', padx=1, pady=1, columnspan=20, rowspan=2)
        self.root.update()
        prog_bar.update()
        
        
        images = []
        average_decibels = []
        
        last_time = 0
        last_time = 0
        time_index = 0
        
        print(max(times), max(times) * time_index_ratio)
        
        def new_frame(time_index, last_time):
            # time_ = times[time_index]
            time_ = np.linspace(0, len_in_seconds, num=total_samples)[time_index]
            # print(time_)
            if time_ >= last_time + time_between_samples:
            # if(True):
                last_time = time_
                this_dec = []
                for freq_index, f in enumerate(self.frequencies):
                    dec = get_decibel(time_, f)
                    cx_index = self.individual.animated_cxs[freq_index]
                    w = ((dec/max_decibel)**mutate_exponent)*weight_variation
                    self.individual.connection_genome[cx_index].weight = (originals[cx_index]-weight_variation/2) + w 
                    this_dec.append(dec)
                average_decibels.append(np.mean(this_dec))
                d_decibels = 0
                if len(average_decibels) > 1 and time_ > 5:
                    d_decibels = max(0, average_decibels[-1] - average_decibels[-2]) # this is prob better
                    # d_decibels = max(0, average_decibels[-2] - average_decibels[-1])
                if d_decibels > comp_thresh:
                    self.individual.add_node()
                    self.individual.add_connection(1, self.config.allow_recurrent)
                    originals.append(self.individual.connection_genome[-1].weight)
                    originals.append(self.individual.connection_genome[-2].weight)
                    originals.append(self.individual.connection_genome[-2].weight)
                    self.individual.animated_cxs.append(len(self.individual.connection_genome)-1)
                    self.num_frequencies+=1
                    self.frequencies = np.arange(100, 8000, int(8000//self.num_frequencies))
                    self.update_title(100*time_/len_in_seconds)
                image = (self.individual.get_image(res[0], res[1], True) * 255.0).astype(np.uint8)
                prog_bar.step(1)
                images.append(image)
            
            time_index+=1
            
            if(time_>=len_in_seconds):
            # if(time_index >= len(times)):
                print(len(images))          
                make_video(images, name=None, fps=num_per_second, audio=self.music_file, show=False)
                for i, cx in enumerate(self.individual.connection_genome):
                    cx.weight = originals[i]
                prog_bar.stop()
                prog_bar.grid_forget()
                
            else:
                prog_bar.update()
                self.root.update()
                
                self.root.after(0, lambda time_index=time_index, last_time=last_time: new_frame(time_index, last_time))
                
        
        self.root.after(0, lambda: new_frame(time_index, last_time))
    def save_button_clicked(self):
        fail = False
        if self.individual is None:
            fail = True
        if self.time_series is None:
            fail = True
        if self.music_file == "" or self.music_file is None:
            fail = True
        if not fail:
            self.make_vis()
        else:
            tk.messagebox.showinfo("Error", "Please load an audio file.")
    
    def update_title(self, percent=None):
        self.root.title(f"Generation: {self.gen} | Nodes: {np.mean([len(self.individual.node_genome) for self.individual in [self.individual]]):.2f} | Connections: {np.mean([len(list(self.individual.enabled_connections())) for self.individual in [self.individual]]):.2f}{f' | {percent:.2f}%' if percent is not None else ''}")