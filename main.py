import tkinter as tk

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from interactive_gui import *
from interactive_gui import GUI as IEC_GUI
from music_gui import GUI as Music_GUI
from poems_gui import GUI as Poems_GUI
from load_gui import GUI as Loader_GUI
from config import Config

def launch_iec(root):    
    root.wm_state('iconic')
    gui = IEC_GUI(root)
    gui.run()

def launch_music(root):    
    root.wm_state('iconic')
    
    config = Config()
    config.load_saved("music_config.json")
    gui = Music_GUI(config,root)
    gui.run()

def launch_poems(root):    
    root.wm_state('iconic')
    gui = Poems_GUI(root)
    
def launch_loader(root):    
    root.wm_state('iconic')
    gui = Loader_GUI(root)
    gui.run()

def wildcard(root):
    for wid in root.winfo_children():
        de=("%02x"%random.randint(0,255))
        re=("%02x"%random.randint(0,255))
        we=("%02x"%random.randint(0,255))
        ge="#"
        color=ge+de+re+we
        wid.configure(bg = color, highlightbackground=color, font= ("Comic Sans MS", random.randint(15,25), "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("500x500")
    root.title("For Rachel")
    
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill=tk.BOTH)
    launch_iec_button = tk.Button(button_frame, text="Evolution", command=lambda: launch_iec(root), font=("Helvetica", 24))    
    launch_iec_button.pack(expand=True, fill=tk.BOTH)    
    launch_music_button = tk.Button(button_frame, text="Music", command=lambda: launch_music(root), font=("Helvetica", 24))    
    launch_music_button.pack(expand=True, fill=tk.BOTH)    
    launch_poems_button = tk.Button(button_frame, text="Poems", command=lambda: launch_poems(root), font=("Helvetica", 24))    
    launch_poems_button.pack(expand=True, fill=tk.BOTH)    
    launch_loader_button = tk.Button(button_frame, text="Load Genomes", command=lambda: launch_loader(root), font=("Helvetica", 24))    
    launch_loader_button.pack(expand=True, fill=tk.BOTH)    
    launch_wildcard_button = tk.Button(button_frame, text="?", command=lambda: wildcard(button_frame), font=("Helvetica", 24))    
    launch_wildcard_button.pack(expand=True, fill=tk.BOTH)    
    root.mainloop()


