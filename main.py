import tkinter as tk

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from InteractiveGUI import *
from InteractiveGUI import GUI as IECGUI
from MusicGUI import GUI as MUSICGUI
from poems_gui import GUI as POEMSGUI
from load_gui import GUI as LOADGUI
from config import Config
# from SCRNNGUI import GUI as SCRNNGUI

# from gan import GUI as GANGUI

def launch_iec(root):    
    root.wm_state('iconic')
    
    config = Config()
    config.load_saved("iec_config.json")
    gui = IECGUI(config,root)
    gui.run()

def launch_music(root):    
    root.wm_state('iconic')
    
    config = Config()
    config.load_saved("music_config.json")
    gui = MUSICGUI(config,root)
    gui.run()

def launch_gan(root):  
    return
    root.wm_state('iconic')
    # gui = GANGUI(root)
    
def launch_poems(root):    
    root.wm_state('iconic')
    gui = POEMSGUI(root)
    
def launch_loader(root):    
    root.wm_state('iconic')
    gui = LOADGUI(root)

def launch_scrnn(root):    
    return
    root.wm_state('iconic')
    gui = SCRNNGUI(root)
    gui.run()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("500x300")
    root.title("For Rachel")
    
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, fill=tk.BOTH)
    launch_iec_button = tk.Button(button_frame, text="Evolution", command=lambda: launch_iec(root), font=("Helvetica", 24))    
    launch_iec_button.pack(expand=True, fill=tk.BOTH)    
    launch_music_button = tk.Button(button_frame, text="Music", command=lambda: launch_music(root), font=("Helvetica", 24))    
    launch_music_button.pack(expand=True, fill=tk.BOTH)    
    # launch_music_button = tk.Button(button_frame, text="GAN", command=lambda: launch_gan(root), font=("Helvetica", 24))    
    # launch_music_button.pack(expand=True, fill=tk.BOTH)    
    launch_music_button = tk.Button(button_frame, text="Poems", command=lambda: launch_poems(root), font=("Helvetica", 24))    
    launch_music_button.pack(expand=True, fill=tk.BOTH)    
    # launch_music_button = tk.Button(button_frame, text="Upscaler", command=lambda: launch_scrnn(root), font=("Helvetica", 24))    
    # launch_music_button.pack(expand=True, fill=tk.BOTH)    
    root.mainloop()


