from distutils.log import error
import tkinter as tk
from tkinter import ttk
from vqgan_clip import *
from matplotlib.pyplot import fill
from tkinter import filedialog
import time
from PIL import ImageTk, Image



class GUI:
    model_dict = {"ImageNet1024 (general/small/fast)": "vqgan_imagenet_f16_1024", "ImageNet16384 (general/big/slow)": "vqgan_imagenet_f16_16384", "WikiArt (art)": "wikiart_16384", "Coco (stuff)": "coco"}
    
    def __init__(self, root=None):
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.width = 256
        self.height = 256
        self.iterations = 250
        self.update_freq = 2
        self.do_video = tk.BooleanVar(root, value=True)
        
        
        self.model_variable = tk.StringVar(self.root)
        self.model_variable.set("ImageNet1024 (general/small/fast)") # default value

        self.model = GUI.model_dict[self.model_variable.get()]
        
        self.style = ttk.Style(self.root)
        # add label in the layout
        self.style.layout('text.Horizontal.TProgressbar', 
                    [('Horizontal.Progressbar.trough',
                    {'children': [('Horizontal.Progressbar.pbar',
                                    {'side': 'left', 'sticky': 'ns'})],
                        'sticky': 'nswe'}), 
                    ('Horizontal.Progressbar.label', {'sticky': 'nswe'})])
        self.style.configure('text.Horizontal.TProgressbar', text='0 %', anchor='center')
        
        
        self.root.title("VQGAN + CLIP")
        self.root.geometry("600x800")
        self.select_image_button = tk.Button(self.root, text="Select Starting Image (Optional)", command=self.select_image, font=("Helvetica", 12))
        self.clear_image_button = tk.Button(self.root, text="Clear Image", command=self.clear_image)
        self.prompt_label = tk.Label(self.root, text="Prompt:", font=("Helvetica", 20))
        self.prompt_text_box = tk.Text(self.root, height=10, width=20)
        self.go_button = tk.Button(self.root, text="Go!", command=self.run)
        self.settings_button = tk.Button(self.root, text="Settings", command=self.settings_window)
                
        
        self.select_image_button.pack(fill=tk.BOTH, expand=True)
        self.clear_image_button.pack(fill=tk.BOTH, expand=True)
        self.settings_button.pack(fill=tk.BOTH, expand=True)
        self.prompt_label.pack(fill=tk.BOTH, expand=True)
        self.prompt_text_box.pack(fill=tk.BOTH, expand=True)
        
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop)
        self.stop_button.pack(fill=tk.BOTH, expand=True)
        self.stop_button.pack_forget()
        self.go_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.pvariable = tk.DoubleVar(root, value=0)
        self.progress_bar = tk.ttk.Progressbar(self.root, style='text.Horizontal.TProgressbar', orient=tk.HORIZONTAL, length=self.iterations, mode="determinate", variable=self.pvariable)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.filename = None
        
        self.image_preview = tk.Label(self.root)
        
        self.requires_stop = False
                
                

        
    def settings_window(self):
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        # self.settings_window.geometry("300x200")
        # self.settings_window.resizable(False, False)
        
        width_label = tk.Label(self.settings_window, text="Width:", anchor=tk.W, font=("Helvetica", 12))
        height_label = tk.Label(self.settings_window, text="Height:", anchor=tk.W, font=("Helvetica", 12))
        width_entry = tk.Entry(self.settings_window, width=5, font=("Helvetica", 12))
        height_entry = tk.Entry(self.settings_window, width=5, font=("Helvetica", 12))
        iterations_label = tk.Label(self.settings_window, text="Iterations:", anchor=tk.W, font=("Helvetica", 12))
        iterations_entry = tk.Entry(self.settings_window, width=5, font=("Helvetica", 12))
        update_freq_label = tk.Label(self.settings_window, text="Preview update frequency:", anchor=tk.W, font=("Helvetica", 12))
        update_freq_entry = tk.Entry(self.settings_window, width=5, font=("Helvetica", 12))
        save_video_label = tk.Label(self.settings_window, text="Save video?", anchor=tk.W, font=("Helvetica", 12))
        save_video_check = tk.Checkbutton(self.settings_window, variable=self.do_video, font=("Helvetica", 12))
        model_label = tk.Label(self.settings_window, text="Model dataset:", anchor=tk.W, font=("Helvetica", 12))
        
        model_dropdown = tk.OptionMenu(self.settings_window, self.model_variable, "ImageNet1024 (general/small/fast)", "ImageNet16384 (general/big/slow)", "WikiArt (art)", "Coco (stuff)")
        
        def save_settings():
            self.width = int(width_entry.get())
            self.height = int(height_entry.get())
            self.iterations = int(iterations_entry.get())
            self.update_freq = int(update_freq_entry.get())
            self.model = GUI.model_dict[self.model_variable.get()]
            
            if self.model == "coco":
                win = tk.Toplevel(self.root)
                win.title("Warning")
                error_label = tk.Text(win, height=5, font=("Helvetica", 12), wrap=tk.WORD, bg="white")
                error_label.insert(tk.END, "Coco requires an additional file to be downloaded from https://dl.nmkd.de/ai/clip/coco/coco.ckpt. It's a big boi (almost 8 GB). If you'd like to continue, please follow the link to download the file and put coco.ckpt in the same folder as coco.yaml. Otherwise, please go back into settings and choose another model.",)
                error_label.pack()
                error_label.configure(state="disabled")
                tk.Button(win, text="OK", command=win.destroy).pack(fill=tk.BOTH, expand=True)
            self.settings_window.destroy()
        
        save_button = tk.Button(self.settings_window, text="Save", command=save_settings)
        
        width_entry.insert(0, self.width)
        height_entry.insert(0, self.height)
        iterations_entry.insert(0, self.iterations)
        update_freq_entry.insert(0, self.update_freq)
        
        # add to window
        width_label.grid(row=0, column=0)
        height_label.grid(row=1, column=0)
        width_entry.grid(row=0, column=1)
        height_entry.grid(row=1, column=1)
        iterations_label.grid(row=2, column=0)
        iterations_entry.grid(row=2, column=1)
        update_freq_label.grid(row=3, column=0)
        update_freq_entry.grid(row=3, column=1)
        save_video_label.grid(row=4, column=0)
        save_video_check.grid(row=4, column=1)
        model_label.grid(row=5, column=0)
        model_dropdown.grid(row=5, column=1)
        
        save_button.grid(row=6, column=0, columnspan=2)
        

    def stop(self):
        self.stop_button.pack_forget()
        self.go_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.progress_bar.pack_forget()
        self.requires_stop = False
        cancel_gan()
        if self.do_video.get():
            save_video(dir="steps", name=f"videos/{time.time()}.mp4")

    def clear_image(self):
        self.filename = None
        self.image_preview.pack_forget()
        self.image_preview = tk.Label(self.root, image=None)
        self.image_preview.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        
    def select_image(self):
        # open tk file dialog
        f_types = [('jpg Files', '*.jpg'),('PNG Files','*.png')]    
        self.filename = filedialog.askopenfilename( multiple=True,
            filetypes=f_types)[0]
        self.img = ImageTk.PhotoImage(Image.open(self.filename))  # PIL solution
        self.image_preview.pack_forget()
        
        self.image_preview = tk.Label(self.root, image=self.img)
        self.image_preview.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    def progress_callback(self, progress, image=None):
        if image is not None:
            self.img = ImageTk.PhotoImage(Image.open(image))  # PIL solution
            self.image_preview.pack_forget()
            self.image_preview = tk.Label(self.root, image=self.img)
            self.image_preview.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        else:
            self.pvariable.set(progress/self.iterations*100)
            # self.progress_bar.step()
            self.style.configure('text.Horizontal.TProgressbar', 
                    text='{:d}/{:d} ({:g} %)'.format(progress, self.iterations, self.pvariable.get()))  # update label
            
        self.root.update()
        
    def done_callback(self, image):
        self.img = ImageTk.PhotoImage(Image.open(image))  # PIL solution
        self.image_preview.pack_forget()
        self.image_preview = tk.Label(self.root, image=self.img)
        self.image_preview.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # self.image_preview.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def run(self):
        self.requires_stop = True
        
        self.pvariable.set(0)
        
        # get prompt
        self.stop_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # self.progress_bar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.go_button.pack_forget()
        self.root.update()
        self.root.update_idletasks()
        
        
        prompt = self.prompt_text_box.get("1.0", tk.END)
        run_gan(prompt, self.filename, modelo=self.model,progress_callback=self.progress_callback, intervalo_imagenes=self.update_freq, width=self.width, height=self.height, done_callback=self.done_callback,max_iteraciones=self.iterations, custom_save=f"saved_imgs/gan_output{time.time()}.png")
        
        if self.requires_stop:
            self.stop()
            


