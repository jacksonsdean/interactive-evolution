import argparse
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from SRCNN import run as upscaler

class GUI():
    def __init__(self, root=None):
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.root.geometry("500x300")
        self.root.title("Upscale")

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(expand=True, fill=tk.BOTH)

        self.choose_image_button = tk.Button(self.button_frame, text="Choose Image", command=self.choose_image, font=("Helvetica", 24))
        self.choose_image_button.pack(expand=True, fill=tk.BOTH)
        self.upscale_button = tk.Button(self.button_frame, text="Upscale", command=self.upscale, font=("Helvetica", 24))
        self.upscale_button.pack(expand=True, fill=tk.BOTH)

        self.image_path=""

        self.image_preview = tk.Label(self.button_frame, text="No image selected")
        self.image_preview.pack(expand=True, fill=tk.BOTH)

        self.wait_text = tk.Label(self.button_frame, text="", font=("Helvetica", 24))
        self.wait_text.pack(expand=True, fill=tk.BOTH)

        self.upscale_button.config(state=tk.DISABLED)


    def choose_image(self):
        self.image_path  = filedialog.askopenfilename(multiple=False,
            filetypes=(('image files', ('.png', '.jpg')), ("all files", "*.*")))
        
        if self.image_path == "":
            return
        self.update_preview()
        self.upscale_button.config(state=tk.NORMAL)


    def update_preview(self):
        img = Image.open(self.image_path)
        self.image_tk = ImageTk.PhotoImage(img)
        self.image_preview.pack_forget()
        self.image_preview=tk.Label(self.button_frame, image=self.image_tk)
        self.image_preview.pack(expand=True, fill=tk.BOTH)

    def upscale(self):
        if self.image_path == "":
            tk.messagebox.showerror(title="Error", message="Please choose an image first!")
            return
        self.wait_text.config(text="Upscaling, please wait...")
        self.upscale_button.config(state=tk.DISABLED)
        self.choose_image_button.config(state=tk.DISABLED)
        self.root.update()
        new_filename = self.image_path.split(".")[0] + "_zoomed2x.png"

        parser = argparse.ArgumentParser(description='SRCNN run parameters')
        parser.add_argument('--model', type=str, default="./upscale_model_2x.pth")
        parser.add_argument('--image', type=str, default=self.image_path)
        parser.add_argument('--zoom_factor', type=int, default=2)
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--out', type=str, default=new_filename)
        args = parser.parse_args()
        upscaler.run(args)

        self.wait_text.config(text="Upscale finished!")
        self.upscale_button.config(state=tk.NORMAL)
        self.choose_image_button.config(state=tk.NORMAL)
        self.root.update()
        self.image_path = new_filename
        self.update_preview()



if __name__ == "__main__":
    gui = GUI()
    gui.root.mainloop()