import os
from random import shuffle
import random
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageOps
from matplotlib.pyplot import fill
import gtp3

class GUI():
    def __init__(self, root=None):
        self.asked_to_quit = False
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.title("Friendly Dragon Poem Generator")
        self.root.geometry("1600x900")
        self.command_button_size = (15, 5)
        button_font =("Arial", 14)
        # frames
        self.entry_frame = tk.Frame(self.root)
        self.output_frame = tk.Frame(self.root)
        self.commands_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.image_frame = tk.Frame(self.root)
        
        self.stack = []
        
        if getattr(sys, 'frozen', False):
            img = Image.open(os.path.join(sys._MEIPASS, "files/dragon.png"))
        else:
            img = Image.open("res/dragon.png")
        img =  ImageOps.mirror(img)
        # img = ImageOps.colorize(black=0, white=0, image=img)
        self.dragon_image = ImageTk.PhotoImage(img)
        dragon_label = tk.Label(self.image_frame, image=self.dragon_image).pack(fill=tk.BOTH, expand=True)
        
        # entry
        self.entry_label = tk.Label(self.entry_frame, text="Input", font=("Arial", 16, "bold"))
        self.entry_label.pack(side=tk.TOP)
        self.entry_box = tk.Text(self.entry_frame, height=25, width=68, font=("Arial", 14))
        self.entry_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.entry_box.tag_configure("center", justify='center')
        self.entry_box.bind('<KeyRelease>', lambda x:self.add_center_tag(self.entry_box))
        self.add_center_tag(self.entry_box)
        
        # output
        self.output_label = tk.Label(self.output_frame, text="Output", font=("Arial", 16, "bold"))
        self.output_label.pack(side=tk.TOP)
        self.output_box = tk.Text(self.output_frame, height=25, width=68, font=("Arial", 14))
        self.output_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.output_box.tag_configure("center", justify='center')
        self.output_box.bind('<KeyRelease>', lambda x:self.add_center_tag(self.output_box))
        self.add_center_tag(self.output_box)
        
        # commands
        self.right_to_left = tk.Button(self.root, text=u"\u261a", command=self.accept, font=("Arial", 25),height=2, width=3)
        
        self.save_output_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text=u"Save Output\n\U0001f4be", command=self.save_output)
        self.save_output_button.pack(side=tk.RIGHT)
        
        self.finish_gtp3_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Finish GTP3\n\u261e", command=self.finish_gtp3)
        self.finish_gtp3_button.pack(side=tk.LEFT)
        
        self.stanza_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Add Stanza Break\n\u261e", command=self.add_stanza_break)
        self.stanza_button.pack(side=tk.LEFT)

        self.add_line_break_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Add Line Break\n\u261e", command=self.add_line_break)
        self.add_line_break_button.pack(side=tk.LEFT)
        
        self.randomize_lines_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Randomize Lines\n\u261e", command=self.randomize_lines)
        self.randomize_lines_button.pack(side=tk.LEFT)
        
        self.randomize_words_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Randomize Words\n\u261e", command=self.randomize_words)
        self.randomize_words_button.pack(side=tk.LEFT)
        
        self.accept_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Accept\n\u261a", command=self.accept)
        self.accept_button.pack(side=tk.LEFT)
        
        self.undo_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Undo Accept\n\u238C", command=self.undo)
        self.undo_button.pack(side=tk.LEFT)
        
        self.load_button = tk.Button(self.commands_frame, font=button_font, width = self.command_button_size[0],height=self.command_button_size[1], text="Load\n\U0001F5C1", command=self.load)
        self.load_button.pack(side=tk.LEFT)
        
        # grid frames
        row = 0
        col = 0
        
        self.entry_frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        col += 1
        self.right_to_left.grid(row=row, column=col, sticky='ew')
        col+=1
        self.output_frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        col=0
        row+=1
        self.commands_frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5, columnspan=3)
        col=0
        row+=1
        self.image_frame.grid(row=row, column=col, rowspan=1, columnspan=3, sticky="nsew")
    
    def quit(self):
        self.asked_to_quit = True
        self.root.destroy()
        
    def load(self):
        f = filedialog.askopenfile(mode='r', defaultextension=".txt")
        if f is None: 
            return
        self.push_input_to_stack()
        self.entry_box.delete(1.0, tk.END)
        self.entry_box.insert(tk.END, "".join(f.readlines()))
        self.add_center_tag(self.entry_box)
        f.close() 
    
    def save_output(self):
        f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if f is None: 
            return
        text2save = str(self.output_box.get(1.0, tk.END)) # starts from `1.0`, not `0.0`
        f.write(text2save)
        f.close() 

    def randomize_lines(self):
        input_text = str(self.entry_box.get(1.0, tk.END))
        lines = input_text.split("\n")
        lines = [line.strip() for line in lines]
        # lines = [line for line in lines if line != ""]
        shuffle(lines)
        lines = "\n".join(lines)
        
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, lines)
        self.add_center_tag(self.output_box)
        
    def randomize_words(self):
        input_text = str(self.entry_box.get(1.0, tk.END))
        lines = input_text.split("\n")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != ""]
        for i in range(len(lines)):
            words = lines[i].split(" ")
            shuffle(words)
            words = " ".join(words)
            lines[i] = words
        lines = "\n".join(lines)
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, lines)
        self.add_center_tag(self.output_box)
        
    def add_stanza_break(self):
        input_text = str(self.entry_box.get(1.0, tk.END))
        lines = input_text.split("\n")
        lines.insert(random.randint(2, len(lines))-1, "\n")
        lines = "\n".join(lines)
        
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, lines)
        self.add_center_tag(self.output_box)
        
    def accept(self):
        self.push_input_to_stack()
        output = str(self.output_box.get(1.0, tk.END))
        self.entry_box.delete(1.0, tk.END)
        self.entry_box.insert(tk.END, output)
        self.add_center_tag(self.entry_box)
        
        
    def add_line_break(self):
        input_text = str(self.entry_box.get(1.0, tk.END))
        lines = input_text.split(" ")
        lines.insert(random.randint(2, len(lines))-1, "\n")
        lines = " ".join(lines)
        
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, lines)
        self.add_center_tag(self.output_box)

    def push_input_to_stack(self):
        self.stack.append(str(self.entry_box.get(1.0, tk.END)))

    def undo(self):
        if len(self.stack) == 0:
            return
        last = self.stack.pop()
        self.entry_box.delete(1.0, tk.END)
        self.entry_box.insert(tk.END, last)
        self.add_center_tag(self.entry_box)
        
    def finish_gtp3(self):
        output = gtp3.finish_poem(self.entry_box.get(1.0, tk.END))
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, output)
        self.add_center_tag(self.output_box)
        
    def add_center_tag(self, box):
        box.tag_add("center", "1.0", "end")
        
    def mainloop(self):
        self.root.mainloop()
        return self.asked_to_quit

        