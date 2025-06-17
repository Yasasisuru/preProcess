import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import tensorflow as tf

class ImagePreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preprocessing GUI")
        self.root.geometry("500x400")
        
        self.source_path = ""
        self.save_path = ""

        # GUI Layout
        tk.Button(root, text="Select Source Folder", command=self.select_source).pack(pady=10)
        tk.Button(root, text="Select Save Folder", command=self.select_save).pack(pady=10)
        tk.Button(root, text="Start Preprocessing", command=self.start_processing).pack(pady=10)

        self.label_status = tk.Label(root, text="", fg="blue")
        self.label_status.pack(pady=10)

        tk.Label(root, text="Individual Image Progress:").pack()
        self.image_progress = Progressbar(root, length=400, mode='determinate')
        self.image_progress.pack(pady=5)

        tk.Label(root, text="Overall Progress:").pack()
        self.total_progress = Progressbar(root, length=400, mode='determinate')
        self.total_progress.pack(pady=5)

    def select_source(self):
        self.source_path = filedialog.askdirectory()
        self.label_status.config(text=f"Source: {self.source_path}")

    def select_save(self):
        self.save_path = filedialog.askdirectory()
        self.label_status.config(text=f"Save To: {self.save_path}")

    def start_processing(self):
        if not self.source_path or not self.save_path:
            messagebox.showwarning("Missing Path", "Please select both source and save folders.")
            return

        image_files = [f for f in os.listdir(self.source_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total = len(image_files)

        if total == 0:
            messagebox.showinfo("No Images", "No valid image files found in the source folder.")
            return

        for i, filename in enumerate(image_files):
            self.label_status.config(text=f"Processing: {filename}")
            self.root.update()

            # Load and preprocess image
            img_path = os.path.join(self.source_path, filename)
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = img_array / 255.0  # Normalize

            # Save processed image
            save_file = os.path.join(self.save_path, filename)
            tf.keras.utils.save_img(save_file, img_array)

            # Update individual and overall progress
            self.image_progress['value'] = 100
            self.total_progress['value'] = ((i + 1) / total) * 100
            self.root.update()

            self.image_progress['value'] = 0  # Reset for next image

        self.label_status.config(text="Processing Complete")
        messagebox.showinfo("Done", "All images processed and saved successfully.")

# Launch the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePreprocessorApp(root)
    root.mainloop()
