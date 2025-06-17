import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from tkinter import Tk, Label, Button, filedialog, messagebox
from tkinter.ttk import Progressbar

IMAGE_SIZE = 224
PATCH_SIZE = 16

class ImagePreprocessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Preprocessor")

        self.source_folder = ''
        self.save_folder = ''

        self.label = Label(master, text="Select folders to begin")
        self.label.pack()

        self.select_source_button = Button(master, text="Select Source Folder", command=self.select_source_folder)
        self.select_source_button.pack()

        self.select_save_button = Button(master, text="Select Save Folder", command=self.select_save_folder)
        self.select_save_button.pack()

        self.start_button = Button(master, text="Start Processing", command=self.process_images)
        self.start_button.pack()

        self.image_progress = Progressbar(master, length=300, mode='determinate')
        self.image_progress.pack(pady=5)
        self.total_progress = Progressbar(master, length=300, mode='determinate')
        self.total_progress.pack(pady=5)

        self.status_label = Label(master, text="")
        self.status_label.pack()

    def select_source_folder(self):
        self.source_folder = filedialog.askdirectory()
        self.label.config(text=f"Source: {self.source_folder}")

    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory()
        self.label.config(text=f"Save: {self.save_folder}")

    def preprocess_image(self, image_path):
        # Open with PIL
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Noise Reduction using Gaussian blur
        img_np = np.array(img)
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        # Data augmentation
        if np.random.rand() > 0.5:
            img_np = cv2.flip(img_np, 1)  # horizontal flip

        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMAGE_SIZE/2, IMAGE_SIZE/2), angle, 1)
        img_np = cv2.warpAffine(img_np, M, (IMAGE_SIZE, IMAGE_SIZE))

        zoom = np.random.uniform(0.9, 1.1)
        zoomed = cv2.resize(img_np, None, fx=zoom, fy=zoom)
        h, w, _ = zoomed.shape
        if h > IMAGE_SIZE:
            start = (h - IMAGE_SIZE) // 2
            img_np = zoomed[start:start+IMAGE_SIZE, start:start+IMAGE_SIZE]
        else:
            pad = (IMAGE_SIZE - h) // 2
            img_np = cv2.copyMakeBorder(zoomed, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        img_pil = Image.fromarray(img_np)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(np.random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(np.random.uniform(0.8, 1.2))

        return img_pil

    def patch_embedding(self, image):
        image_np = np.array(image)
        patches = tf.image.extract_patches(
            images=tf.expand_dims(image_np, 0),
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        return patches

    def process_images(self):
        if not self.source_folder or not self.save_folder:
            messagebox.showerror("Error", "Please select both folders")
            return

        image_files = [f for f in os.listdir(self.source_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        total_images = len(image_files)
        self.total_progress["maximum"] = total_images
        self.status_label.config(text=f"Processing {total_images} images...")

        for idx, filename in enumerate(image_files):
            try:
                full_path = os.path.join(self.source_folder, filename)
                preprocessed_image = self.preprocess_image(full_path)

                # Optional patch embeddings
                _ = self.patch_embedding(preprocessed_image)

                save_path = os.path.join(self.save_folder, filename)
                preprocessed_image.save(save_path)

                self.image_progress["value"] = 100
                self.total_progress["value"] = idx + 1
                self.master.update_idletasks()
                self.status_label.config(
                    text=f"Processed {idx+1}/{total_images} images")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        self.status_label.config(text="All images processed successfully.")
        self.image_progress["value"] = 0

# ==== Run GUI ====
root = Tk()
app = ImagePreprocessorApp(root)
root.mainloop()
