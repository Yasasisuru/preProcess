import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from tkinter import Tk, Label, Button, filedialog, messagebox
from tkinter.ttk import Progressbar

# Settings
IMAGE_SIZE = 224
PATCH_SIZE = 16

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Preprocessing Tool")

        self.source_folder = ""
        self.save_folder = ""

        Label(master, text="Image Preprocessing (224x224 + Noise + Augmentation)").pack()

        Button(master, text="Select Source Folder", command=self.select_source_folder).pack(pady=2)
        Button(master, text="Select Save Folder", command=self.select_save_folder).pack(pady=2)
        Button(master, text="Start Processing", command=self.start_processing).pack(pady=10)

        self.image_progress = Progressbar(master, length=300, mode='determinate')
        self.image_progress.pack()
        self.total_progress = Progressbar(master, length=300, mode='determinate')
        self.total_progress.pack()

        self.status = Label(master, text="")
        self.status.pack(pady=5)

    def select_source_folder(self):
        self.source_folder = filedialog.askdirectory()
        self.status.config(text=f"Source: {self.source_folder}")

    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory()
        self.status.config(text=f"Save: {self.save_folder}")

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert to OpenCV for filtering
        img_cv = np.array(img)
        img_cv = cv2.medianBlur(img_cv, 3)  # Or use Gaussian: cv2.GaussianBlur(img_cv, (3, 3), 0)

        # Data Augmentation (flip, rotate, zoom)
        if np.random.rand() > 0.5:
            img_cv = cv2.flip(img_cv, 1)  # Horizontal flip

        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), angle, 1)
        img_cv = cv2.warpAffine(img_cv, M, (IMAGE_SIZE, IMAGE_SIZE))

        zoom_factor = np.random.uniform(0.9, 1.1)
        zoomed = cv2.resize(img_cv, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = zoomed.shape[:2]

        # Crop or pad to original size
        if zh > IMAGE_SIZE:
            start = (zh - IMAGE_SIZE) // 2
            img_cv = zoomed[start:start+IMAGE_SIZE, start:start+IMAGE_SIZE]
        else:
            pad = (IMAGE_SIZE - zh) // 2
            img_cv = cv2.copyMakeBorder(zoomed, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        # Convert to PIL
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # Brightness/Contrast
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(np.random.uniform(0.8, 1.2))

        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(np.random.uniform(0.8, 1.2))

        return img

    def patch_embedding(self, image):
        np_img = np.array(image)
        patches = tf.image.extract_patches(
            images=tf.expand_dims(np_img, axis=0),
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        return patches

    def start_processing(self):
        if not self.source_folder or not self.save_folder:
            messagebox.showwarning("Warning", "Please select both folders.")
            return

        image_files = [f for f in os.listdir(self.source_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        total = len(image_files)
        self.total_progress['maximum'] = total

        for i, filename in enumerate(image_files):
            try:
                full_path = os.path.join(self.source_folder, filename)
                img = self.preprocess_image(full_path)
                _ = self.patch_embedding(img)  # just to simulate patching, can be removed

                save_path = os.path.join(self.save_folder, filename)
                img.save(save_path)

                self.image_progress['value'] = 100
                self.total_progress['value'] = i + 1
                self.status.config(text=f"Processed {i+1}/{total}")
                self.master.update_idletasks()
            except Exception as e:
                print(f"Error: {e}")
                continue

        self.status.config(text="✅ All images processed.")
        self.image_progress['value'] = 0

# Run GUI
root = Tk()
app = ImageProcessorApp(root)
root.mainloop()
