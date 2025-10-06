import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("book_small.h5")
IMG_SIZE = (224, 224)

# Function to predict a single image
def isBookDetected(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    probabilityOfBook = 1-pred
    if probabilityOfBook > 0.65:
        return True
    return False

# Folder to predict
folder_path_book = "dataset/test/book" 
folder_path_not_book = "dataset/test/not_book" 


# Loop through all images in the folder
for img_file in os.listdir(folder_path_book):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path_book, img_file)
        result = isBookDetected(img_path)
        print(f"{img_file} -> isBookDetected: {result}")
print("\n")

for img_file in os.listdir(folder_path_not_book):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path_not_book, img_file)
        result = isBookDetected(img_path)
        print(f"{img_file} -> isBookDetected: {result}")

    
