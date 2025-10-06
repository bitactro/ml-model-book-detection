import os

train_dir = r"G:\Ankit Projects\ml-model-book-detection\dataset\train"

for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{cls}: {len(images)} images")
