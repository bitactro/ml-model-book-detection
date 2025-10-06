# ML Model — Book Detection

## Overview

This repository contains a machine learning-based **binary image classifier** that identifies whether a given image contains a **book** or **not**. It is built for automation scenarios such as:

* 📚 Document scanning workflows
* 🧾 Knowledge-base indexing
* 🗂 Content moderation and categorization

The project leverages **Transfer Learning (MobileNetV2)** for high performance on small datasets, making it lightweight and deployable on both **servers and edge devices**.

---

## Key Features

| Capability                        | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| ✅ Book vs Non-Book Classification | Outputs `True` / `False` for image inputs            |
| ✅ Optimized Inference             | Achieves ~50ms prediction time on CPU                |
| ✅ Command-line Prediction         | Simple script `predict_book.py` for batch inference  |
| ✅ Modular Training                | Easily retrain with new dataset via `train_model.py` |
| ✅ Clean Architecture              | Separated logic for Dataset, Training, and Inference |

---

## Project Structure

```
ml-model-book-detection/
│── dataset/              # (Ignored in Git) - Place training images here
│   ├── books/           # Book images
│   └── not_books/       # Non-book images
│
│── models/              # Saved Keras model (.h5 or SavedModel)
│── train_model.py       # Training script
│── predict_book.py      # Inference script
│── requirements.txt
│── README.md
```

---

## Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/bitactro/ml-model-book-detection.git
cd ml-model-book-detection

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running Inference

```bash
python predict_book.py
```

**Example Output:**

```
b1.jpeg -> isBookDetected: True
b2.jpeg -> isBookDetected: True
nb1.jpg -> isBookDetected: False
nb2.jpg -> isBookDetected: False
```

---

## Model Architecture

| Component           | Details                                  |
| ------------------- | ---------------------------------------- |
| Base Model          | MobileNetV2 (pretrained on ImageNet)     |
| Classification Head | GlobalAveragePooling + Dense(1, Sigmoid) |
| Loss Function       | Binary Crossentropy                      |
| Optimizer           | Adam (LR Scheduling Enabled)             |

---

## Training Notes

* **Data Augmentation:** Random flips, rotations, and scaling applied to improve generalization.
* **Transfer Learning:** Base model frozen initially; classification head trained first, then fine-tuned.
* **Batch Size & Epochs:** Configurable in `train_model.py`.
* **Validation Split:** Recommended 80/20 train/validation.

---

## Future Roadmap

* [ ] Convert to **ONNX / TensorRT** for optimized deployment
* [ ] Add **confidence thresholding** for probabilistic output
* [ ] Expand dataset for **multi-class book detection** (e.g., cover, open book, spine)
* [ ] Add visual output with bounding boxes or highlighted regions

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Ankit Mishra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

**Ankit Mishra**
Machine Learning Engineer • Open Source Contributor

```}
```
