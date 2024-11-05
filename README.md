---

# YOLOv11 Beer Bottle Detection Project 🍺🚀

Welcome to the **YOLOv11 Beer Bottle Detection Project**! 🎉 This project demonstrates how to use the **YOLO (You Only Look Once)** model for **real-time object detection**, specifically to identify **beer bottles 🍻** in images! 📸✨

---

## 📦 Installation

Before you dive into the fun, let’s get the necessary libraries installed. You’ll need **Ultralytics YOLO**, **Torch**, **Flask**, and other dependencies to run this project smoothly. Use the following command to install them:

```bash
%pip install ultralytics flask torch roboflow opencv-python
```

---

## 🛠️ Setup

### 1. **Import Libraries** 📚

To get started, you'll need to import the necessary libraries in Python. These libraries will help you load the YOLO model, run inference, and process images!

```python
import ultralytics
from ultralytics import YOLO
import torch
import cv2
```

### 2. **Check GPU Availability** ⚡️

If you have access to a GPU, it will speed up the process! Let’s check if your system has CUDA enabled:

```python
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
```

---

## 🚀 Train the YOLOv11 Model for Beer Detection 🍺

### 1. **Prepare Your Dataset** 🗂️

For optimal results, you'll need a **dataset of beer images** 🍻 (or any object you'd like to detect). Here’s an example `data.yaml` file for beer detection:

```yaml
# Dataset Configuration

path: '/content/drive/MyDrive/yolov11'  # Path to the dataset
train: '/content/drive/MyDrive/yolov11/images/train'  # Training images
val: '/content/drive/MyDrive/yolov11/images/val'      # Validation images

nc: 1  # Number of classes (1 for beer detection)
names: ['beer']  # List of class names
```

### 2. **Training the Model** 🎓

Now, let’s train the **YOLOv11** model using your custom dataset! 🧑‍🏫 This code will load the **YOLO model** and begin training:

```python
model = YOLO('yolo11m.pt')  # Load the pre-trained YOLOv11 model
results = model.train(
    data='/content/drive/MyDrive/yolov11/data.yaml',  # Path to your dataset YAML file
    epochs=100,  # Number of epochs for training
    device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)
```

---

## 🔍 Run Inference on Images 📸

Once the model is trained, it’s time to test it out! 🎯 Here’s how you can use the trained model to detect beer bottles 🍺 in images:

### 1. **Specify Your Image Paths** 📷

Provide the paths to the images you want to test. For example:

```python
image_paths = [
    '/content/drive/MyDrive/beer images/beer-1218742_640.jpg',
    '/content/drive/MyDrive/beer images/beer-199650_640.jpg',
    '/content/drive/MyDrive/beer images/beer-2370783_640.jpg'
]
```

### 2. **Run the Model** 🎯

Use this code to run the trained model on your test images:

```python
results = model(image_paths)  # Run YOLOv11 on the images
```

The model will output bounding boxes around detected beer bottles 🍻! 🎉

---

## 🖼️ Visualizing and Saving Results 🎉

Once the inference is done, you can display and save the annotated images. Here’s how:

### 1. **Save Annotated Images** 💾

You can save each output image with the beer bottles highlighted as follows:

```python
for result in results:
    output_image_path = '/path/to/save/output_' + os.path.basename(image_path)
    result.save(filename=output_image_path)  # Save the annotated image
    print(f"Saved output image to {output_image_path}")
```

### 2. **Display the Results** 👀

You can also display the results directly in your Python environment:

```python
result.show()  # Display the image with bounding boxes around detected beer bottles
```

---

## 💻 Flask Web App Integration 🖥️

Let’s add some interactivity! 🚀 You can turn this into a **Flask Web App** where you can upload images and get real-time beer bottle detection results 🍻.

---

## 📂 Project Folder Structure 📁

Here’s how your project directory should look:

```
/beer-detection-yolov11/
    ├── model/                # Folder for YOLO model weights
    ├── uploads/              # Folder for storing uploaded images
    ├── output_images/        # Folder for saving output images
    ├── static/               # Folder for CSS, JS files
    ├── templates/            # HTML templates for Flask app
    ├── requirements.txt      # Python dependencies for the app
    └── README.md             # This file
```

---

## 📈 Customizing the Script

Feel free to tweak the code to fit your own use case! Here are a few ideas:

- **Custom Weights**: You can replace `yolo11m.pt` with your own **custom-trained YOLOv11 model** for different object detection tasks. 🎨
- **New Datasets**: Modify the `data.yaml` to use your own dataset for detecting **other objects**! 🌟
- **Test with New Images**: Update the `image_paths` with new images to test out the model. 📸

---

## 📝 Example Outputs

Here’s what you can expect after running the inference:

- **Input Image** 📸: A photo of a beer 🍻

    ![beer-5910451_640](https://github.com/user-attachments/assets/286359ef-27bf-4035-92d5-b93df7042c65)

- **Output Image** 🎯: Same image with bounding boxes 🟩 around the detected beer 🍻

    ![output_beer-5910451_640](https://github.com/user-attachments/assets/18d297be-9068-4957-ac9a-359a44857c4c)

---

## 🎨 Customize Your Project

- **Model Weights**: Swap out the model file for a custom-trained version.
- **Dataset**: Train the model on your own dataset for any object detection task!
- **Images**: Test it on different images (beers, cats, dogs, anything)! 🐶🐱🍔

---

## 💬 Questions or Feedback? 💭

We’re here for it! If you have questions, issues, or suggestions, feel free to open an issue or send a pull request! 📝✨

Let’s start detecting those **beer bottles** 🍺! 🥳

---

## 🎉 Enjoy and Happy Coding! 💻👨‍💻👩‍💻

---

With this updated README, I’ve removed references to the `app.py`, `index.html`, and `result.html` files, and added more fun emojis throughout! I hope this makes the README more lively and engaging for users! 🎉🍻
