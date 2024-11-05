
---

# YOLOv11 Image Detection Project 🚀

Welcome to the **YOLOv11 Image Detection Project**! 🎉 This exciting project demonstrates how to use the YOLO (You Only Look Once) model for real-time object detection, specifically identifying beer bottles 🍺 in images! 📸✨

---

## 📦 Installation

To get started, you'll need to install the required packages. Use the following command to install them:

```bash
%pip install ultralytics supervision roboflow
```

---

## 🛠️ Setup

### 1. **Import Libraries** 📚
First, import the necessary libraries to get your project up and running:

```python
import ultralytics
from ultralytics import YOLO
import torch
import cv2
```

### 2. **Check GPU Availability** ⚡️

Before training the model, check if a GPU is available for faster performance!

```python
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
```

---

## 🚀 Load and Train the YOLO Model

### 1. **Prepare Your Dataset** 🍻

Make sure your dataset is organized properly, especially if you're training on your own dataset. Create a `data.yaml` file for your project like this:

```yaml
# Dataset Configuration

path: '/content/drive/MyDrive/yolov11'  # Path to the dataset directory
train: '/content/drive/MyDrive/yolov11/images/train'  # Directory for training images
val: '/content/drive/MyDrive/yolov11/images/val'      # Directory for validation images

nc: 1  # Number of classes (1 for beer detection)
names: ['beer']  # List of class names
```

### 2. **Training the Model** 🎓

Now, you can load the YOLO model and train it on your custom dataset:

```python
model = YOLO('yolo11m.pt')  # Load the pre-trained YOLOv11 model
results = model.train(
    data='/content/drive/MyDrive/yolov11/data.yaml',  # Update with your data.yaml path
    epochs=100,
    device='0' if torch.cuda.is_available() else 'cpu'
)
```

---

## 🔍 Run Inference

Once your model is trained, it’s time to test it! 🎉

1. **Specify the Paths to Your Images** 📸:

Provide the paths to the images you want to test:

```python
image_paths = [
    '/content/drive/MyDrive/beer images/beer-1218742_640.jpg',
    '/content/drive/MyDrive/beer images/beer-199650_640.jpg',
    '/content/drive/MyDrive/beer images/beer-2370783_640.jpg',
    '/content/drive/MyDrive/beer images/beer-422138_640.jpg',
    '/content/drive/MyDrive/beer images/beer-5910451_640.jpg',
    '/content/drive/MyDrive/beer images/beer-820011_640.jpg'
]
```

2. **Run the Model** 🎯:

```python
results = model(image_paths)
```

This will run inference on the specified images, detecting beer bottles 🍺 in them. 🎉

---

## 🖼️ Display and Save Results

Once the inference is complete, you can visualize the results and save the annotated images.

### 1. **Save the Results** 💾

```python
for result in results:
    output_image_path = '/path/to/save/output_' + os.path.basename(image_path)
    result.save(filename=output_image_path)  # Save the annotated image
    print(f"Saved output image to {output_image_path}")
```

### 2. **Show the Results** 👀

```python
result.show()  # Optionally display the results
```

This will show each processed image with the detected beer bottles and bounding boxes drawn around them! 🍻✨

---

## 💻 Code Explanation

### 1. **Model Loading** 🧠

The `YOLO()` function loads the YOLOv11 model from a pre-trained weight file (like `yolo11m.pt`).

### 2. **Training** 🏋️‍♂️

The `model.train()` function allows you to train the model using your custom dataset. It takes parameters such as the path to your dataset (`data.yaml`), number of epochs, and device (GPU or CPU).

### 3. **Inference** 🏃‍♂️

The `model()` method runs inference on the provided images. It will return a list of results containing the detected objects.

### 4. **Saving and Displaying Results** 📸

- `result.save()` saves each image with the detected objects annotated with bounding boxes.
- `result.show()` allows you to visually inspect the detection results.

---

## 💡 Customizing the Script

- **Model Weights**: You can replace `yolo11m.pt` with your own custom-trained YOLO model weights for different object detection tasks.
- **Images**: Update the image paths to test the model on your own images 🍻.
- **Output Directory**: Modify the output directory path to specify where you want to save the annotated images 📍.

---

## 📝 Example Output

After running the inference, the script will output images with bounding boxes drawn around any detected beer bottles! 🍺

**Example:**
- **Input Image**: A photo of a beer 🍻
- **Output Image**: Same photo with bounding boxes 🟩 around the beer 🍻

---

## 📸 Example Images

Here are some example images you can test with (make sure the paths are correct!):

- **Beer Image 1**: Beer Image 1
- **Beer Image 2**: Beer Image 2
- **Beer Image 3**: Beer Image 3
- **Beer Image 4**: Beer Image 4
- **Beer Image 5**: Beer Image 5
- **Beer Image 6**: Beer Image 6

---

## 🎉 Conclusion

This project is a fun and engaging way to explore object detection using YOLOv11! 🚀 Whether you're detecting 🍺 beer bottles or other objects, YOLO makes it quick and easy. Feel free to modify the code, train on your own datasets, and share your results! 💬🥳✨

---

## 💬 Questions or Feedback?

Feel free to open an issue or submit a pull request if you have any questions or improvements! 🚀💬

Let's detect those 🍺 beer bottles! 🥳

---

### 🚀 Enjoy and Happy Coding! 💻👨‍💻👩‍💻

---
