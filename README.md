
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

- **Input Image**:  A photo of a beer 🍻
-  ![beer-5910451_640](https://github.com/user-attachments/assets/32093f5b-ecfd-4e90-be4d-347be14cc7b5)

- **Output Image**: Same photo with bounding boxes 🟩 around the beer 🍻
- ![output_beer-5910451_640](https://github.com/user-attachments/assets/18d297be-9068-4957-ac9a-359a44857c4c)
---

## 📸 Example Images

Here are some example images you can test with (make sure the paths are correct!):

- **Beer Image 1**:
  ![5](https://github.com/user-attachments/assets/286359ef-27bf-4035-92d5-b93df7042c65)
- **Beer Image 2**:
   ![3](https://github.com/user-attachments/assets/62525fbb-d451-47d5-b205-1056f76f7db6)
- **Beer Image 3**: ![4](https://github.com/user-attachments/assets/04fd786c-0254-4235-a0d0-1cde3a4d8a79)
- **Beer Image 4**: ![1](https://github.com/user-attachments/assets/28c8f95b-99e3-426e-a1b8-8a87c7034f4d)
- **Beer Image 5**: ![2](https://github.com/user-attachments/assets/33ea8e9e-bbe3-4bb3-a3c1-096d412637f6)

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



I see! You want to keep the **readme** focused on the project itself and remove references to `app.py`, `index.html`, and `result.html`. Also, you'd like me to add more emojis to make the readme more engaging. Here’s an updated version with those changes:

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
