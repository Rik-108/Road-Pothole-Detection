# Road Pothole Classification & Detection 🛣️🚗

## 📌 Project Overview
**Road Pothole Classification** is an end-to-end computer vision and web deployment project designed to automate the detection of road damage. By leveraging deep **Convolutional Neural Networks (CNNs)**, the system analyzes images of road surfaces and classifies them as either "Plain" or containing a "Pothole." To make this model accessible, it is fully integrated into a **Django** web application, allowing users to upload images and receive real-time structural assessments. 

## 🚀 Key Features
* **Multi-Architecture Experimentation:** Built and evaluated multiple CNN architectures (including a custom AlexNet-style network) across different modules to find the optimal balance of accuracy and computational efficiency.
* **Robust Image Augmentation:** Utilizes TensorFlow's `ImageDataGenerator` for dynamic rescaling, shearing, zooming, and horizontal flipping to prevent model overfitting on the training data.
* **Full-Stack Web Deployment:** Features a complete **Django** backend with asynchronous image uploading and routing, rendering the predictive results on a custom HTML frontend.
* **Seamless Model Integration:** Directly loads the saved `.h5` deep learning weights into the web application's views, bridging the gap between raw data science and user-facing software.

---

## 🏗️ Architecture Details

### The Computer Vision Engine (`TensorFlow/Keras`)
The predictive engine relies on a sequence of deep spatial feature extractors.
* **Layers:** Employs multiple blocks of `Conv2D` and `MaxPooling2D` to downsample the images and identify key structural anomalies (like the jagged edges of a pothole).
* **Classification Head:** Flattens the 2D feature maps into a 1D vector, passing it through dense layers with **ReLU** activations, culminating in a **Softmax** output for categorical prediction.

### The Application Backend (`Django`)
The server-side infrastructure handles user inputs and inference.
* **Routing (`Views.py`):** Uses Django's `TemplateView` and `DetailView` to securely handle image POST requests via `EmployeeForm` (acting as the image uploader).
* **Inference Pipeline:** Upon upload, the image is passed to the `Road Pothole` function, which dynamically loads the `e.h5` model, preprocesses the target image to the required `(225, 225)` tensor, and returns the classification result.

---

## ⚖️ Model Training Logic
In this framework, the model ($M$) is trained to distinguish complex visual textures:

1.  **Spatial Preprocessing:** Images are dynamically loaded from directories and normalized (pixel values scaled between 0 and 1).
2.  **Model Optimization:** The network updates its weights using the **RMSprop/Adam Optimizer** and minimizes the **Categorical Cross-Entropy** loss function. The model learns to output a probability distribution across the defined classes.

$$Loss = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$

*(Where $M$ is the number of classes, $y$ is the binary indicator of the correct class label, and $p$ is the predicted probability).*

---

## 📈 Evaluation Metrics
This project uses visual and statistical metrics to ensure the reliability of the computer vision model:
* **Accuracy & Loss Curves:** Implements custom `matplotlib` functions (`graph()`) to plot training vs. validation accuracy and loss over epochs, allowing for the visual detection of overfitting or bias.
* **Real-World Inference:** The final model is evaluated not just on test sets, but on fresh user uploads via the Django interface.

---

## 🛠️ Requirements
Ensure the following libraries are installed before running the deep learning training or starting the Django server:
```text
tensorflow
keras
django
numpy
matplotlib
pillow
h5py
```

---

## 👥 Contributors
* **Anik Basu** 
