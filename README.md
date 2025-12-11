# 😴 Drowsiness Detection System (Open/Closed Eye Classification)

## 💡 Project Overview

This project focuses on building a highly accurate binary classification system to determine the state of an eye (Open or Closed). This is a foundational and critical step for real-time applications such as **Driver Drowsiness Detection Systems**.

The project was developed in full compliance with the academic requirements:
1.  Exclusive use of the **PyTorch** library for model design and training.
2.  Selection of a complex, industry-relevant problem (Driver Safety Systems).
3.  Delivery of an interactive Graphical User Interface (GUI) for application running.

**Technical Stack:**
* **Model Architecture:** DrowsinessCNN – A custom Convolutional Neural Network (CNN).
* **GUI Framework:** **Flask** (using HTML, CSS, and JavaScript)








---

## ✅ Project Phases and Deliverables

The workflow was divided into the required seven core phases. Below is a detailed documentation of the progress in each stage, based on the implemented code:

### 1. Problem Definition and Data Collection

* **Problem Definition:** To classify static eye images into two distinct categories: **Closed (0)** and **Open (1)**, supporting subsequent drowsiness detection algorithms.
* **Data Collection:** The **Open Closed Eyes Dataset** was downloaded from Kaggle.
* **Initial Setup:** Data directories were organized and prepared for processing.

### 2. Data Cleaning and Analysis (if applicable)

* **Data Cleaning:** A crucial step of renaming files was executed to ensure proper labeling: image files were prefixed with **`closed_`** or **`open_`** based on their classification.
* **Data Aggregation:** All images from the 'Open' and 'Closed' categories were successfully aggregated into a single directory (`/content/train/dataset`) for unified PyTorch data loading.

### 3. Feature Engineering (if applicable)

* **Data Augmentation:** To enhance model robustness and prevent overfitting, the following augmentations were applied to the training set:
    * **RandomHorizontalFlip** ($p=0.5$)
    * **RandomRotation** (up to $15^\circ$)
* **Normalization:** Image tensors were normalized using the standard **ImageNet Mean** and **Standard Deviation** for efficient deep learning training.

### 4. Model Design

* **Model Architecture:** A custom CNN model named **`DrowsinessCNN`** was implemented from scratch using PyTorch.
* **Structure:** The model features 4 sequential convolutional blocks, each utilizing **Conv2D**, **BatchNorm2d**, **ReLU**, and **MaxPool2d** layers. The classification head uses dense layers with **Dropout (p=0.5)**.
* **Library:** Designed exclusively using the **`torch.nn`** module.

### 5. Model Training

* **Training Environment:** **[Specify the environment, e.g., Google Colab GPU / AWS / Local Machine]**
* **Hyperparameters:**
    * **Epochs:** **15** (managed with Early Stopping)
    * **Batch Size:** **128**
    * **Optimizer:** **AdamW**
    * **Learning Rate (LR):** **$1e-4$**
    * **Loss Function:** **nn.BCEWithLogitsLoss()**
* **Best Model Saving:** An **Early Stopping** mechanism (Patience = 5) was implemented to save the model weights that achieved the lowest **Validation Loss**.

### 6. Model Testing and Inference

* **Data Splitting:** The aggregated dataset was split into Train, Validation, and Test sets (typically 80/10/10 split).
* **Evaluation Metrics:** Testing was conducted on the unseen Test Set using the best-saved weights. Key metrics included the **Confusion Matrix** and the detailed **Classification Report** (Precision, Recall, F1-Score).
* **Performance Results (To be filled after execution):**
    * **Overall Accuracy:** **[Percentage %]**
    * **Closed Class F1-Score:** **[Value]**
    * **Open Class F1-Score:** **[Value]**

### 7. GUI Implementation and Application Running

### 🖥 Interface Overview
The graphical interface of the drowsiness-detection system is built using:

- **Flask** → Backend web framework  
- **HTML, CSS, JavaScript** → Frontend interface  
- **MediaPipe FaceMesh** → Face & eye landmark detection  
- **PyTorch CNN Model** → Eye state classification (Open/Closed)

### ⚙️ How the Interface Works
- The system streams real-time video from the webcam through the `/video_feed` route.
- MediaPipe extracts eye regions from each frame.
- Each eye is passed to the CNN model (running in inference mode using `model.eval()`).
- Predictions are sent back to the frontend through the `/status` API endpoint.
- JavaScript updates the driver state on the page:
  - **AWAKE** if average prediction ≥ 0.5  
  - **SLEEPY** otherwise
- HTML/CSS are used to display a clean, responsive dashboard.

### 🔧 Deployment
- The application runs locally using Flask.
- The video feed uses MJPEG streaming so it works on any modern browser.
- No installation required on the client side — only the Flask server needs to run.

---

## 👥 Team Task Distribution

| Team Member | Core Responsibilities and Focus Areas |
| :--- | :--- |
| **Sayed** | Problem Definition, Data Collection (Kaggle), Initial Data Setup |
| **Ibrahim** | Data Cleaning and Analysis, Feature Engineering (Data Augmentation) |
| **Emad** | Model Design (DrowsinessCNN Architecture), PyTorch Implementation, Data Loaders Setup |
| **Karim** | Model Training, Hyperparameter Tuning, Implementing Early Stopping |
| **Hatem** | Model Testing and Evaluation, Confusion Matrix, Classification Report Generation |
| **Morsi** | GUI Implementation (Flask + HTML/CSS/JavaScript), Application Running and Deployment Setup|
 

---

## 🛠️ Running the Project

Follow these steps to run the application locally:

### 1. Prerequisites

First, install all necessary libraries, including PyTorch and Flask, using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

python web2.py

http://localhost:5000




