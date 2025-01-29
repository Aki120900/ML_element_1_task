# Image Classification using MobileNetV2  
**Machine Learning Model for Classifying Images into Four Categories**  

---

## 1. Project Overview  

This project implements an **image classification system** using **deep learning**. The goal is to **train a model that classifies images into four categories** based on the folder names in the dataset.  

**Task:**
You can complete the following task using any programming language and ML framework as you wish. 

1. Create an image preprocessing pipeline that prepares the images and potentially new images to be passed through an ML model that you will design in the second step.

2. Design, write, and train an ML model that will perform the classification of the provided image data. The classes required correspond to the folder name of the provided dataset.
You should also include either in comments or in a separate file your reasoning for choosing said ML model and document your training approach.

3. The trained model should be exported. A separate program be written that can accept an image not previously included in the dataset and classification be performed on it using the trained model.

For this task, you will be using the images provided in the “images” folder of the provided dataset. 
The datasets to be used for the task are available here: https://drive.google.com/drive/folders/1UtgQWE3AO0GLvom-osrJVQ7M6y83MajF?usp=sharing 


My *structure* preparation for this task included:  
1. **Preprocessing images** to prepare them for deep learning.  
2. **Training a MobileNetV2-based neural network** for classification.  
3. **Deploying the model** to recognise unseen images based on trained categories.  

**Why MobileNetV2?**  
The decision to use in this project **MobileNetV2** was made regarding the following **reasons**:  
✅ **Pre-trained on ImageNet** – Works well even with a small dataset.  
✅ **Lightweight and Efficient** – Ideal for real-time classification.  
✅ **Good Generalization** – Transfer learning helps adapt to new datasets quickly.  

---

## 2. Dataset & Preprocessing  

### 📂 Dataset Structure  
The dataset consists of **images structured into folders**, where each folder name represents the **class label**:  

```
Images/
│── bicycles/
│── car/
│── deer/
│── mountains/
```

### Preprocessing Steps  
The following steps were added to **improve model accuracy**:  
✅ **Removing corrupted images** (some images were found to be truncated).  
✅ **Resizing images** to `224x224` (required by MobileNetV2).  
✅ **Scaling pixel values** to `[0,1]` range for better training.  
✅ **Splitting dataset** into training (`80%`) and validation (`20%`).  

**Issues Encountered & Fixes:**  
1️⃣ *Corrupted images (`OSError: image file is truncated`)* → **Automatically detected and removed.**  
2️⃣ *`.DS_Store` files appearing in datasets* → **Ignored in preprocessing.**  

---

## 3. Model Selection: Why MobileNetV2?  

### Why not a Custom CNN?  
- Custom CNNs **take longer to train** and **require more data** to generalise well.  
- MobileNetV2 is **pre-trained on ImageNet**, making it an ideal **transfer learning** approach.  

### Why MobileNetV2?  
✅ **Pre-trained on ImageNet** – Works well even with limited data.  
✅ **Highly Efficient** – Uses **depthwise separable convolutions** to be **lightweight and fast**.  
✅ **Great Generalization** – Transfer learning allows quick adaptation to new datasets.  

**Model Architecture**:  
- MobileNetV2 **base model** (pre-trained on ImageNet).  
- **Global Average Pooling layer** to reduce feature dimensions.  
- **Fully connected (Dense) layers** with **ReLU** and **Softmax** activation.  
- **Dropout layers** to prevent overfitting.  

---

## 4. Training Approach & Hyperparameters  

### Hyperparameters Used:  
- **Optimiser**: `Adam` (`lr=0.001` initially, dynamically reduced)  
- **Loss Function**: `categorical_crossentropy` (for multi-class classification)  
- **Batch Size**: `32`  
- **Epochs**: `20` (Early stopping applied)  
- **Learning Rate Reduction**: `ReduceLROnPlateau` (auto-adjusts learning rate)  
- **Regularisation**: `Dropout layers` to prevent overfitting  

### Training Enhancements:  
✅ **`EarlyStopping`** → Stops training if validation loss stops improving.  
✅ **`ReduceLROnPlateau`** → Lowers learning rate if model performance plateaus.  

---

## 5. Results & Analysis  

### Final Model Performance:  

| Metric         | Training | Validation |  
|---------------|---------|------------|  
| **Accuracy**  | 99.9%   | 93.3%      |  
| **Loss**      | 0.0449  | 0.2496     |  

### Observations:  
✔ **Good Generalisation** – Validation accuracy is close to training accuracy.  
✔ **Overfitting Controlled** – `ReduceLROnPlateau` helped avoid overfitting.  
✔ **Fast Convergence** – High accuracy achieved in just **7 epochs**.  


**Potential Improvements:**  
1️⃣ **More data augmentation techniques** could be applied to further improve generalization.  
2️⃣ **Training for a few more epochs** with gradual learning rate decay could refine results.  

---

## 6. Issues that have been fixed during progress  

###  1. Image File Corruption Errors  
- **Error:** `OSError: image file is truncated (51 bytes not processed)`  
- **Fix:** A function was implemented to scan for and **remove corrupted images** before training.  

### 2. Unexpected `.DS_Store` Files  
- **Error:** macOS automatically creates `.DS_Store` files in directories.  
- **Fix:** The code was updated to **ignore non-image files** when processing data.  

### 3. Model Architecture Warning  
- **Error:** `UserWarning: Do not pass an input_shape argument to a layer.`  
- **Fix:** The model was **modified to use an Input layer explicitly** instead of passing `input_shape` in layers.  

### 4. Classification Predictions Returning Numerical Labels  
- **Error:** `Predicted class: 0` instead of `cars`, `deer`, etc.  
- **Fix:** The final prediction output was **mapped to class labels**.  

---

## 7. How to Train & Use the Model  

### Step 1️⃣: Train the Model  
Run the following command:  
```bash
python image_prepare.py
```
- Select `'train'` when prompted.  
- The model will train and save as **`trained_model.keras`**.  

### Step 2️⃣: Classify an Image  
Run the following command:  
```bash
python image_prepare.py
```
- Select `'recognise'` and **enter the image path**.  
- Example Output:  
  ```
  Predicted class: Car (99.5% confidence)
  ```

### Step 3️⃣: Exit 
After finishing with either training the model or classifying an Image or both, you have an option to **exit**.

- Select `'exit'`.  

  
---

## 8. Installation & Setup  

### 1️⃣ Install Dependencies  
```bash
pip install tensorflow keras numpy pillow
```

### 2️⃣ Set Up a Virtual Environment (Recommended)
To **avoid dependency issues**, create a **new virtual environment**:  
```bash
python -m venv myenv
source myenv/bin/activate  # Mac/Linux
myenv\Scriptsctivate  # Windows
```
Then, reinstall the dependencies:
```bash
pip install tensorflow keras numpy pillow
```

---

## Final Notes  
✅ The model **achieves 93%+ validation accuracy**.  
✅ 🚀 **For best performance, run this in a dedicated virtual environment (`venv`).**  

---

## 9. Future Improvements  
**Fine-tuning MobileNetV2 layers** to improve performance.  
**Experimenting with data augmentation** to improve generalization.  
**Using a larger dataset** for better accuracy.  
