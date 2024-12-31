# 1. State the Problem (2%)

## 1.1 Problem and Pain Point
- Currently, Thailand is experiencing a banana shortage, leading to reduced banana production. Therefore, it is crucial to maximize the use of the limited banana supply. The “Banana Tester” was created to evaluate the ripeness level of bananas, as relying solely on human visual inspection of peel color and softness can be highly uncertain—subject to individual experience and environmental conditions. Moreover, fluctuating weather significantly affects the ripening rate, making it difficult to predict the ideal harvest time. Overripe or under-ripe bananas cannot be sold, resulting in economic losses, potential taste changes, and higher spoilage risks. Additionally, inconsistent ripeness levels among bananas increase labor and time spent sorting.

## 1.2 Related Work
- **Research References:** [Google Drive Link](https://drive.google.com/drive/folders/14BA3__-aD1RJfIRNJK6RZnHvRPOLwT9w?usp=drive_link)  
- **Strengths and Weaknesses of Previous Research:**  
  From the studies reviewed, there are diverse methods of data collection, and the approaches for classifying banana ripeness are highly detailed. However, such methods can be time-consuming for data gathering. In contrast, our current experimental project involves faster data collection but still achieves comparable results.

- **Identified Gap:**  
  Existing research lacks a user-friendly program or website capable of real-time analysis of banana ripeness. Addressing this gap is the key motivation for our system’s development.

## 1.3 Unique Method
- **Connecting Previous Research with Our Approach:**  
  Both the reviewed studies and our current project share a similar objective—solving the uncertainty in human-based banana ripeness classification. Such misclassification often leads to waste from prematurely rotten bananas. Therefore, the main aim of these studies and our project is to increase the accuracy of banana ripeness classification by leveraging AI. Ultimately, this extends to creating a real-time web-based tool for easy and immediate use.

---

# 2. Data Preparation (2%)

## 2.1 Data Source
- Bananas with four different ripeness levels (unripe, ripe, overripe, rotten) were obtained from:
  - [Kaggle Dataset](https://www.kaggle.com/datasets/atrithakar/banana-classification)
  - [Google Drive Link](https://drive.google.com/drive/folders/14Owh92DyWmgg9gTpC-8o-vFmWlJEux9a?usp=drive_link) (the drive link used for training)

## 2.2 Data Cleaning
- Images are categorized into folders according to their ripeness level.

## 2.3 Data Transformation
- All images used in this project are standardized to `.JPG` format. Each ripeness category is placed in its corresponding folder.

---

# 3. Training (1.5%)

## 3.1 AI Model
- **Supervised Learning: Classification**  
  We employ a classification approach to distinguish between the four banana ripeness levels.

## 3.2 Hyperparameters

### 3.2.1 Data Augmentation Hyperparameters
- `rotation_range=100`: Randomly rotate images within ±100 degrees  
- `zoom_range=0.05`: Randomly zoom images up to 5%  
- `width_shift_range=0.5`: Shift images horizontally by up to 50% of the width  
- `height_shift_range=0.5`: Shift images vertically by up to 50% of the height  
- `shear_range=0.15`: Apply shear transformations up to 15%  
- `horizontal_flip=True`: Enable horizontal flipping  
- `fill_mode="nearest"`: Fill in empty pixels using the nearest value

### 3.2.2 Data Splitting Hyperparameters
- `validation_split=0.2`: Reserve 20% of the data for validation; 80% is used for training  
- `train_size=0.7`: 70% of the dataset is used for training, and 30% for testing

### 3.2.3 Model Architecture Hyperparameters
- **MobileNetV2 Pre-trained Model**  
  - `input_shape=(224, 224, 3)`: Input images have a resolution of 224×224×3  
  - `include_top=False`: The final classification layers of MobileNetV2 are excluded  
- **Dense Layers**  
  - 1 dense layer with 64 neurons, `activation='relu'`  
  - 1 dense layer with 32 neurons, `activation='relu'`  
  - Final dense layer with 4 neurons (`activation='softmax'`) for the 4 classes  
- **Dropout Layers**  
  - Dropout rate of 0.5 (50%) to help mitigate overfitting

### 3.2.4 Model Compilation Hyperparameters
- `optimizer='adam'`: Use Adam optimizer to adjust model weights  
- `loss='binary_crossentropy'`: Use binary cross-entropy as the loss function

### 3.2.5 Model Training Hyperparameters
- `batch_size=32`: Number of examples processed in each weight update step  
- `epochs=13`: Total number of training epochs  
- **EarlyStopping**  
  - `monitor='val_loss'`: Monitor validation loss  
  - `patience=3`: Stop training if validation loss does not improve after 3 consecutive epochs  
  - `restore_best_weights=True`: Revert to the best weights obtained during training

### 3.2.6 ImageDataGenerator Settings
- `shuffle=False`: Disable shuffling of training samples (in certain generator setups)  
- `seed=0`: Fix the random seed for reproducible results

## 3.3 Model Training
- In this project, we use MobileNetV2 (pre-trained on ImageNet) and fine-tune it for banana ripeness classification.

---

# 4. Testing (1.5%)

## 4.1 Evaluation

![accuracy](https://i.pinimg.com/736x/b4/22/b9/b422b962b63584c10c1f0073f03cedff.jpg)

- **Accuracy: 97.29%**

## 4.2 Explanation of Results
- After testing, our model achieved an accuracy of 97.29%. This score represents the proportion of correctly predicted outcomes compared to the total number of predictions, indicating high model performance.

---

# 5. Model Improvement (5%)

## 5.1 Model Deficiencies
- Some instances were misclassified, with the model predicting the opposite class from reality. This issue appeared during the initial training phase. Also, version discrepancies (e.g., TensorFlow 2.12.0 vs. 2.17.0, and Keras 3.6.0) contributed to training complications.

## 5.2 Model Adjustments
- Increasing the image size (resolution) for uploads can improve clarity and potentially reduce misclassification.

## 5.3 Re-Evaluation After Improvements
- For the initial training, we used our own dataset, which introduced some uncertainty and lowered the model’s performance. We then switched to data from Kaggle, resulting in a significant improvement—up to 97.29% accuracy.

---

# 6. Others (Optional)

This document consolidates the essential information regarding the Banana Tester project. All references and data links are retained, and the text has been translated into English to provide broader accessibility.
