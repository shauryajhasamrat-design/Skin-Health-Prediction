# 🩺 AI-Powered Skin Condition Detection System

## 📌 Overview

The **AI-Powered Skin Condition Detection System** is a deep learning-based image classification project that analyzes skin images and predicts the most likely skin-condition category.

The system uses **MobileNetV2 with Transfer Learning** and is designed to classify images into **16 different skin-condition categories**.

The project covers the complete machine learning workflow, including:

* Image preprocessing
* Dataset validation
* Stratified train/validation/test splitting
* Data augmentation
* Class-imbalance handling
* Transfer learning
* Model fine-tuning
* Performance evaluation
* Single-image prediction
* Top-3 prediction probabilities
* Rule-based next-step recommendations

> ⚠️ **Disclaimer:** This is an educational/research prototype and is **not a medical diagnostic system**. The predictions should not be used as a substitute for professional medical advice.

---

## 🎯 Project Objective

The goal of this project is to develop an image-based AI system capable of identifying the most likely skin-condition category from an uploaded image.

The project focuses on applying **Computer Vision and Deep Learning** techniques to a real-world classification problem while addressing challenges such as class imbalance, overfitting, and model generalization.

---

## 📊 Dataset

The project uses a dataset containing:

| Parameter         |         Value |
| ----------------- | ------------: |
| Total Images      |    **17,856** |
| Number of Classes |        **16** |
| Training Images   |    **12,498** |
| Validation Images |     **2,679** |
| Test Images       |     **2,679** |
| Input Image Size  | **224 × 224** |
| Batch Size        |        **32** |
| Corrupted Images  |         **0** |

The dataset was divided using a **stratified 70/15/15 split** to maintain class distribution across the training, validation, and test datasets.

---

## 🧠 Model

The project uses **MobileNetV2**, pretrained on ImageNet, as the feature-extraction backbone.

### Model Pipeline

```text
                    Input Image
                         │
                         ▼
                   Image Loading
                         │
                         ▼
                 Resize → 224×224
                         │
                         ▼
                 Data Augmentation
                         │
                         ▼
              MobileNetV2 Backbone
                (ImageNet Weights)
                         │
                         ▼
              Global Average Pooling
                         │
                         ▼
                   Dense Layer
                         │
                         ▼
              Batch Normalization
                         │
                         ▼
                    Dropout
                         │
                         ▼
                   Dense Layer
                         │
                         ▼
                    Dropout
                         │
                         ▼
                 Softmax Output
                         │
                         ▼
                 16 Skin Classes
```

---

## 🔄 Transfer Learning

Instead of training a CNN completely from scratch, the project uses **transfer learning**.

### Phase 1 — Feature Extraction

The pretrained MobileNetV2 layers were initially frozen while the newly added classification head was trained.

### Phase 2 — Fine-Tuning

After the classification head was trained, later layers of MobileNetV2 were unfrozen and fine-tuned using a significantly smaller learning rate.

This allows the pretrained model to adapt its learned visual features to the skin-condition classification task.

---

## 🖼️ Image Preprocessing

Each image passes through a preprocessing pipeline before being used by the model.

The pipeline includes:

1. Image loading
2. RGB conversion
3. Image resizing to **224 × 224**
4. Conversion to floating-point values
5. MobileNetV2 preprocessing

The dataset was also checked for corrupted or unreadable images.

**Result: 0 corrupted images detected.**

---

## 🔀 Data Augmentation

To improve generalization and reduce overfitting, the training pipeline applies augmentation techniques such as:

* Horizontal flipping
* Rotation
* Zoom
* Translation
* Contrast adjustment

The augmentation is applied dynamically during training rather than permanently creating additional image files.

---

## ⚖️ Class Imbalance

The dataset contains different numbers of samples for different classes.

To reduce the model's bias toward classes with more training examples, **class weights** were calculated using Scikit-learn.

```python
compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
```

The resulting weights were passed to the training process so that minority-class errors received greater importance during optimization.

---

## ⚙️ Training Configuration

| Parameter                 | Configuration                   |
| ------------------------- | ------------------------------- |
| Architecture              | MobileNetV2                     |
| Pretrained Weights        | ImageNet                        |
| Number of Classes         | 16                              |
| Image Size                | 224 × 224                       |
| Batch Size                | 32                              |
| Optimizer                 | Adam                            |
| Loss Function             | Sparse Categorical Crossentropy |
| Initial Learning Rate     | 0.001                           |
| Fine-Tuning Learning Rate | 0.00001                         |
| Initial Training          | Up to 12 epochs                 |
| Fine-Tuning               | Up to 10 epochs                 |
| Dropout                   | 0.40 + 0.30                     |

---

## 🛑 Regularization & Training Optimization

The project uses several techniques to improve training stability and reduce overfitting.

### Dropout

Dropout rates of **0.40 and 0.30** are used in the classification head.

### Early Stopping

Training monitors validation loss and stops when the validation performance stops improving for several epochs.

### ReduceLROnPlateau

The learning rate is reduced when validation loss stops improving, allowing the model to make smaller optimization steps during later stages of training.

---

## 📈 Model Evaluation

The model is evaluated on an **unseen test dataset**.

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Why multiple metrics?

Accuracy provides an overall measure of model performance, but it may not fully represent performance when classes are imbalanced.

Precision, recall, and F1-score provide additional insight into how well the model performs for individual classes.

The confusion matrix is used to identify classes that the model frequently confuses with one another.

---

## 🔍 Single Image Prediction

The project includes a prediction pipeline that allows a new image to be passed through the trained model.

The system returns:

```text
Predicted Condition
        +
Top-3 Predictions
        +
Prediction Probabilities
        +
Predefined Next-Step Guidance
```

Example:

```text
Top Predictions

1. Eczema       61.2%
2. Psoriasis    18.4%
3. Rashes        9.7%
```

The **next-step guidance is rule-based** and separate from the CNN prediction model.

---

## 💡 Key Features

### 🔹 Multi-Class Classification

Predicts among **16 skin-condition categories**.

### 🔹 Transfer Learning

Uses pretrained MobileNetV2 visual features.

### 🔹 Class-Imbalance Handling

Uses class weights during training.

### 🔹 Data Augmentation

Introduces image variations to improve generalization.

### 🔹 Fine-Tuning

Adapts later MobileNetV2 layers to the target dataset.

### 🔹 Model Evaluation

Uses multiple classification metrics and a confusion matrix.

### 🔹 Top-3 Predictions

Displays the three most probable classes instead of only the highest-probability class.

### 🔹 Educational Guidance

Provides a predefined next-step message based on the predicted class.

---

## 📁 Project Structure

```text
AI-Powered-Skin-Condition-Detection-System/
│
├── dataset/
│   └── skin-condition-images/
│
├── notebooks/
│   └── skin_condition_classification.ipynb
│
├── models/
│   └── trained_model.keras
│
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

**Update the filenames/folders above to match your actual GitHub repository.**

---



## 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
numpy
pandas
scikit-learn
opencv-python
Pillow
matplotlib
seaborn
jupyter
```

---

## 📊 Results

| Metric | Score |
|---|---:|
| Test Accuracy | 78.84% |
| Macro Precision | 79.20% |
| Macro Recall | 78.52% |
| Macro F1-Score | 78.41% |

Test Set: 2,679 images  
Number of Classes: 16

## 🧩 Challenges Addressed

This project addresses several common machine-learning challenges:

| Challenge              | Approach                                |
| ---------------------- | --------------------------------------- |
| Class imbalance        | Class weighting                         |
| Overfitting            | Dropout + augmentation + early stopping |
| Limited training data  | Transfer learning                       |
| Image variability      | Data augmentation                       |
| Model adaptation       | Fine-tuning                             |
| Class-specific errors  | Confusion matrix                        |
| Prediction uncertainty | Top-3 predictions                       |

---

## 🔮 Future Improvements

Potential improvements include:

* Testing additional CNN architectures
* Hyperparameter optimization
* Increasing dataset diversity
* Testing on an independent external dataset
* Adding image-quality checks
* Improving performance on difficult classes
* Building a Streamlit web application
* Creating a Flask/FastAPI prediction API
* Deploying the model to the cloud
* Adding confidence-based prediction handling

---

## ⚠️ Limitations

This system should be considered an **educational AI prototype** rather than a clinical diagnostic system.

Limitations include:

* Similar skin conditions may have visually similar features.
* Image quality and lighting can affect predictions.
* The dataset may not represent all real-world populations.
* Performance on the test dataset does not guarantee real-world generalization.
* External validation would be required before considering practical clinical use.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience with:

* Python
* TensorFlow & Keras
* Computer Vision
* CNNs
* Transfer Learning
* MobileNetV2
* Data Augmentation
* Class Imbalance
* Model Fine-Tuning
* Regularization
* Model Evaluation
* Confusion Matrix Analysis
* Top-K Predictions
* Deep Learning Pipelines

---

## 👨‍💻 Author

**Samrat Shaurya Jha**

### Areas of Interest

* Machine Learning
* Deep Learning
* Computer Vision
* Data Science
* AI Applications

---

## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.

The predictions generated by this system should **not** be considered medical diagnoses or professional medical advice. Users with health concerns should consult a qualified healthcare professional.

