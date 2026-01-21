# Parkinson-s-Disease-Detection-using-XGBoost
A machine learning project that uses voice measurement data to classify Parkinson’s disease, including preprocessing, model training, evaluation, and visualizations.

# Parkinson’s Disease Detection using XGBoost

This project uses **machine learning** to detect Parkinson’s disease based on **voice measurement data**.  
An **XGBoost classifier** is trained to distinguish between healthy individuals and patients with Parkinson’s disease.

The project covers the full ML workflow:  
data loading, preprocessing, model training, evaluation, and visualization.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Dataset:** Parkinson’s Disease Dataset  
- **Link:** https://archive.ics.uci.edu/ml/datasets/parkinsons

Each row represents a person, and each feature represents a vocal measurement.  
The target variable is:
- `status = 0` → Healthy
- `status = 1` → Parkinson’s disease

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

---

## ⚙️ Project Workflow

1. Load and explore the dataset
2. Separate features and target labels
3. Split data into training and testing sets
4. Scale features using StandardScaler
5. Train an XGBoost classifier
6. Evaluate model performance
7. Visualize results (confusion matrix & feature importance)

---

## 📈 Model Performance

The model is evaluated using:
- Accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix

Visual results are saved as an image
