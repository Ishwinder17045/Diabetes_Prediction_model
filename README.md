# 🩺 Diabetes Prediction Using Machine Learning

This project is a machine learning-based diabetes prediction system built using Python and Scikit-learn. It uses a dataset containing medical information to predict whether a person is likely to be diabetic.

# 📌 Project Overview
The goal of this project is to predict diabetes in patients based on certain diagnostic health parameters. It uses a Support Vector Machine (SVM) with a linear kernel to classify individuals as diabetic or non-diabetic.

# 💡 Features
Data Loading & Exploration
- Reads a CSV dataset (training_data.csv) containing patient health data.
- Provides summaries like shape, statistical description, and outcome distribution.

Data Preprocessing
- Separates the features and labels (Outcome column).
- Standardizes the feature data to improve model performance using StandardScaler.

Model Building
- Splits the dataset into training and testing sets (80% training, 20% testing).
- Uses a Support Vector Machine (SVM) classifier with a linear kernel.

Model Evaluation
- Calculates accuracy on both training and test datasets.
- Uses accuracy_score to measure prediction performance.

User Input for Live Prediction
- Accepts user input via the command line for key medical metrics:
  - Number of Pregnancies
  - Glucose Level
  - Blood Pressure
  - Skin Thickness
  - Insulin Level
  - BMI
  - Diabetes Pedigree Function
  - Age
- Standardizes this input using the same scaler.
- Predicts and displays whether the user is diabetic or not.

# ⚙️ How It Works

1. Load Data
- Reads a dataset with features related to diabetes.

2. Preprocess Data
- Standardizes the input features to make the SVM model more effective.

3. Train Model
- Trains an SVM classifier using labeled data.

4. Test Model
- Evaluates model accuracy on unseen data.

5. Interactive Prediction
- Takes real-time user input, applies the trained model, and outputs the diagnosis.

# 🛠 Technologies Used
- Python 3
- NumPy
- Pandas
- Scikit-learn (SVM, StandardScaler, train_test_split, accuracy_score)

# Accuracy

Training Accuracy: ~ (displayed during program run)

Test Accuracy: ~ (displayed during program run)









