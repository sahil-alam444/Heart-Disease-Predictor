# Overview
The Heart Disease Predictor project aims to predict the likelihood of heart disease in patients based on various health indicators. Using machine learning algorithms, this project helps in early diagnosis, enabling timely medical intervention.
# Features
Text Preprocessing: Tokenization, stopword removal, and vectorization using TF-IDF (Term Frequency-Inverse Document Frequency).

Classification Model: Logistic Regression is used for binary classification.

Performance Metrics: Evaluates model performance using accuracy, precision, recall, and F1 score.

# Dataset
The dataset contains several health-related attributes like age, cholesterol level, blood pressure, etc., and a target label indicating whether a patient has heart disease. The training dataset is split into training & test dataset.

# Workflow
1. Data Loading: Load the dataset of news articles with their labels.
   
2. Data Preprocessing:
   
   -Handle missing values.
   
   -Normalize features like cholesterol and blood pressure.
   
   -Encode categorical variables (e.g., gender).

3. Model Building:
   
   Implement Logistic Regression for classification.

4. Model Evaluation:
   
   Test the model using the test set and evaluate using performance metrics.

# Installation

1. CLone the repository
   ```bash
   git clone https://github.com/sahil-alam444/Heart-Disease-Predictor.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Predictor
   ```
   
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
# Usage

1. Train the model: Run the script to train the model on the provided dataset.
   ```bash
   python train_model.py
   ```

2. Evaluate the model: Evaluate the model's performance on the test data.
   ```bash
   python evaluate_model.py
   ```


