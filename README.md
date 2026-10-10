# 📉 Customer Churn Prevention Model

A machine learning project that predicts customer churn using customer demographics, account information, and service-related features. The goal is to help businesses identify customers who may be likely to discontinue their services.

## 🌐 Live Demo

[**Try the Customer Churn Prevention Model**](https://customer-churn-prediction-system-bhoomi.streamlit.app/)

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses machine learning to predict whether a customer is likely to leave a service, helping businesses identify customers who may need retention efforts.

The project uses the Telco Customer Churn dataset containing 7,043 customer records and 21 columns.

## ✨ Features

- Predicts whether a customer is likely to churn.
- Processes customer and service-related information.
- Handles missing or improperly formatted values.
- Uses a Random Forest classification model.
- Evaluates model performance using classification metrics.
- Supports data-driven customer retention decisions.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model:** Random Forest Classifier
- **Visualization:** Matplotlib and Seaborn, if used in the project
- **Web Application:** Streamlit, if included in the final application

## 📊 Model Performance

The model achieved the following results in the reported evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 79.56% |
| F1-score | 0.55 |
| ROC-AUC | 0.837 |

These results reflect the recorded evaluation and may vary with different data splits or model configurations.

## ⚙️ Project Workflow

1. **Data Collection:** Load the Telco Customer Churn dataset.
2. **Data Preprocessing:** Clean the data and convert `TotalCharges` into a numeric format.
3. **Feature Preparation:** Remove the `customerID` column and encode categorical features as required.
4. **Model Training:** Train a Random Forest classification model.
5. **Model Evaluation:** Evaluate the model using accuracy, F1-score, and ROC-AUC.
6. **Prediction:** Use the trained model to predict potential customer churn.

## 💼 Business Applications

- Identify customers at risk of leaving.
- Help prioritize customer retention campaigns.
- Support targeted offers and customer engagement.
- Assist businesses in making data-driven decisions.

## ▶️ Run Locally

**1. Clone your repository**

```bash
git clone https://github.com/Bhoomi-kumari08/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System
```

**2. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Run the application**

If your project includes a Streamlit interface:

```bash
streamlit run app.py
```

If it contains only a training script or notebook, run the appropriate Python file or open the notebook instead.

## 🎯 Learning Outcomes

- Understanding binary classification.
- Data cleaning and feature preprocessing.
- Training and evaluating a Random Forest classifier.
- Interpreting classification metrics.
- Applying machine learning to a real-world business problem.

## 👩‍💻 Author

**Bhoomi Kumari**

B.Tech — Computer Science and Engineering