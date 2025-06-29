# 🏡 House Price Prediction

A machine learning project built in **Google Colab** to predict house prices using features like location, condition, and garage availability.

## 📁 Files
- `House_Price_Prediction.ipynb`: Main Colab notebook
- `archive.zip`: Dataset (from Kaggle)

## 📌 Features Used
- Location (encoded)
- Condition (encoded)
- Garage (encoded)
- Other numerical features

## 🧠 Models
- Linear Regression
- Gradient Boosting Regressor

## 📊 Evaluation
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## 📈 Visuals
- Scatter plots: Actual vs Predicted prices
- Comparison between both models



# 🫀 Heart Disease Prediction

## 🔍 Objective
Predict the risk of heart disease using health data from the UCI Heart Disease Dataset.

## 📁 Dataset
- **Source**: UCI (https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)  
- **Target Variable**: `num` (0 = No disease, 1 = Disease)

## 🧹 Preprocessing
- Handled missing values using mean imputation  
- Converted target to binary  
- One-hot encoded categorical features

## 📊 EDA
- Visualized age distribution, correlation heatmap, and class distribution  
- Analyzed key features like sex, chest pain type, and fasting blood sugar

## 🧠 Model
- **Algorithm**: Logistic Regression  
- **Train/Test Split**: 80% / 20%  
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - ROC Curve & AUC

## 🔑 Key Features
- Chest pain type  
- Age  
- Maximum heart rate  
- ST depression (oldpeak)  
- Exercise-induced angina

## 🛠 Tools & Libraries
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn

The feature importance plot highlights the most influential variables in the prediction.

# 💊 Health Query ChatBot

This is a simple and interactive chatbot built with **Streamlit** and powered by **Groq's LLaMA 3.1 model**. It is designed to answer general **health-related questions**, provide information, and promote wellness awareness.

---

## 🧠 Features

- 🔬 Ask health-related queries (e.g., symptoms, wellness tips, diet, etc.)
- 🧾 Interactive chat interface powered by LLaMA 3.1 (via Groq API)
- 📜 Maintains chat history during the session
- 💡 Provides general health guidance (not medical advice)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Groq Python SDK
- dotenv

---

## 🚀 How to Run

### 1. Install Dependencies
    pip install -r requirements.txt
    pip install streamlit groq python-dotenv
### 2. Set Up .env
      GROQ_API_KEY=your_groq_api_key_here
### 3. Run the App
     streamlit run main.py
# 📌 Important Notes

This chatbot is for educational and informational purposes only.

It does not provide professional medical advice or emergency support.

For serious health concerns, always consult a licensed healthcare provider.

  
