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


 # Advance AI/ML Tasks

# 🧠 Context-Aware Chatbot with RAG (Retrieval-Augmented Generation)

A simple RAG-based chatbot built using **LangChain**, **Streamlit**, and **Hugging Face Transformers**. This chatbot can answer questions from your own documents (PDF, text files) using local models — no OpenAI or API keys required!

---

## 🚀 Features

- 📄 Load knowledge from your own PDFs or `.txt` files
- 🤖 Answer questions using local Hugging Face models (e.g. `flan-t5-base`)
- 🧠 Remembers chat history using conversational memory
- 🔍 Uses FAISS for fast document search
- ✅ Runs locally — no internet or API needed!

---

## 🗂️ Project Structure
ContextAwareChatbot/
├── app/
│ ├── main.py # Streamlit app
│ ├── rag_pipeline.py # Loads and indexes PDF/text
│ └── memory.py # Manages chat history
├── data/
│ └── ml_tasks.pdf # Your knowledge source
├── .env
├── requirements.txt


#task_1  News Topic Classifier using BERT

Classify news articles into four categories — World, Sports, Business, and Science — using a fine-tuned BERT model.
📂 Dataset
AG News Dataset

Columns used: Title, Description, and Class Index

Source: Hugging Face Datasets

# Workflow
# Data Preprocessing

Combine Title and Description into a single text field

Clean text (optional)

Encode class labels (1–4 → 0–3)

# Model Pipeline

Tokenization using bert-base-uncased

Model: BERT with classification head (Dense(4, softmax))

Loss: SparseCategoricalCrossentropy

Optimizer: Adam

Evaluation: Accuracy, Confusion Matrix, Classification Report

# Training

Fine-tuned on training data for 3 epochs

Validated on 20% holdout set



# 🛠️ Tools & Libraries
Python, Pandas, NumPy

TensorFlow, TensorFlow Hub, Transformers

Matplotlib, Seaborn

scikit-learn

# 📁 Files
train.csv, test.csv: Input data

news_classifier.ipynb: Main code



# 📉 Customer Churn Prediction - Telco Dataset
This project builds a machine learning pipeline to predict customer churn using the Telco dataset. The goal is to help businesses identify customers likely to leave and take proactive retention measures.

# 🔍 Project Overview
Dataset: Telco Customer Churn

Objective: Predict whether a customer will churn

Tech Stack: Python, Pandas, Scikit-learn, Joblib

# 🛠️ Pipeline Steps
1-Data Preprocessing

Handling missing values

Encoding categorical features

Scaling numerical values

2-Feature Engineering

Creating informative features

Reducing noise and redundancy

3-Model Selection

Tried multiple classifiers

Used GridSearchCV for hyperparameter tuning

4-Model Evaluation

Evaluated using a classification report and a confusion matrix

5-Model Saving

Final model saved using joblib for deployment

# 📈 Results
The final model shows strong performance in identifying potential churners. This enables businesses to act ahead of time and improve customer retention.

