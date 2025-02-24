# 📩 Mail Detector - Spam or Not?

### 🚀 A Machine Learning Project by **RoboZe aka Prem**

## 📌 Overview
Ever wondered how email services detect spam? This project implements a **Mail Detector** using **Logistic Regression**, achieving an accuracy of **96.6%** on test data! The key challenge? **Extracting meaningful features from raw text data** and encoding labels effectively. 

## 📂 Dataset
We used an open-source **email dataset from Kaggle**, containing labeled mail texts categorized as **spam or ham (not spam)**. 

📌 **Key Features of the Dataset:**
- Text content of emails
- Labels indicating whether the email is **spam (1)** or **ham (0)**


## 🔥 Workflow

1️⃣ **Get Mail Data** – Collected labeled email data from Kaggle.
2️⃣ **Pre-processing** – Cleaned text data (removed stopwords, punctuation, tokenization, etc.).
3️⃣ **Split Data** – Divided into **training (80%) and testing (20%)**.
4️⃣ **Train Model** – Used **Logistic Regression** for classification.
5️⃣ **Testing & Evaluation** – Achieved **96.76% accuracy on training data** and **96.6% on test data**.

## 🛠️ Tech Stack
- **Python** 🐍
- **Pandas, NumPy** (Data Handling)
- **NLTK, re** (Text Preprocessing)
- **Scikit-learn** (ML Model & Evaluation)

## 🏆 Key Learning: **Extracting Features from Text Data**
Most ML models work on numerical data, but **emails are just text**. How do we make them usable for machine learning?

🔹 **Text Preprocessing Techniques:**
✅ Removing punctuation & special characters
✅ Converting to lowercase
✅ Tokenization & stopword removal
✅ Lemmatization (reducing words to base form)

🔹 **Feature Extraction Methods:**
✅ **TF-IDF (Term Frequency-Inverse Document Frequency)** – Helps in identifying important words.
✅ **Count Vectorization** – Converts text into a frequency matrix.
✅ **Word Embeddings** – More advanced method using NLP models.

🔹 **Label Encoding:**
We converted labels into **numerical values**: 
- **Spam → 1**
- **Ham → 0**

## 🎯 Results & Performance
📊 **Model Used:** Logistic Regression
- **Training Accuracy:** 96.76%
- **Test Accuracy:** 96.6%

## 🚀 How to Run This Project
1️⃣ Clone the repo:  
```bash
 git clone https://github.com/yourusername/mail-detector.git
```

2️⃣ Install dependencies:  
```bash
pip install pandas numpy sklearn nltk
```

3️⃣ Run the script:  
```bash
python mail_detector.py
```

## 📌 Future Improvements
🔹 Explore **Deep Learning Models (LSTMs, BERT)** for better accuracy.  
🔹 Deploy as a **real-time spam filter** using Flask or FastAPI.  
🔹 Add **real-world email data** for better generalization.

---

### 💡 Got Ideas or Suggestions?
Open an **Issue** or **Pull Request** to contribute! 🚀  
Let's connect on **[LinkedIn](https://linkedin.com/in/yourprofile)** if you love ML projects! 🤝

#MachineLearning #SpamDetection #TextProcessing #AI #Python
