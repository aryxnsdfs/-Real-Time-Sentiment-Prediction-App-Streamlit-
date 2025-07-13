# Real-Time Sentiment Prediction App (Streamlit)

This app uses a trained ML model to classify reviews in real-time as **Positive** or **Negative**. It reads a local `TestRevs.csv` file every 5 seconds and updates predictions automatically.

---

## 🚀 Features
- Live updates every 5 seconds — no need to refresh!
- Displays the **last 5 sentiment predictions**
- Easy-to-use Streamlit dashboard
- Powered by a trained Scikit-learn model and TF-IDF vectorizer
- Just add reviews (no labels required) to `TestRevs.csv` and watch predictions appear

---

## 🧠 Model Info
- Binary sentiment classification: Positive (1) / Negative (0)
- Vectorizer: TF-IDF
- Classifier: Trained offline and saved using `joblib`

✅ requirements.txt Content

pip install -r requirements.txt
