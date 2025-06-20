import streamlit as st
import joblib
import re
import string

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # hapus url
    text = re.sub(r'\d+', '', text) # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # hapus tanda baca, string.punctuation berisi semua karakter randa baca
    text = re.sub(r'[^\x00-\x7F]+', '', text) # hapus karakter non-ASCII
    text = text.strip() # hapus spasi berlebih
    return text

# Load vectorizer dan model SVM
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')
svm_model = joblib.load('./model/svm_model.pkl')

# Judul aplikasi
st.title("Deteksi Ulasan Palsu")
st.write("Masukan ulasan untuk melakukan analisis menggunakan model SVM oleh Najib Sauqi R (21.11.4406).")

# Source Code
st.markdown("Web App Source Code: [Github](https://github.com/Njsa13/fake-review-detection-web-app)")

# Input dari pengguna
input_text = st.text_area("Input teks disini")

if st.button("Analisis"):
    if not input_text:
        st.warning("Masukan teks ulasan terlebih dahulu.")
    else:
        # Preprocessing dan vektorisasi
        processed_text = preprocess_text(input_text)
        vectorized_text = vectorizer.transform([processed_text])

        # Prediksi dengan model SVM
        prediction = svm_model.predict(vectorized_text)[0]
        
        # Output hasil
        result = 'Palsu' if prediction == 1 else 'Asli'
        st.write(f"**Teks Input:** {input_text}")
        st.write(f"**Hasil Prediksi:** {result}")
