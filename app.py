import gradio as gr
import numpy as np
import pickle

# Load model
with open('/mnt/data/model_knn.pkl', 'rb') as f:
    model_knn = pickle.load(f)

with open('/mnt/data/model_nb.pkl', 'rb') as f:
    model_nb = pickle.load(f)

def predict_risk(
    model_type,
    gender,
    race,
    parental_edu,
    lunch,
    prep_course,
    math_score,
    reading_score,
    writing_score
):
    data = np.array([
        gender,
        race,
        parental_edu,
        lunch,
        prep_course,
        math_score,
        reading_score,
        writing_score
    ]).reshape(1, -1)

    if model_type == "KNN":
        pred = model_knn.predict(data)
    else:
        pred = model_nb.predict(data)

    return "Berisiko Nilai Rendah" if pred[0] == 1 else "Tidak Berisiko Nilai Rendah"

interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Radio(["KNN", "Naive Bayes"], label="Pilih Model"),
        gr.Number(label="Gender (Encoded)"),
        gr.Number(label="Race/Ethnicity (Encoded)"),
        gr.Number(label="Pendidikan Orang Tua (Encoded)"),
        gr.Number(label="Lunch (Encoded)"),
        gr.Number(label="Test Preparation Course (Encoded)"),
        gr.Number(label="Nilai Matematika"),
        gr.Number(label="Nilai Membaca"),
        gr.Number(label="Nilai Menulis"),
    ],
    outputs="text",
    title="Deteksi Dini Siswa Berisiko Mendapat Nilai Rendah",
    description="Aplikasi prediksi risiko nilai rendah siswa menggunakan model Machine Learning"
)

interface.launch()
