# 🚀 Student Performance Index Predictor
 
## A GUI-based Machine Learning app built with Python & Streamlit that predicts student performance based on study hours and previous scores. This tool makes tracking student performance easy, interactive, and actionable 📊✨

## 🔹 Project Overview

This project allows educators, students, and data enthusiasts to:

Predict performance for individual students or batch predictions via CSV upload

Visualize how study hours and previous scores influence predicted performance

Load a trained scikit-learn model or fallback to a simple Linear Regression model

Use a clean, interactive GUI built with Streamlit — no coding required

This project strengthened skills in Python, Machine Learning, Streamlit, and data visualization, while also providing a practical tool for real-world use

🔹 Features

✅ Single & batch predictions
✅ Interactive charts & visualizations
✅ Model flexibility (trained model or default Linear Regression)
✅ User-friendly GUI
✅ Easy deployment and sharing

🔹 Installation & Run

Clone the repository

git clone [Your GitHub Repo Link]
cd Student-Performance-Predictor


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run student_performance_gui.py


Input individual student data or upload CSV to get predictions and visualizations

🔹 Project Structure
Student-Performance-Predictor/
│
├── student_performance_gui.py   # Main Streamlit app
├── model.pkl                    # Trained ML model (or default Linear Regression fallback)
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── data/                        # Optional: example CSV files for batch predictions
└── assets/                      # Optional: images or other assets

🔹 Learnings

Building interactive ML applications using Streamlit

Handling user input & batch predictions

Visualizing data with charts and dynamic graphs

Integrating pre-trained models with GUI for real-world usability

🔹 Future Improvements

Add more features (attendance, extra-curricular activities, etc.) for better prediction

Deploy as a web app for public access

Enhance visualizations with Plotly or Altair for richer interactivity

🔹 Technologies Used

Python 🐍

Streamlit ✨

Scikit-learn ⚙

Pandas & NumPy 📊

Matplotlib / Seaborn 🎨

🔹 Try it Yourself

Check out the code and explore the project here: [Your GitHub Link] 🔗


