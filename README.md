AAPL Stock Price Prediction & Deployment Dashboard 📈

A production-grade Deep Learning and Regression pipeline that analyzes historical market data for Apple Inc. (AAPL) and serves real-time stock price trend predictions through a fully synchronized, interactive Flask web interface.

🎯 Project Overview
This project processes historical financial data, handles feature engineering, trains a predictive deep learning model, and serializes the preprocessing pipelines for scalable inference. The final model is served via a lightweight Flask API web application with a responsive user interface dashboard, breaking away from traditional static Jupyter notebooks.

🏗️ System Architecture & Layout
The project repository is structured following professional software engineering and MLOps practices:

AAPL-Stock-Prediction/
│
├── data/                            # Core dataset layer
│   └── HistoricalQuotes.csv         # Raw Apple Inc. market data
│
├── models/                          # Serialized ML assets and weights
│   ├── aapl_price_prediction_model.h5
│   ├── feature_columns.joblib
│   ├── lookback_days.joblib
│   └── scaler_x.joblib
│
├── templates/                       # Frontend presentation layer
│   └── index.html                   # Interactive UI dashboard (Chart.js / FontAwesome)
│
├── .gitignore                       # Prevents tracking system dependencies & junk files
├── app.py                           # Core Flask application backend & inference API
├── train.py                         # Training pipeline script (renamed from mondel.py)
└── requirements.txt                 # Exact environment configuration metrics

🗺️ Completed Project Roadmap
[x] Phase 1: Project Setup — Configured Git environment, repository schema, and safety guidelines via .gitignore.

[x] Phase 2: Data Acquisition — Curated historical dataset parsing open, high, low, close, and volume technical figures.

[x] Phase 3: Exploratory Data Analysis (EDA) — Evaluated pricing cycles, trend vectors, and structural consistency.

[x] Phase 4: Feature Engineering — Structured historical sliding sequences and dimensional min-max scaling normalizations.

[x] Phase 5: Model Training — Trained a deep learning sequential network to target future price changes.

[x] Phase 6: Deployment — Engineered an operational Flask API backend synced with an interactive frontend template UI.

🛠️ Tech Stack & Dependencies
Core Language: Python

Deep Learning & Scaling: TensorFlow/Keras, Scikit-Learn

Data Processing: Pandas, NumPy, Joblib

Web Serving Architecture: Flask, Flask-CORS

Frontend Design: HTML5, CSS3, JavaScript (ES6+), Chart.js

⚡ How to Run and Initialize Locally
Clone the Repository:

Bash
git clone https://github.com/calvinnova1/AAPL-Stock-Prediction.git
cd AAPL-Stock-Prediction
Install Required Environment Pack:

Bash
pip install -r requirements.txt
Launch the Flask Application Server:

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/ to interact with the live analytics platform.

🛡️ Key Software Engineering Implementations
Defensive Asset Verification: The server utilizes pre-flight checking routines to scan for weights and pre-processors in the models/ directory prior to deployment execution, keeping the engine highly stable.

CORS Policy Protection: Integrated explicit Cross-Origin Resource Sharing protocols to ensure data transfers between the browser and backend work seamlessly.

Granular Error Vectors: Routes return targeted HTTP status codes and explicit structured JSON debug logs for bad payloads instead of general system failures.
