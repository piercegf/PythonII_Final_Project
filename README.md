# 📈 Automated Trading System with Streamlit

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Team Members

- **Project Lead**: Alejandro Pérez
- **Data Engineer**: Ricardo Urech
- **ML Engineers**: Vibhushan Balaji and Manuel Bonnelly
- **Frontend Developer**: Dhabia Saad

## 🌟 Key Features

### Real-Time Trading Dashboard
- Interactive visualization of stock data from SimFin API
- Customizable date ranges and ticker selection
- Dynamic charting with Plotly Express

### Machine Learning Predictions
- **Binary Classification**: Predicts daily price direction (up/down)
- **Regression Model**: Forecasts next-day closing price
- Model performance metrics (accuracy, precision, recall)

### Backtesting Environment
- Portfolio simulation with historical data
- Customizable trading strategies
- Performance analytics and metrics

## 🛠️ Technology Stack

| Component               | Technologies Used |
|-------------------------|-------------------|
| Frontend                | Streamlit |
| Data Processing         | Pandas |
| Machine Learning        | Scikit-learn, XGBoost |
| Visualization           | Plotly, Matplotlib |
| Data API                | SimFin Python SDK |

## 🗂️ Project Structure

Python_II_Final_Project/
│
├── data/
│ ├── merged_data.csv
│ ├── ml_predictions_2.csv
│ ├── us-companies.csv
│ └── us-shareprices-daily.csv
│
├── pages/
│ ├── Explanation.py
│ ├── Live.py
│ └── Simulator.py
│
├── env/
├── .gitattributes
├── .gitignore
├── app.py
├── ETL.ipynb
├── ML.ipynb
└── README.md

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git LFS (for large files)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Python_II_Final_Project.git
cd Python_II_Final_Project

# Install dependencies
pip install -r requirements.txt