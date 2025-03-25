import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Stock Prediction Pipeline",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid=stSidebar] {
    }
    /* Disable click highlight */
    [data-testid=stSidebar] *:focus:not(:active) {
        outline: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem 10rem;
    }
    h2 {
        border-bottom: 2px solid #4a6bff;
        padding-bottom: 0.3rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .etl-step {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Stock Prediction Pipeline")
    st.markdown("""
    **Companies Analyzed:**  
    🍏 Apple (AAPL) | 🪟 Microsoft (MSFT) | 📦 Amazon (AMZN)  
    🚗 Tesla (TSLA) | 👍 Meta (META)
    """)
    
    st.divider()
    
    st.markdown("""
    ### Model Performance
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Logistic Regression", "50.9%", "Accuracy")
        st.metric("Random Forest", "52.1%", "Accuracy")
    with col2:
        st.metric("XGBoost Classifier", "50.0%", "Accuracy")
    
    style_metric_cards(border_left_color="#4a6bff")

# Main Content
st.title("Stock Price Prediction Pipeline")
st.markdown("""
This application demonstrates our complete workflow from data extraction to machine learning predictions 
for five major tech stocks.
""")

# Enhanced ETL Section
with st.expander("🔧 **ETL Process (Extract-Transform-Load)**", expanded=True):
    st.markdown("""
    ### Data Extraction
    We gathered financial data from SimFin's API with these key steps:
    """)
    
    with st.container():
        st.markdown("""
        <div class="etl-step">
            <h4>🔑 Authentication & Setup</h4>
            <ul>
                <li>Loaded API keys and paths from <code>.env</code> file using <code>python-dotenv</code></li>
                <li>Configured SimFin API with proper authentication</li>
                <li>Set up local cache directory for efficient data retrieval</li>
            </ul>
        </div>
        
        <div class="etl-step">
            <h4>📥 Primary Datasets Extracted</h4>
            <ul>
                <li><strong>Company Information</strong>: Metadata including tickers, names, sectors (filtered to US market)</li>
                <li><strong>Daily Share Prices</strong>: OHLCV data (Open, High, Low, Close, Volume) with historical records</li>
            </ul>
        </div>
        
        <div class="etl-step">
            <h4>🔄 Alternative Data Source</h4>
            <ul>
                <li>Fallback to pre-downloaded CSV files when API unavailable</li>
                <li>Local cache system for offline development</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Data Transformation
    The raw data underwent rigorous processing:
    """)
    
    with st.container():
        st.markdown("""
        <div class="etl-step">
            <h4>🎯 Company Filtering</h4>
            <ul>
                <li>Selected focus companies: AAPL, MSFT, AMZN, TSLA, META</li>
                <li>Applied ticker-based filtering to both datasets</li>
            </ul>
        </div>
        
        <div class="etl-step">
            <h4>🧹 Data Cleaning</h4>
            <ul>
                <li>Removed irrelevant columns (e.g., 'Dividend' from prices data)</li>
                <li>Converted 'Date' to datetime format for time-series analysis</li>
                <li>Verified no missing values in critical columns</li>
            </ul>
        </div>
        
        <div class="etl-step">
            <h4>🤝 Dataset Merging</h4>
            <ul>
                <li>Joined company metadata with price data on 'Ticker' column</li>
                <li>Used left join to preserve all price records</li>
                <li>Ensured consistent date ranges across all companies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Data Loading
    Final output preparation:
    """)
    
    with st.container():
        st.markdown("""
        <div class="etl-step">
            <ul>
                <li>Saved merged dataset as CSV for future use</li>
                <li>Structured format with columns: <code>[Date, Ticker, Open, High, Low, Close, Volume, CompanyName, Sector, ...]</code></li>
                <li>Optimized data types for efficient storage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization for ETL
    st.markdown("### ETL Process Overview")
    etl_steps = pd.DataFrame({
        'Phase': ['Extract', 'Transform', 'Load'],
        'Key Actions': [
            "API authentication, data download, local caching",
            "Filtering, cleaning, type conversion, merging",
            "CSV export, data validation"
        ],
        'Tools': [
            "simfin API, pandas, dotenv",
            "pandas, numpy",
            "pandas, os"
        ]
    })
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(etl_steps.columns),
            fill_color='#4a6bff',
            font_color='white',
            align='left'
        ),
        cells=dict(
            values=[etl_steps.Phase, etl_steps['Key Actions'], etl_steps.Tools],
            fill_color='#f8f9fa',
            align='left'
        ))
    ])
    st.plotly_chart(fig, use_container_width=True)

# ML Section (unchanged)
with st.expander("🤖 **Machine Learning Pipeline**", expanded=True):
    st.markdown("""
    ### Why XGBoost Regressor?
    We chose **XGBoost Regressor** because it:
    - Predicts the actual next day's closing price (continuous value)
    - Allows conversion to binary classification (up/down)
    - Handles tabular financial data exceptionally well
    - Provides feature importance metrics
    """)
    
    st.markdown("""
    ### Feature Engineering
    We created these predictive features:
    """)
    
    # Feature cards
    features = [
        ("📊 Daily Return", "Percentage change between closing prices"),
        ("📈 Moving Averages", "5/10/20-day rolling windows"),
        ("⚡ Volatility", "5-day standard deviation of closing prices"),
        ("📊 Volume Ratio", "Current volume vs 5-day average"),
        ("🎯 Price Range", "(High - Low)/Close price normalization")
    ]
    
    for name, desc in features:
        with st.container():
            st.markdown(f"""
            <div class="feature-card">
                <strong>{name}</strong>: {desc}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Target Variable
    ```python
    # Binary classification target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    ```
    """)