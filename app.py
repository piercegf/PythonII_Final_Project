import streamlit as st

# Set up page configuration with title and icon.
st.set_page_config(
    page_title="Automated Daily Trading System",
    page_icon="🏠",
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 2rem 1rem;
        background-color: #004B87;
        color: white;
        border-radius: 8px;
    }
    .section {
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section h3 {
        color: #004B87;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Header Section ---
st.markdown("<h1>Automated Daily Trading System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Python for Data Analysis II Group Assignment</h3>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Optional: Display a header image (replace the URL with one of your choice)
st.image(
    "https://www.pnc.com/content/dam/pnc-thought-leadership/corporate-institutional/pnc_tl_hub_ci_global-market-snapshot-march-2018.jpg", 
    caption="Market Prediction", 
    use_container_width=True
)

# --- Overview Section ---
st.markdown("### Overview", unsafe_allow_html=True)
st.markdown(
    """
    Welcome to the **Automated Daily Trading System**. This project is divided into two main parts:
    
    1. **Data Analytics Module:**  
      
      **ETL Process:**  
      We build an Extract, Transform, Load (ETL) pipeline using libraries like Pandas (or Polars) to process historical market data from SimFin. This step cleans and normalizes share price and company data—handling missing values and applying minimal transformations—to prepare a robust dataset for analysis and modeling.
      
      **Machine Learning Model:**  
      With the cleaned data, we develop a predictive model (starting with a simple Logistic Regression) to forecast next-day market movements (i.e., whether the price will rise or fall). The model leverages engineered features such as daily returns, moving averages, and volatility to provide actionable trading signals.
    
    2. **Web-Based Trading System:**  
      
      **Python API Wrapper for SimFin:**  
      An object-oriented API wrapper that:
      - Initializes with an API key and sets up the base endpoint.
      - Provides methods to fetch share prices and financial statements for specified companies and date ranges.
      - Incorporates logging and error handling for robust performance.
      
      **Web Application Development:**  
      Built with Streamlit, this interactive web app includes:
      - **app Page:** An overview of the system, detailing its functionalities, project objectives, and team information.
      - **Go Live Page:** A dynamic dashboard where users can select stock tickers, view historical and real-time market data, and see model-generated trading signals.
      - **Simulator Page:** A simulator where the Machine Learning model used is tested against a trading strategy and an inital balance to see potential gains/losses.
    """, unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# --- Data & SimFin Platform Section ---
st.markdown("### Data & SimFin Platform", unsafe_allow_html=True)
st.markdown(
    """
    Our data is sourced from [SimFin](https://www.simfin.com/), which provides:
    
    - **Share Prices:** Daily stock prices spanning 5 years.
    - **Companies Data:** Essential company details like ticker, name, and industry.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Machine Learning & Trading Strategy Section ---
st.markdown("### Machine Learning & Trading Strategy", unsafe_allow_html=True)
st.markdown(
    """
    **Machine Learning Module:**
    - Uses a simple Logistic Regression, XGBoost, and Random Forest model to predict if the market will rise or fall the next day.
    - Features include daily returns, moving averages, volatility, and more.
    - XGBRegressor was the model of choice in the Simulation and Live page. 
    
    """
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Development Team Section ---
st.markdown("### Development Team", unsafe_allow_html=True)

# List of team members with their names and image URLs (replace the URL with actual image links)
team_members = [
    {"name": "Vibhushan Balaji Neethi Mohan", "image": "https://media.licdn.com/dms/image/v2/D5603AQFzT0vqj0fbAw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1711963467534?e=1747267200&v=beta&t=Ko85KuxjasGxzRrZN6uVOcBm6VDweLqJL3MOUglT-jU"},
    {"name": "Manuel Eduardo Bonnelly Sanchez", "image": "https://media.licdn.com/dms/image/v2/D4D03AQFxkbRHcJCuoA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1681790835582?e=1748476800&v=beta&t=RPnrpFFLoIw-aaWAoOSIBm2EkoTlC5-hF9ykj_keLLI"},
    {"name": "Dhabia Saad Albuainain", "image": "https://media.licdn.com/dms/image/v2/D4D03AQFCFTO9vMumIA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1715493216201?e=1748476800&v=beta&t=mJlhjEG6VdtDvW9c-mTqmkznlJvHWhV4-TvfC32hd_0"},
    {"name": "Ricardo Urech García de la Vega", "image": "https://media.licdn.com/dms/image/v2/C4D03AQEJSLBYDbguoA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1619249710497?e=1747267200&v=beta&t=5FC5X84DmAOyk1-tO4B5rIvMJ1Y3NlDLgoHDx0PSzjQ"},
    {"name": "Alejandro Felipe Pérez Vargas", "image": "https://media.licdn.com/dms/image/v2/D4D03AQHQxhjlxAE0CA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1709819360560?e=1748476800&v=beta&t=45cWhgDyY_gLMXqRt9zHH0dnfPvrPc_wIh5vSigs568"}
]

# Create columns based on the number of team members
cols = st.columns(len(team_members))

# Display each team member's picture and name in their respective column
for col, member in zip(cols, team_members):
    with col:
        st.image(member["image"], width=150)
        st.markdown(f"**{member['name']}**")

# --- Footer Section ---
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("© 2025 Automated Daily Trading System | All Rights Reserved", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
