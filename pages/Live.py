import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport  
from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score


# --- Page Configuration ---
st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="🚀",
)

# --- Title and Description ---
st.title("Historical Market Data & Predictions")
st.markdown(""" Access real-time historical market data and detailed analytics for your selected stock.
Use the sidebar below to choose a ticker and date range, then click the **'Load Data'** button to see historical data.
""")

# --- Sidebar Inputs ---
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "META"]
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())

# --- Load Data Button ---
if st.sidebar.button("Load Data"):
    # --- Load Dataset ---
    # --- Define the Path to Your Data File ---
    BASE_DIR = Path(__file__).resolve().parent.parent  # Two levels up from 'pages/' to the root folder

    merged_data_path = BASE_DIR / "data" / "merged_data.csv"

    # --- Load the Dataset Safely ---
    if merged_data_path.exists():
        df = pd.read_csv(merged_data_path, parse_dates=["Date"])
        #st.write(df.head())  # Display the data to confirm it's loaded correctly
    else:
        st.error(f"File not found: {merged_data_path}")

    # --- Filter Dataset ---
    df_filtered = df[
        (df["Ticker"] == selected_ticker) & 
        (df["Date"] >= pd.to_datetime(start_date)) & 
        (df["Date"] <= pd.to_datetime(end_date))
    ]

    # --- Display Summary Metrics for the Selected Ticker ---
    st.subheader(f"Closing Price Metrics for {selected_ticker}")

    if df_filtered.empty:
        st.warning(f"No data available for {selected_ticker}.")
    else:
        # Calculate metrics for the 'Close' price from the filtered data
        low_price = df_filtered["Close"].min()
        high_price = df_filtered["Close"].max()
        mean_price = df_filtered["Close"].mean()

        # Calculate deltas relative to the mean
        delta_low = low_price - mean_price    # This will be negative
        delta_high = high_price - mean_price  # This will be positive

        # Create three columns for Low, Mean, and High
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Low", 
            value=f"${low_price:.2f}", 
            delta=f"${delta_low:.2f}",  # Remove manual arrow here
            delta_color="inverse",      # Streamlit will add the correct arrow automatically
            border=True
        )
        col2.metric(
            label="Mean", 
            value=f"${mean_price:.2f}", 
            border=True
        )
        col3.metric(
            label="High", 
            value=f"${high_price:.2f}", 
            delta=f"${delta_high:.2f}", # Remove manual arrow here
            delta_color="normal",       # Streamlit will add the correct arrow automatically
            border=True
        )

        # --- Stock Price Chart ---
        st.subheader("Stock Price Chart")
        fig = px.line(
            df_filtered, 
            x="Date", 
            y="Close", 
            title=f"{selected_ticker} Closing Prices Over Time", 
            labels={"Date": "Date", "Close": "Closing Price ($)"}
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Price ($)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# ... (keep all your previous imports and setup code)

BASE_DIR = Path(__file__).resolve().parent.parent
ml_predictions_path = BASE_DIR / "data" / "ml_predictions_2.csv"

if ml_predictions_path.exists():
    df_preds = pd.read_csv(ml_predictions_path, parse_dates=["Date"])
else:
    st.error(f"File not found: {ml_predictions_path}")
    st.stop()  # Stop execution if file not found

st.subheader("Predicted vs Actual Closing Price")
st.markdown("""
Here choose a ticker and you will see a graph comparing the actual closing prices versus the predicted ones from our
            proprietary Machine Learning model.
""")

# --- User Selection ---
selected_ticker = st.selectbox("Select a Ticker", sorted(df_preds["Ticker"].unique()))

# --- Filter ML Predictions ---
df_preds_filtered = df_preds[df_preds["Ticker"] == selected_ticker]

# Separate historical data (with actuals) and future prediction
historical_data = df_preds_filtered[df_preds_filtered['Close'].notna()]
future_prediction = df_preds_filtered[df_preds_filtered['Close'].isna()]

# Plot the chart
fig = px.line(
    historical_data,
    x="Date",
    y=["Close", "Predicted_Close"],
    title=f"{selected_ticker} Actual vs Predicted Closing Prices",
    labels={"Date": "Date", "value": "Closing Price ($)", "variable": "Legend"}
)

# Add future prediction if available
if not future_prediction.empty:
    fig.add_scatter(
        x=future_prediction["Date"],
        y=future_prediction["Predicted_Close"],
        mode='markers',
        name='Future Prediction',
        marker=dict(color='red', size=10)
    )

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Closing Price ($)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Show future prediction in a text box if available
if not future_prediction.empty:
    next_date = future_prediction["Date"].iloc[0].strftime('%Y-%m-%d')
    next_pred = future_prediction["Predicted_Close"].iloc[0]
    st.success(f"🚀 Next day prediction ({next_date}): ${next_pred:.2f}")

# Calculate performance metrics only for historical data
if not historical_data.empty:
    mae = mean_absolute_error(
        historical_data['Close'], 
        historical_data['Predicted_Close']
    )
    direction_accuracy = accuracy_score(
        (historical_data['Close'].diff() > 0).dropna(), 
        (historical_data['Predicted_Close'].diff() > 0).dropna()
    )
    correlation = historical_data['Close'].corr(historical_data['Predicted_Close'])
    
    # Create metrics columns
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="💰 Mean Absolute Error",
            value=f"${mae:.2f}",
            help="Average dollar difference between predicted and actual prices"
        )

    with metric_col2:
        st.metric(
            label="📈 Direction Accuracy",
            value=f"{direction_accuracy:.1%}",
            help="Percentage of correct up/down predictions"
        )

    with metric_col3:
        st.metric(
            label="🔗 Price Correlation",
            value=f"{correlation:.2f}",
            help="Strength of relationship between predictions and reality (1 = perfect)"
        )

    # Add performance insights
    st.markdown("""
    ### 📌 Model Performance Insights
    - **When MAE < $2.00**: The model predictions are very close to actual market prices  
    - **Direction Accuracy > 70%**: The model reliably predicts price movement direction  
    - **Correlation > 0.85**: Strong linear relationship between predictions and actuals  
    """)
else:
    st.warning("No historical data available for performance metrics.")

# Add disclaimer
st.caption("Note: Metrics calculated for the selected date range and ticker only")