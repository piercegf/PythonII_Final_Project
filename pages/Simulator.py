import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


st.set_page_config(page_title="Simplified Trading Simulator", page_icon="💹")

st.title("Trading Simulator")

st.markdown("""
### Trading Strategy Rules

1. **Entry Condition**  
   - Compute the predicted percentage change from today's closing price:  
     `Δ_pred = (P_pred - P_today) / P_today`  
     If `Δ_pred > 0.02` (i.e., greater than 2%), **buy** using all available capital at today's close.

2. **Exit Conditions**  
   - Once in a position, calculate the current return:  
     `Δ_current = (P_current - P_entry) / P_entry`  
     - **Take Profit:** If `Δ_current ≥ 0.05` (i.e., 5% gain), **sell**  
     - **Stop Loss:** If `Δ_current ≤ -0.03` (i.e., 3% loss), **sell**  

At the end of the simulation period, if still holding a position, we sell on the final day.
            
Example: 
- The strategy starts with a certain initial capital (e.g., $10,000).
- Whenever the strategy buys, all capital is converted into shares at that day’s closing price.
- Whenever the strategy sells, those shares are converted back to cash at the selling price.
- This results in a single position at most (all-in or all-out).
- The simulator logs each trade (buy or sell) and tracks the portfolio value day by day.
""")

# Define the absolute path dynamically
ml_predictions_path = Path(__file__).resolve().parent.parent / "data" / "ml_predictions_2.csv"

if ml_predictions_path.exists():
    df_preds = pd.read_csv(ml_predictions_path, parse_dates=["Date"])
    #st.write(df_preds.head())
else:
    st.error(f"File not found: {ml_predictions_path}")

# --- Select Ticker for Simulation ---
selected_ticker = st.selectbox("Select a Ticker for Simulation", sorted(df_preds["Ticker"].unique()))

# Filter predictions for the selected ticker and sort by Date
df_sim = df_preds[df_preds["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)

if df_sim.empty:
    st.error("No predictions available for the selected ticker.")
else:
    # --- Add Simulation Date Range Selection ---
    min_date = df_sim["Date"].min()
    max_date = df_sim["Date"].max()
    sim_date_range = st.date_input(
        "Select Simulation Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(sim_date_range, (list, tuple)) and len(sim_date_range) == 2:
        sim_start_date, sim_end_date = sim_date_range
    else:
        sim_start_date = min_date
        sim_end_date = max_date

    # Filter simulation data by selected date range
    df_sim = df_sim[
        (df_sim["Date"] >= pd.to_datetime(sim_start_date)) & 
        (df_sim["Date"] <= pd.to_datetime(sim_end_date))
    ]
    
    if df_sim.empty:
        st.error("No simulation data available in the selected date range.")
    else:
        # --- Input Initial Capital ---
        initial_capital = st.number_input(
            "Enter Initial Capital ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )
    
        # --- Parameters for Trading Strategy ---
        buy_threshold = 0.02   # 2% predicted gain to enter
        take_profit = 0.05     # 5% gain to exit
        stop_loss = -0.03      # 3% loss to exit
    
        if st.button("Simulate Trading"):
            capital = initial_capital
            position = None  # Will hold {'entry_price': float, 'shares': float}
            portfolio_values = []  # Track daily portfolio value
            dates = []
            trade_log = []   # Log of trades (Date, Action, Price, Shares)
            
            n = len(df_sim)
            for i in range(n - 1):
                current_date = df_sim.iloc[i]["Date"]
                current_price = df_sim.iloc[i]["Close"]
                predicted_price = df_sim.iloc[i]["Predicted_Close"]
                
                # Predicted percentage change
                predicted_pct = (predicted_price - current_price) / current_price
                
                # Record current portfolio value
                if position is None:
                    current_portfolio_value = capital
                else:
                    current_portfolio_value = position["shares"] * current_price
                portfolio_values.append(current_portfolio_value)
                dates.append(current_date)
                
                # --- Trading Logic ---
                if position is None:
                    # Entry: If predicted change > 2%
                    if predicted_pct > buy_threshold:
                        shares = capital / current_price
                        position = {"entry_price": current_price, "shares": shares}
                        trade_log.append((current_date, "BUY", current_price, shares))
                        capital = 0
                else:
                    # Already in position: compute current return
                    entry_price = position["entry_price"]
                    current_return = (current_price - entry_price) / entry_price
                    
                    # Exit conditions
                    if current_return >= take_profit or current_return <= stop_loss:
                        capital = position["shares"] * current_price
                        trade_log.append((current_date, "SELL", current_price, position["shares"]))
                        position = None
            
            # End of simulation: if still holding, liquidate on the last day
            final_date = df_sim.iloc[-1]["Date"]
            final_price = df_sim.iloc[-1]["Close"]
            if position is not None:
                capital = position["shares"] * final_price
                trade_log.append((final_date, "SELL (End)", final_price, position["shares"]))
                position = None
            
            portfolio_values.append(capital)
            dates.append(final_date)
            
            # Display results
            st.success(f"Final Portfolio Value: ${capital:,.2f}")
            
            if trade_log:
                st.subheader("Trade Log")
                trade_log_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Shares"])
                st.dataframe(trade_log_df)
            
            # Plot portfolio value over time
            df_portfolio = pd.DataFrame({"Date": dates, "Portfolio Value": portfolio_values})
            fig = px.line(
                df_portfolio, 
                x="Date", 
                y="Portfolio Value", 
                title="Portfolio Value Over Time (Simplified Strategy)"
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Calculate Portfolio Metrics ---
            total_return = ((capital - initial_capital) / initial_capital) * 100
            num_trades = len(trade_log)

            # Calculate maximum drawdown
            running_max = 0
            max_drawdown = 0
            for val in portfolio_values:
                running_max = max(running_max, val)
                drawdown = (val - running_max) / running_max
                max_drawdown = min(max_drawdown, drawdown)
            max_drawdown_pct = abs(max_drawdown) * 100

            # --- Enhanced Metrics Display ---
            st.subheader("📊 Portfolio Performance Snapshot")

            # Create metric cards with emojis and color coding
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="💰 Initial Capital",
                    value=f"${initial_capital:,.2f}",
                    help="Starting investment amount"
                )

            with col2:
                delta_color = "inverse" if total_return < 0 else "normal"
                st.metric(
                    label="📈 Final Value",
                    value=f"${capital:,.2f}",
                    delta=f"{total_return:+.2f}%",
                    delta_color=delta_color,
                    help="Portfolio value including all gains/losses"
                )

            with col3:
                st.metric(
                    label="🔄 Trades Executed",
                    value=num_trades,
                    help="Total buy/sell orders during period"
                )

            with col4:
                danger_color = "normal" if max_drawdown_pct < 20 else "off"
                st.metric(
                    label="⚠️ Max Drawdown",
                    value=f"{max_drawdown_pct:.2f}%",
                    help="Largest peak-to-trough decline during period",
                    delta_color=danger_color
                )

            # Add performance summary with emoji indicators
            performance_emoji = "🚀" if total_return > 0 else "💸"
            st.markdown(f"""
            ### {performance_emoji} Performance Summary
            - **{"✅ Profitable" if total_return > 0 else "❌ Losing"} Strategy**: {abs(total_return):.2f}% {"gain" if total_return > 0 else "loss"}
            - **{"🛡️ Conservative" if max_drawdown_pct < 15 else "🎢 Risky"} Drawdown**: {max_drawdown_pct:.2f}% at lowest point
            - **🏦 Trade Frequency**: {num_trades} trades ({num_trades/len(portfolio_values):.1f} trades/day)
            """)

