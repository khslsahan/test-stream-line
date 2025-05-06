import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from streamlit.components.v1 import html
import psycopg2
import os
import urllib.parse as urlparse
import plotly.express as px
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv(dotenv_path=r"E:\Side Projects\CSE bot\myenv\streamlit\myenv\Scripts\.env")
load_dotenv()

# --- Database Connection ---
@st.cache_resource
def init_connection():
    # This line will now directly get the secret from Streamlit Cloud's environment
    db_url = os.environ.get("NEON_DB_URL")
    if not db_url:
        # This error will trigger if the secret is not set in Streamlit Cloud
        st.error("Database URL not found in environment variables! Please configure Streamlit Secrets.")
        return None
    url = urlparse.urlparse(db_url)
    return psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or "5432",
        sslmode="require",
    )

# --- Load Data ---
# Use `st.cache_data` for the dataframe result.
# It depends on the connection obtained from init_connection.
@st.cache_data(ttl=600)
def load_data():
    status_message = st.empty()

    status_message.text("Attempting to load data...") # Write initial status
    
    # Get a connection from the cache. It will be created only once by init_connection
    # until the cache is cleared or arguments change.
    conn = init_connection()

    if conn is None:
        status_message.error("Cannot load data: Database connection failed.")
        return pd.DataFrame() # Return empty DataFrame if connection wasn't established

    # --- Add a try-except block to handle stale connections ---
    try:
        # Use the connection within a 'with' block for safe cursor handling
        # Check if the connection is closed before using (optional but can help catch early)
        # Note: Checking conn.closed might not always catch a network-level closure immediately
        if conn.closed != 0:
             status_message.warning("Cached connection found but it was closed. Attempting to re-establish...")
             init_connection.clear() # Clear the broken cached connection
             conn = init_connection() # Try to get a new connection immediately
             if conn is None:
                 status_message.error("Failed to re-establish database connection.")
                 return pd.DataFrame() # Return empty if reconnect failed


        with conn.cursor() as cur:
            status_message.text("Executing SQL query...") # Update status
            cur.execute("SELECT * FROM stock_analysis_all_results;")
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
        status_message.success("Data loaded successfully.") # Debugging
        # DO NOT CLOSE THE CONNECTION HERE! @st.cache_resource manages its lifecycle.
        # conn.close() # <--- REMOVE THIS LINE! (You already did this, good!)

        df = pd.DataFrame(rows, columns=colnames)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Ensure 'date' is timezone-naive if it's not already, for consistent comparisons later
        if 'date' in df.columns and pd.api.types.is_datetime64tz_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Ensure 'date' is timezone-naive if it's not already, for consistent comparisons later
            # Use isinstance(dtype, pd.DatetimeTZDtype) as recommended by the warning
            if isinstance(df['date'].dtype, pd.DatetimeTZDtype): # <--- Corrected line
                df['date'] = df['date'].dt.tz_convert(None)
            # Drop rows where date conversion failed
            df.dropna(subset=['date'], inplace=True)


        return df

    except psycopg2.OperationalError as e:
        # This specific error often indicates connection problems (like being closed)
        status_message.error(f"Database Operational Error: {e}")
        st.info("Attempting to clear cached connection and reload.")
        # Clear the cached connection resource so init_connection will run again
        init_connection.clear()
        # Returning an empty DataFrame. Streamlit will rerun on user interaction,
        # or you could use st.rerun() but clearing cache and letting user interact
        # is often sufficient and less disruptive.
        return pd.DataFrame()

    except Exception as e:
        # Catch any other unexpected errors during data loading
        st.error(f"An unexpected error occurred while loading data: {e}")
        return pd.DataFrame()

from tvDatafeed import TvDatafeed, Interval
from streamlit.components.v1 import html

# Initialize TradingView datafeed
tv = TvDatafeed()

def calculate_performance(tier_2_picks):
    """Calculate Maverick's Picks performance based on capital gain."""
    performance_data = []

    for symbol in tier_2_picks['symbol'].unique():
        try:
            # Fetch historical data for the stock symbol
            cse_data = tv.get_hist(symbol=symbol, exchange='CSELK', interval=Interval.in_daily, n_bars=200)

            if cse_data is not None and not cse_data.empty:
                # Get the initial close price (from the earliest date in the filtered data)
                oldest_date = tier_2_picks[tier_2_picks['symbol'] == symbol]['date'].min()
                initial_close = tier_2_picks[
                    (tier_2_picks['symbol'] == symbol) & (tier_2_picks['date'] == oldest_date)
                ]['closing_price'].iloc[0]

                # Get the latest close price from TradingView data
                latest_close = cse_data['close'].iloc[-1]

                # Convert to float if necessary
                initial_close = float(initial_close)
                latest_close = float(latest_close)

                # Calculate capital gain
                capital_gain = ((latest_close - initial_close) / initial_close) * 100

                # Append the result to the performance data
                performance_data.append({
                    'Date Detected': oldest_date.date(),
                    'symbol': symbol,
                    'Detected Day Close': initial_close,
                    'Latest Close': latest_close,
                    'Capital Gain Til Date(%)': capital_gain
                })
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")

    # Convert performance data to a DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Sort the DataFrame by Capital Gain in descending order
    performance_df = performance_df.sort_values(by='Capital Gain Til Date(%)', ascending=False)
    
    # Calculate the number of counters and overall PNL
    num_counters = len(performance_df)
    
    overall_pnl_percent_sum = performance_df['Capital Gain Til Date(%)'].sum()
    overall_pnl_if_100_each = overall_pnl_percent_sum # 
    
    # Calculate Hit Rate (Positive Gain / Total Counters)
    positive_gains_count = (performance_df['Capital Gain Til Date(%)'] > 0).sum()
    hit_rate = (positive_gains_count / num_counters * 100) if num_counters > 0 else 0

    # Calculate Performance Buckets
    gains = performance_df['Capital Gain Til Date(%)']

    count_below_minus_10 = (gains < -10).sum()
    count_minus_10_to_minus_5 = ((gains >= -10) & (gains < -5)).sum()
    count_minus_5_to_plus_5 = ((gains >= -5) & (gains < 5)).sum()
    count_plus_5_to_plus_10 = ((gains >= 5) & (gains < 10)).sum()
    count_above_plus_10 = (gains >= 10).sum()

    # Calculate the percentage of stocks in different capital gain ranges
    below_neg_10 = len(performance_df[performance_df['Capital Gain Til Date(%)'] < -10]) / num_counters * 100 if num_counters > 0 else 0
    minus_5_to_0 = len(performance_df[(performance_df['Capital Gain Til Date(%)'] >= -5) & (performance_df['Capital Gain Til Date(%)'] < 0)]) / num_counters * 100 if num_counters > 0 else 0
    zero_to_5 = len(performance_df[(performance_df['Capital Gain Til Date(%)'] >= 0) & (performance_df['Capital Gain Til Date(%)'] < 5)]) / num_counters * 100 if num_counters > 0 else 0
    above_10 = len(performance_df[performance_df['Capital Gain Til Date(%)'] >= 10]) / num_counters * 100 if num_counters > 0 else 0
    above_5 = len(performance_df[performance_df['Capital Gain Til Date(%)'] >= 5]) / num_counters * 100 if num_counters > 0 else 0

    
    # Highlight positive and negative gains
    def highlight_gain(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    # Display the performance table
    if not performance_df.empty:
        st.markdown("### 📊 Maverick's Picks Performance")
        st.dataframe(
            performance_df.style.format({
                'Detected Day Close': '{:,.2f}',
                'Latest Close': '{:,.2f}',
                'Capital Gain Til Date(%)': '{:,.2f}'
            }).applymap(highlight_gain, subset=['Capital Gain Til Date(%)']),
            use_container_width=True
        )
    else:
        st.info("No performance data available.")
        
    # Display Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    st.markdown(f"**Detected Counters in the selected time period:** {num_counters}")
    st.markdown(f"<span style='font-size:18px; color:green;'>**Hit Rate (Positive Gains):** {hit_rate:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Capital Gain % Sum (Sum of individual stock % gains):** {overall_pnl_percent_sum:.2f}%")
    st.markdown(f"*(Interpretation: If you invested 1% of your portfolio in each counter, the total portfolio gain would be ~{overall_pnl_percent_sum:.2f}%)*")
    st.markdown(f"*(Note: This is not a portfolio return calculation, just a sum of individual percentage gains)*")

    st.markdown("")
    st.markdown(f"**% of Stocks Below -10%:** {below_neg_10:.2f}%")
    st.markdown(f"**% of Stocks Between -5% and 0%:** {minus_5_to_0:.2f}%")
    st.markdown(f"**% of Stocks Between 0% and 5%:**<span style='font-size:18px; color:green;'> {zero_to_5:.2f}%,</span>", unsafe_allow_html=True)
    st.markdown(f"**% of Stocks Above 5%:** <span style='font-size:18px; color:green;'> {above_5:.2f}%, </span>", unsafe_allow_html=True)
    st.markdown(f"**% of Stocks Above 10%:**<span style='font-size:18px; color:green;'> {above_10:.2f}% </span>", unsafe_allow_html=True)



# --- Streamlit App ---
st.title("📈 CSE Gem Finder by CSE Maverick")
st.markdown("💡An intelligent assistant to help you discover high-potential stocks")
st.markdown("Discover Maverick's Picks generated by his Magic algorithm, or identify opportunities yourself by leveraging these technical analysis tools!!")
st.markdown("Let's find Gems!")


# Add a button to force a data reload (clears cache)
if st.button("Reload Data"):
    load_data.clear() # Clear the data cache
    init_connection.clear() # Clear the connection cache
    st.rerun() # Rerun the app immediately

def get_mavericks_picks(results_df):
    """Filters stocks for Mavericks Picks based on Tier 1 and Tier 2 conditions."""
    # Ensure numeric columns are properly typed
    for col in ['turnover', 'volume', 'relative_strength']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
    
    
    tier_1_conditions = (
        (results_df['rsi_divergence'] == "Bullish Divergence") &
        (results_df['volume_analysis'].isin(["Emerging Bullish Momentum", "Increase in weekly Volume Activity Detected"]))
    ) & (
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) &
        (results_df['relative_strength'] >= 1)
    )

    tier_2_conditions = (
        (results_df['volume_analysis'].isin(["Emerging Bullish Momentum", "High Bullish Momentum"]))&
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) &
        (results_df['relative_strength'] >= 1)
    ) | (
        (results_df['rsi_divergence'] == "Bullish Divergence")&
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) 

    )

    tier_1_picks = results_df[tier_1_conditions]
    tier_2_picks = results_df[tier_2_conditions]

    return tier_1_picks, tier_2_picks

try:
    df = load_data()

    if df.empty:
        st.warning("No data found in the table.")
        st.stop()

    # Remove unwanted columns
    df = df.drop(columns=[col for col in ['id'] if col in df.columns])
    
    # Rename headers
    #df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Format numeric values with commas
    #for col in df.select_dtypes(include=['float64', 'int64']).columns:
    #    df[col] = df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")


    # === Display Filtered Table ===
    #st.subheader("📄 Filtered Analysis Results")
    #st.dataframe(df, use_container_width=True)

    # === Filters Section ===
    st.markdown("### 🔍 Apply Filters")

    # Dropdown filters
    selected_symbol = st.selectbox("Select Symbol", options=["All"] + list(df['symbol'].unique()))
    selected_divergence = st.selectbox("Select Divergence Check", options=["All"] + list(df['rsi_divergence'].dropna().unique()))
    selected_volume_analysis = st.selectbox("Select Volume Analysis", options=["All"] + list(df['volume_analysis'].dropna().unique()))
    
    # Turnover ranges
    turnover_ranges = {
        "100K-1M": (100000, 1000000),
        "1M-10M": (1000000, 10000000),
        "10M-100M": (10000000, 100000000),
        "100M+": (100000000, float('inf'))
    }
    selected_turnover_ranges = st.multiselect(
        "Select Turnover Ranges",
        options=list(turnover_ranges.keys()),
        default=["100K-1M", "1M-10M"]
    )

    # Range sliders
    rsi_range = st.slider("RSI Range", float(df['rsi'].min()), float(df['rsi'].max()), (30.0, 70.0))
    date_range = st.slider(
        "Date Range",
        min_value=df['last_updated'].min().date(),
        max_value=df['last_updated'].max().date(),
        value=(df['last_updated'].min().date(), df['last_updated'].max().date())
    )
    
    # EMA Checker
    st.markdown("### EMA Checker")
    ema_20_check = st.checkbox("Price Above EMA 20")
    ema_50_check = st.checkbox("Price Above EMA 50")
    ema_100_check = st.checkbox("Price Above EMA 100")
    ema_200_check = st.checkbox("Price Above EMA 200")
    
    
    st.markdown("## Filtered Results")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_symbol != "All":
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
    if selected_divergence != "All":
        filtered_df = filtered_df[filtered_df['rsi_divergence'] == selected_divergence]
    if selected_volume_analysis != "All":
        filtered_df = filtered_df[filtered_df['volume_analysis'] == selected_volume_analysis]
    filtered_df = filtered_df[
        (filtered_df['rsi'].between(rsi_range[0], rsi_range[1])) &
        (filtered_df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]
    
    # Apply turnover range filters
    if selected_turnover_ranges:
        turnover_conditions = []
        for range_key in selected_turnover_ranges:
            min_turnover, max_turnover = turnover_ranges[range_key]
            turnover_conditions.append(
                (filtered_df['turnover'] >= min_turnover) & (filtered_df['turnover'] < max_turnover)
            )
        filtered_df = filtered_df[pd.concat(turnover_conditions, axis=1).any(axis=1)]

    # Apply EMA filters
    if ema_20_check and 'ema_20' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_20']]
    if ema_50_check and 'ema_50' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_50']]
    if ema_100_check and 'ema_100' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_100']]
    if ema_200_check and 'ema_200' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_200']]

    
    # Rename headers
    filtered_df.columns = [col.replace('_', ' ').title() for col in filtered_df.columns]

    numeric_columns = [
    'Closing Price', 'Prev Close', 'Turnover'
    ]
    
    for col in numeric_columns:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
     # Sort the table by Turnover in descending order
    if 'Turnover' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Turnover', ascending=False)
        
    # Format numeric values with commas
    for col in filtered_df.select_dtypes(include=['float64', 'int64']).columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")

    filtered_df = filtered_df.drop(columns=[col for col in ['Vol Avg 5D','Vol Avg 20D', 'Ema 20', 'Ema 50', 'Ema 100', 'Ema 200', 'Last Updated'] if col in filtered_df.columns])
        
    # Display the filtered table
    st.dataframe(filtered_df, use_container_width=True)
    
    if not filtered_df.empty:
        st.markdown("## 💎 Maverick's Potential Gems")
        
        # Add a date picker for filtering Maverick's Picks
        selected_maverick_date = st.date_input(
        "Select Start Date for Maverick's Picks",
        value=filtered_df['Date'].min().date(),  # Default to the earliest date in the filtered data
        min_value=filtered_df['Date'].min().date(),
        max_value=filtered_df['Date'].max().date()
        )
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
        'Turnover', 'Volume', 'Relative Strength', 'Closing Price', 'Prev Close'
        ]
        for col in numeric_columns:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

        
        # Filter data based on the selected date
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure Date column is datetime
        maverick_filtered_df = df[df['date'] >= pd.to_datetime(selected_maverick_date)]
        columns_to_remove = ['Vol Avg 5D', 'Vol Avg 20D','Last Updated']
        maverick_filtered_df = maverick_filtered_df.drop(columns=[col for col in columns_to_remove if col in maverick_filtered_df.columns])       
        # Debugging: Display the filtered DataFrame
        #st.write("Filtered Maverick DataFrame:", maverick_filtered_df)
        
        # Get Tier 1 and Tier 2 picks
        tier_1_picks, tier_2_picks = get_mavericks_picks(maverick_filtered_df)

        # Display Tier 1 Picks
        st.markdown("### 🌟 Tier 1 Picks")
        st.markdown("These are the counters identified by Maverick as having the highest potential for Gains.")
        if not tier_1_picks.empty:
            
            columns_to_remove = ['vol_avg_5d', 'vol_avg_20d']
            tier_1_picks = tier_1_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            # Format numeric values with commas
            for col in ['Turnover', 'Volume']:
                if col in tier_1_picks.columns:
                    tier_1_picks[col] = tier_1_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_1_picks = tier_1_picks.sort_values(by='date', ascending=False)  
            
            st.markdown("These are the counters identified by Maverick as having the highest potential for Gains.")
            st.dataframe(tier_1_picks, use_container_width=True)
            
        else:
            st.info("No stocks meet Tier 1 conditions.")
        # Display Tier 2 Picks
        st.markdown("### 🔹Tier  2 Picks")
        if not tier_2_picks.empty:
            
            columns_to_remove = ['vol_avg_5d', 'vol_avg_20d']
            tier_2_picks = tier_2_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            # Format numeric values with commas
            for col in ['turnover', 'volume']:
                if col in tier_2_picks.columns:
                    tier_2_picks[col] = tier_2_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_2_picks = tier_2_picks.sort_values(by='date', ascending=False)
            
            
            st.markdown("These stocks show moderate upside potential compared to the broader market. While not as strong as Tier 1 picks, they still present relatively favorable opportunities._")
            st.markdown("Pay attention to the stocks that have recurring mentions in the list, they have much better chances!")
            st.dataframe(tier_2_picks, use_container_width=True)
        else:
            st.info("No stocks meet Tier 2 conditions.")
            
        if not tier_2_picks.empty:
         # Find the most recurring stocks and their counts
            recurring_stocks = tier_2_picks['symbol'].value_counts()
            recurring_stocks = recurring_stocks[recurring_stocks >= 2]  # Filter stocks with count >= 2

            if not recurring_stocks.empty:
                st.markdown("Most Recurring Stocks in Tier 2 Picks:")
                for stock, count in recurring_stocks.items():
                    st.markdown(f"- **{stock}**: {count} times")
            else:
                st.info("No recurring stocks found with a count of 2 or more.")
            
        
        if not tier_2_picks.empty:
            # Call the performance calculation function
            calculate_performance(tier_2_picks)
    
    
    # === Legend Section ===
    st.markdown("## 📘 Legend: Understanding Key Terms")
    st.markdown("""
Here are some key terms to help you understand the analysis better:

- **📈 Relative Strength (RS)**:
  - A momentum indicator that compares the performance of a stock to the overall market or to the ASI.
  - **RS >= 1**: Indicates the stock is outperforming the market.
  - **RS < 1**: Indicates the stock is underperforming the market.

- **🔄 Bullish Divergence**:
  - Occurs when the stock's price is making lower lows, but the RSI (Relative Strength Index) is making higher lows.
  - This is a potential signal for a reversal to the upside.

- **📊 Volume Analysis Criteria**:
  - **Emerging Bullish Momentum**: Indicates a sudden increase in buying activity,compared to their weekly average volumes.Suggesting in start of interest shown to the stock.
  - **High Bullish Momentum**: Indicates break-out buying activity, higher volume than their weekly or monthly averages.Suggesting a strong,commited interest in the stock.
  - **Increase in Weekly Volume Activity Detected**: Highlights stocks with a gradual increase in trading volume compared to their weekly average.

- **📐 EMAs (Exponential Moving Averages)**:
  - A type of moving average that gives more weight to recent prices, making it more responsive to new information.
  - **EMA 20**: Short-term trend indicator.
  - **EMA 50**: Medium-term trend indicator.
  - **EMA 100**: Long-term trend indicator.
  - **EMA 200**: Very long-term trend indicator, often used to identify major support or resistance levels.

We hope this helps you better understand the analysis and make informed decisions! 🚀
""")

except Exception as e:
    st.error(f"An error occurred: {e}")
