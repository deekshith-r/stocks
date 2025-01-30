import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from ta import add_all_ta_features
import time
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced animations and responsive design
st.markdown("""
<style>
:root {
    --primary: #6366f1;
    --secondary: #3b82f6;
    --success: #22c55e;
    --danger: #ef4444;
    --dark: #1e293b;
    --light: #f8fafc;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: var(--light);
}

.stButton>button {
    background: var(--primary);
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton>button:hover {
    background: var(--secondary);
    transform: scale(1.05);
}

.stMetric {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeIn 0.6s ease;
}

.stMetric h3 {
    color: var(--primary);
    font-size: 2rem;
    margin: 0.5rem 0;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--light) !important;
    animation: fadeIn 0.8s ease;
}

.stDataFrame {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

@media (max-width: 768px) {
    .stMetric h3 {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Ticker data URL (using a sample dataset)
TICKER_DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

@st.cache_data
def load_ticker_data():
    """Load supported tickers data"""
    try:
        response = requests.get(TICKER_DATA_URL)
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df[['Symbol', 'Name', 'Sector']]
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Sector'])

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'company_name' not in st.session_state:
    st.session_state.company_name = ""
if 'show_ticker_search' not in st.session_state:
    st.session_state.show_ticker_search = False

# Helper functions (keep load_data, calculate_technical_indicators, 
# create_features, train_model, plot_interactive_chart, 
# get_recommendation same as before)

# Main App Title
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>📈 StockSense AI</h1>", 
            unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    symbol = st.text_input("Stock Symbol", "AAPL").upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    forecast_days = st.number_input("Forecast Days", min_value=1, max_value=365, value=7, step=1)
    model_type = st.radio("Model Type", ["Random Forest", "Linear Regression"])
    
    if st.button("Analyze Stock"):
        with st.spinner("Crunching numbers..."):
            df = load_data(symbol, start_date, end_date)
            if df is not None:
                st.session_state.data = df
                try:
                    ticker_info = yf.Ticker(symbol).info
                    company_name = ticker_info.get('longName', symbol)
                    st.session_state.company_name = f"{company_name} ({symbol})"
                except:
                    st.session_state.company_name = f"({symbol})"
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Check symbol and dates.")

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Dynamic title with company name
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem;'>📊 Analyzing: {st.session_state.company_name}</h2>", 
                unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stMetric">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">💰</span>
                <div>
                    <div>Current Price</div>
                    <h3>${:.2f}</h3>
                </div>
            </div>
        </div>
        """.format(df['Close'].iloc[-1]), unsafe_allow_html=True)
    
    with col2:
        change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        st.markdown("""
        <div class="stMetric">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">📅</span>
                <div>
                    <div>24h Change</div>
                    <h3>${:.2f}</h3>
                </div>
            </div>
        </div>
        """.format(change), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stMetric">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">📊</span>
                <div>
                    <div>Volume</div>
                    <h3>{:,.0f}</h3>
                </div>
            </div>
        </div>
        """.format(df['Volume'].iloc[-1]), unsafe_allow_html=True)

    # Interactive chart
    plot_interactive_chart(df)

    # Recent Market Data Table
    st.subheader("📊 Recent Market Data")
    recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).reset_index()
    recent_data.insert(0, 'SL No', range(1, 6))
    recent_data.rename(columns={'Date': 'Trade Date'}, inplace=True)
    
    st.dataframe(
        recent_data.style.format({
            'Trade Date': lambda x: x.strftime('%Y-%m-%d'),
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Low': '${:.2f}',
            'Close': '${:.2f}',
            'Volume': '{:,.0f}'
        }).hide(axis='index'),
        use_container_width=True,
        hide_index=True
    )

    # Generate predictions
    if st.button("Generate Forecast"):
        with st.spinner("Training AI model..."):
            start_time = time.time()
            predictions = train_model(
                df, 
                forecast_days,
                'rf' if model_type == "Random Forest" else 'lr'
            )
            
            # Create forecast dataframe
            last_date = df.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='B'
            )
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Prediction': predictions
            }).set_index('Date')
            
            # Show results
            st.subheader("📅 Forecast Results")
            plot_interactive_chart(df, forecast_df)
            
            # Recommendation
            current_price = df['Close'].iloc[-1]
            rec_text, rec_color, rec_icon = get_recommendation(current_price, predictions)
            
            st.markdown(f"""
            <div style='
                background: {rec_color}20;
                border-left: 4px solid {rec_color};
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                animation: fadeIn 0.6s ease;
            '>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 2rem;">{rec_text.split()[0]}</span>
                    <div>
                        <h3 style="margin: 0;">{rec_text}</h3>
                        <p style="margin: 0; opacity: 0.9;">Average predicted price: ${np.mean(predictions):.2f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Predicted Prices Table
            st.subheader("📈 Predicted Prices")
            forecast_display = forecast_df.reset_index()
            forecast_display.insert(0, 'SL No', range(1, len(forecast_df)+1))
            forecast_display.columns = ['SL No', 'Prediction Date', 'Predicted Price']
            
            st.dataframe(
                forecast_display.style.format({
                    'Prediction Date': lambda x: x.strftime('%Y-%m-%d'),
                    'Predicted Price': '${:.2f}'
                }).hide(axis='index'),
                use_container_width=True,
                hide_index=True
            )
            
            # Performance metrics
            st.subheader("⚡ Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Time", f"{time.time() - start_time:.2f}s")
            with col2:
                st.metric("Prediction Range", f"{forecast_days} days")
            with col3:
                st.metric("Model Type", model_type)

# Ticker Lookup Section (Collapsible)
with st.expander("🔍 Don't know the ticker symbol? Search here"):
    st.markdown("### Ticker Symbol Lookup")
    search_query = st.text_input("Search companies (name or symbol):", "")
    
    if st.button("Load Supported Tickers"):
        with st.spinner("Loading ticker database..."):
            ticker_data = load_ticker_data()
            
            if not ticker_data.empty:
                if search_query:
                    search = search_query.lower()
                    filtered_data = ticker_data[
                        ticker_data['Name'].str.lower().str.contains(search) |
                        ticker_data['Symbol'].str.lower().str.contains(search)
                    ]
                else:
                    filtered_data = ticker_data

                st.markdown(f"**Found {len(filtered_data)} companies**")
                
                st.dataframe(
                    filtered_data.style.format({
                        'Symbol': '{}',
                        'Name': '{}',
                        'Sector': '{}'
                    }).hide(axis='index'),
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Symbol": "Ticker Symbol",
                        "Name": "Company Name",
                        "Sector": "Industry Sector"
                    }
                )
            else:
                st.error("Failed to load ticker data. Please try again later.")

# Initial state message
else:
    st.markdown("""
    <div style='
        text-align: center;
        padding: 4rem;
        opacity: 0.8;
        animation: fadeIn 1s ease;
    '>
        <h2>🔍 Analyze Any Stock</h2>
        <p>Enter a stock symbol and date range to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)
