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

# Page configuration
st.set_page_config(page_title="StockXpert", layout="wide", page_icon="📈")

# Custom CSS for animations and styling
st.markdown(
    """
    <style>
    @keyframes gradientBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBackground 15s ease infinite;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stHeader {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown('<p class="stHeader">📈 TradeSense: AI-Powered Stock Predictions</p>', unsafe_allow_html=True)
st.sidebar.markdown("### 🛠️ Input Parameters")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators."""
    df = df.copy()
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return df

def create_features(df):
    """Create features for prediction."""
    df = df.copy()
    df = calculate_technical_indicators(df)
    df['Price_Change'] = df['Close'].pct_change().fillna(0)
    return df

def train_model(df, forecast_days, model_type='rf'):
    """Train the prediction model."""
    df = create_features(df)
    features = ['Close', 'volume_adi', 'volatility_bbm', 'trend_macd', 'momentum_rsi', 'Price_Change']
    X = df[features].copy()
    y = df['Close'].shift(-forecast_days)
    X = X.iloc[:-forecast_days]
    y = y.iloc[:-forecast_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    forecast_data = df[features].iloc[-forecast_days:].copy()
    forecast_data_scaled = scaler.transform(forecast_data)
    predictions = model.predict(forecast_data_scaled)
    return predictions

def plot_candlestick(df):
    """Plot interactive candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title='Stock Price History (Candlestick Chart)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def get_recommendation(current_price, predicted_prices):
    """Generate trading recommendation with emojis."""
    avg_predicted = np.mean(predicted_prices)
    percent_change = ((avg_predicted - current_price) / current_price) * 100
    if percent_change > 2:
        return "🚀 Strong Buy", "green", "The stock is expected to rise significantly. A great time to invest!"
    elif percent_change > 0:
        return "📈 Buy", "lightgreen", "The stock is expected to rise. Consider buying."
    elif percent_change < -2:
        return "🔥 Strong Sell", "red", "The stock is expected to drop significantly. Consider selling."
    elif percent_change < 0:
        return "📉 Sell", "pink", "The stock is expected to drop. Consider selling."
    else:
        return "🤝 Hold", "gray", "The stock is expected to remain stable. Hold your position."

def main():
    # Sidebar inputs
    symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL').upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_date = st.sidebar.date_input('Start Date', start_date)
    end_date = st.sidebar.date_input('End Date', end_date)

    # Load data
    if st.sidebar.button('Load Data'):
        if start_date < end_date:
            with st.spinner('Loading data...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                df = load_data(symbol, start_date, end_date)
                if df is not None:
                    st.session_state.data = df
                    st.success('✅ Data loaded successfully!')
                else:
                    st.error('❌ Error loading data. Please check the stock symbol and try again.')
        else:
            st.error('❌ Error: End date must be after start date.')

    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data

        # Display candlestick chart
        plot_candlestick(data)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="stMetric">📊 Current Price<br><h3>${:.2f}</h3></div>'.format(data['Close'].iloc[-1]), unsafe_allow_html=True)
        with col2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            st.markdown('<div class="stMetric">📈 Daily Change<br><h3>${:.2f}</h3></div>'.format(price_change), unsafe_allow_html=True)
        with col3:
            volume = data['Volume'].iloc[-1]
            st.markdown('<div class="stMetric">📦 Volume<br><h3>{:,.0f}</h3></div>'.format(volume), unsafe_allow_html=True)

        # Prediction section
        st.markdown('## 🔮 Price Prediction')
        model_type = st.radio('Select Model', ['Random Forest', 'Linear Regression'])
        forecast_days = st.slider('Forecast Days', 1, 30, 5)

        if st.button('Generate Forecast'):
            with st.spinner('Generating forecast...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                predictions = train_model(data, forecast_days, 'rf' if model_type == 'Random Forest' else 'lr')
                last_date = data.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})
                forecast_df.set_index('Date', inplace=True)
                st.dataframe(forecast_df)

                # Recommendation
                recommendation, color, explanation = get_recommendation(data['Close'].iloc[-1], predictions)
                st.markdown(
                    f"""
                    <div style='
                        padding: 20px;
                        border-radius: 12px;
                        background-color: {color};
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        font-size: 24px;
                    '>
                        🎯 Recommendation: {recommendation}<br>
                        <span style='font-size: 16px;'>{explanation}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Plot forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, name='Forecast', line=dict(color='red', dash='dash')))
                fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

        # Recent data
        st.markdown('## 📅 Recent Data')
        st.dataframe(data.tail())

if __name__ == "__main__":
    main()
