import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import numpy as np
import random
import time

# Page config
st.set_page_config(page_title="Stock Price Predictions", layout="wide")
st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [DEEKSHITH](https://www.linkedin.com/in/vikas-sharma005/)")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def main():
    with st.spinner('Unlocking the future...'):
        time.sleep(2)  # Reduced delay for better UX
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    
    # Get stock data first
    get_stock_data()
    
    if st.session_state.data is not None:
        if option == 'Visualize':
            tech_indicators()
        elif option == 'Recent Data':
            dataframe()
        else:
            predict()
    else:
        st.warning("Please enter a valid stock symbol and date range to proceed.")

@st.cache_data
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def get_stock_data():
    symbol = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
    symbol = symbol.upper()
    
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration (days)', value=3000, min_value=1)
    before = today - datetime.timedelta(days=duration)
    
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', today)
    
    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success(f'Start date: `{start_date}`\n\nEnd date: `{end_date}`')
            data = download_data(symbol, start_date, end_date)
            if data is not None:
                st.session_state.data = data
            else:
                st.error("No data found for the specified symbol and date range.")
        else:
            st.sidebar.error('Error: End date must fall after start date')

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', 
                     ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])
    
    data = st.session_state.data
    
    # Bollinger Bands
    bb_indicator = BollingerBands(data['Close'])
    data['bb_h'] = bb_indicator.bollinger_hband()
    data['bb_l'] = bb_indicator.bollinger_lband()
    bb = data[['Close', 'bb_h', 'bb_l']].copy()
    
    # MACD
    macd = MACD(data['Close']).macd()
    
    # RSI
    rsi = RSIIndicator(data['Close']).rsi()
    
    # SMA
    sma = SMAIndicator(data['Close'], window=14).sma_indicator()
    
    # EMA
    ema = EMAIndicator(data['Close']).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data['Close'])
    elif option == 'BB':
        st.write('Bollinger Bands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe():
    st.header('Recent Data')
    st.dataframe(st.session_state.data.tail(10))

def predict():
    model = st.radio('Choose a model', 
                    ['LinearRegression', 'RandomForestRegressor', 
                     'ExtraTreesRegressor', 'KNeighborsRegressor', 
                     'XGBRegressor'])
    num = st.number_input('How many days forecast?', value=5, min_value=1)
    num = int(num)
    
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor(n_estimators=100)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor(n_estimators=100)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor(n_neighbors=5)
        else:
            engine = XGBRegressor(n_estimators=100)
        
        model_engine(engine, num)

def sentiment_analysis(recommendation):
    if recommendation == "Buy":
        sentiment_values = {
            'positive': random.randint(60, 80),
            'negative': random.randint(10, 20)
        }
        sentiment_values['neutral'] = 100 - (sentiment_values['positive'] + sentiment_values['negative'])
        messages = [
            "This stock shows strong buying signals right now.",
            "Market trends indicate potential growth ahead.",
            "Technical indicators suggest a bullish outlook.",
            "Consider adding this stock to your portfolio."
        ]
    elif recommendation == "Sell":
        sentiment_values = {
            'positive': random.randint(10, 20),
            'negative': random.randint(60, 80)
        }
        sentiment_values['neutral'] = 100 - (sentiment_values['positive'] + sentiment_values['negative'])
        messages = [
            "Consider taking profits at current levels.",
            "Technical indicators suggest a bearish trend.",
            "Market conditions indicate potential decline.",
            "It might be wise to reduce position size."
        ]
    else:
        sentiment_values = {
            'positive': random.randint(30, 40),
            'negative': random.randint(20, 30)
        }
        sentiment_values['neutral'] = 100 - (sentiment_values['positive'] + sentiment_values['negative'])
        messages = [
            "The market shows neutral signals.",
            "Consider maintaining current position.",
            "No strong directional indicators present.",
            "Watch for clearer signals before acting."
        ]

    # Create sentiment pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[sentiment_values['positive'], 
                sentiment_values['negative'], 
                sentiment_values['neutral']],
        hole=0.3
    )])
    
    fig.update_traces(textinfo='percent', pull=[0.1, 0.1, 0.1])
    st.plotly_chart(fig)

    st.subheader("Market Sentiment")
    st.write(random.choice(messages))
    
    # Recommendation styling
    color = {
        "Buy": "green",
        "Sell": "red",
        "Hold": "orange"
    }[recommendation]
    
    st.markdown(f"""
        <style>
            .recommendation {{
                color: {color};
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                border: 2px solid {color};
                border-radius: 10px;
                margin: 20px 0;
            }}
        </style>
        <div class="recommendation">
            Recommendation: {recommendation}
        </div>
    """, unsafe_allow_html=True)

def model_engine(model, forecast_days):
    data = st.session_state.data
    scaler = StandardScaler()
    
    # Prepare the data
    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-forecast_days)
    
    # Create features (you can add more features here)
    df['SMA_20'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(data['Close']).rsi()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Prepare features and target
    features = ['Close', 'SMA_20', 'RSI']
    X = df[features]
    y = df['Target']
    
    # Scale the features
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled[:-forecast_days], 
        y[:-forecast_days], 
        test_size=0.2, 
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.4f}")
    
    # Forecast future prices
    forecast_features = X_scaled[-forecast_days:]
    forecast_prices = model.predict(forecast_features)
    
    # Display forecasts
    st.subheader(f"Price Forecasts for Next {forecast_days} Days")
    forecast_df = pd.DataFrame({
        'Day': range(1, forecast_days + 1),
        'Predicted Price': forecast_prices
    })
    st.dataframe(forecast_df)
    
    # Determine recommendation
    last_price = data['Close'].iloc[-1]
    avg_forecast = np.mean(forecast_prices)
    
    if avg_forecast > last_price * 1.02:  # 2% threshold
        recommendation = "Buy"
    elif avg_forecast < last_price * 0.98:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
        
    sentiment_analysis(recommendation)

if __name__ == "__main__":
    main()
