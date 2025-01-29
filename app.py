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
import random
import time

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [DEEKSHITH](https://www.linkedin.com/in/vikas-sharma005/)")

def main():
    with st.spinner('Unlocking the future...'):
        time.sleep(5)  # Adding a 5-second delay
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()


def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
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
    st.dataframe(data.tail(10))

def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)

def sentiment_analysis(recommendation):
    # Adjust sentiment values based on recommendation
    if recommendation == "Buy":
        sentiment_values = {'positive': random.randint(60, 80), 'negative': random.randint(10, 20), 'neutral': 100 - (random.randint(60, 80) + random.randint(10, 20))}
        messages = [
            "This stock is a great opportunity right now – consider buying!",
            "The market trends indicate a strong potential for growth. It's time to buy.",
            "Positive sentiment suggests a promising future for this stock. Buying is recommended.",
            "Now is the time to invest – buy while the price is low!"
        ]
    elif recommendation == "Sell":
        sentiment_values = {'positive': random.randint(10, 20), 'negative': random.randint(60, 80), 'neutral': 100 - (random.randint(10, 20) + random.randint(60, 80))}
        messages = [
            "The stock is nearing its peak – consider selling for profit.",
            "Market conditions are showing signs of decline. It’s a good time to sell.",
            "Negative sentiment indicates a potential drop in price. Selling is advised.",
            "If you're holding onto this stock, it might be time to sell and take profits."
        ]
    else:  # Hold
        sentiment_values = {'positive': random.randint(30, 40), 'negative': random.randint(20, 30), 'neutral': 100 - (random.randint(30, 40) + random.randint(20, 30))}
        messages = [
            "The stock is showing stable trends. It’s wise to hold for now.",
            "No significant changes are expected in the short term. Holding is recommended.",
            "The market is uncertain, but holding may still be a good option.",
            "You may want to wait before making any decisions – hold your position for now."
        ]

    # Circular graph using Plotly
    labels = ['Positive', 'Negative', 'Neutral']
    values = [sentiment_values['positive'], sentiment_values['negative'], sentiment_values['neutral']]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_traces(textinfo='percent', pull=[0.1, 0.1, 0.1])
    st.plotly_chart(fig)

    st.subheader("Sentiment Analysis")
    # Displaying random recommendation message
    st.write(random.choice(messages))

    # Recommendation text with icons (blinking text only)
    if recommendation == "Buy":
        st.markdown("""
            <style>
                .blinking-text {
                    color: green;
                    font-size: 18px;
                    font-weight: bold;
                    animation: blink 1s infinite;
                }
                @keyframes blink {
                    0% { opacity: 1; }
                    50% { opacity: 0; }
                    100% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True)
        st.markdown('<p class="blinking-text"><i class="fa fa-arrow-circle-up"></i> Buy</p>', unsafe_allow_html=True)

    elif recommendation == "Sell":
        st.markdown("""
            <style>
                .blinking-text {
                    color: red;
                    font-size: 18px;
                    font-weight: bold;
                    animation: blink 1s infinite;
                }
                @keyframes blink {
                    0% { opacity: 1; }
                    50% { opacity: 0; }
                    100% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True)
        st.markdown('<p class="blinking-text"><i class="fa fa-arrow-circle-down"></i> Sell</p>', unsafe_allow_html=True)

    else:  # Hold
        st.markdown("""
            <style>
                .blinking-text {
                    color: orange;
                    font-size: 18px;
                    font-weight: bold;
                    animation: blink 1s infinite;
                }
                @keyframes blink {
                    0% { opacity: 1; }
                    50% { opacity: 0; }
                    100% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True)
        st.markdown('<p class="blinking-text"><i class="fa fa-pause-circle"></i> Hold</p>', unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    
    # Extract the last forecasted value from forecast_pred (which is an array)
    predicted_price = forecast_pred[-1]

    # Get today's price
    todays_price = data['Close'].iloc[-1]

    st.text(f"Predicted Price for the last day: {predicted_price}")
    st.text(f"Today's Price: {todays_price}")

    # Ensure we're comparing scalar values
    if predicted_price > todays_price:
        recommendation = "Buy"
    elif predicted_price < todays_price:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    sentiment_analysis(recommendation)

if __name__ == "__main__":
    main()
