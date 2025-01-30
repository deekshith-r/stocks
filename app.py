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
import plotly.express as px
from ta import add_all_ta_features
from ta.utils import dropna

# Page configuration
st.set_page_config(page_title="StockXpert", layout="wide")

# Title and description
st.title('TradeSense')
st.sidebar.info('TradeSense - Leveraging Deep Learning For Stock Prediction')

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
    
    # Technical indicators
    df = calculate_technical_indicators(df)
    
    # Price changes
    df['Price_Change'] = df['Close'].pct_change().fillna(0)
    
    return df

def train_model(df, forecast_days, model_type='rf'):
    """Train the prediction model."""
    # Create features
    df = create_features(df)
    
    # Prepare features
    features = ['Close', 'volume_adi', 'volatility_bbm', 'trend_macd', 'momentum_rsi', 'Price_Change']
    X = df[features].copy()
    
    # Create target (future price changes)
    y = df['Close'].shift(-forecast_days)
    
    # Remove NaN values
    X = X.iloc[:-forecast_days]
    y = y.iloc[:-forecast_days]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select and train model
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)
    
    # Prepare forecast data
    forecast_data = df[features].iloc[-forecast_days:].copy()
    forecast_data_scaled = scaler.transform(forecast_data)
    
    # Make predictions
    predictions = model.predict(forecast_data_scaled)
    
    return predictions

def plot_stock_data(df):
    """Plot stock price chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Close Price',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Stock Price History',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    st.plotly_chart(fig)

def get_recommendation(current_price, predicted_prices):
    """Generate trading recommendation."""
    avg_predicted = np.mean(predicted_prices)
    percent_change = ((avg_predicted - current_price) / current_price) * 100
    
    if percent_change > 2:
        return "Strong Buy", "green"
    elif percent_change > 0:
        return "Buy", "lightgreen"
    elif percent_change < -2:
        return "Strong Sell", "red"
    elif percent_change < 0:
        return "Sell", "pink"
    else:
        return "Hold", "gray"

def main():
    # Sidebar inputs
    st.sidebar.header('Input Parameters')
    
    # Stock symbol input
    symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL').upper()
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_date = st.sidebar.date_input('Start Date', start_date)
    end_date = st.sidebar.date_input('End Date', end_date)
    
    # Load data when user clicks
    if st.sidebar.button('Load Data'):
        if start_date < end_date:
            with st.spinner('Loading data...'):
                df = load_data(symbol, start_date, end_date)
                if df is not None:
                    st.session_state.data = df
                    st.success('Data loaded successfully!')
                else:
                    st.error('Error loading data. Please check the stock symbol and try again.')
        else:
            st.error('Error: End date must be after start date.')
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Display stock price chart
        plot_stock_data(data)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            st.metric("Daily Change", f"${price_change:.2f}")
        with col3:
            volume = data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        # Prediction section
        st.header('Price Prediction')
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.radio('Select Model', ['Random Forest', 'Linear Regression'])
        with col2:
            forecast_days = st.number_input('Forecast Days', min_value=1, max_value=30, value=5)
        
        if st.button('Generate Forecast'):
            with st.spinner('Generating forecast...'):
                # Generate predictions
                predictions = train_model(
                    data,
                    forecast_days,
                    'rf' if model_type == 'Random Forest' else 'lr'
                )
                
                # Create forecast dates
                last_date = data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq='B'  # Business days
                )
                
                # Display forecast results
                st.subheader('Price Forecast')
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted Price': predictions
                })
                forecast_df.set_index('Date', inplace=True)
                st.dataframe(forecast_df)
                
                # Get recommendation
                recommendation, color = get_recommendation(
                    data['Close'].iloc[-1],
                    predictions
                )
                
                # Display recommendation
                st.markdown(f"""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        background-color: {color};
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        font-size: 24px;
                    '>
                        Recommendation: {recommendation}
                    </div>
                """, unsafe_allow_html=True)
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=predictions,
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Stock Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig)
        
        # Recent data
        st.header('Recent Data')
        st.dataframe(data.tail())

if __name__ == "__main__":
    main()
