import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from alpha_vantage import timeseries

# Initialize the sentiment analyzer once
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

def ML_web_pp():
    st.title('ML Web App Dashboard')

def stock_predictor():
    st.title('Stock Dashboard')
    
    # Sidebar for ticker input and date range
    ticker = st.sidebar.selectbox('Ticker Symbol', ['MSFT', 'GOOGL', 'AAPL', 'AMZN', 'NVDA'])
    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2021-01-01'))
    
    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if not data.empty:
        tab1, tab2, tab3 = st.tabs(["Pricing Data", "Fundamental Data", "Top News"])
        
        with tab1:
            st.write(f'<b>{ticker} Adjusted Close Price</b>', unsafe_allow_html=True)
            fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Adjusted Close Price')
            st.plotly_chart(fig)

            st.write('<b>Price Data</b>', unsafe_allow_html=True)
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
            
            data['% Change'] = data['Adj Close'].pct_change()
            data.dropna(inplace=True)
            annualized_return = (data['% Change'].mean() * 252) * 100
            st.write(f'<b>Annualized Return:</b> {annualized_return:.2f}%', unsafe_allow_html=True)
            
            std_dev = data['% Change'].std() * 252 ** 0.5
            st.write(f'<b>Volatility:</b> {std_dev:.2f}', unsafe_allow_html=True)
            
            data['SMA50'] = data['Adj Close'].rolling(window=50).mean()
            data['SMA200'] = data['Adj Close'].rolling(window=200).mean()
            data['Buy Signal'] = (data['SMA50'] > data['SMA200']) & (data['SMA50'].shift(1) <= data['SMA200'].shift(1))
            
            st.write('<b>Buy Signals</b>', unsafe_allow_html=True)
            st.dataframe(data[['Adj Close', 'SMA50', 'SMA200', 'Buy Signal']])
            
            fig2 = px.line(data, x=data.index, y=['Adj Close', 'SMA50', 'SMA200'], title=f'{ticker} Price with Buy Signals')
            fig2.add_scatter(x=data[data['Buy Signal']].index, y=data[data['Buy Signal']]['Adj Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal')
            st.plotly_chart(fig2)
        
        with tab2:
            st.subheader('Fundamental Data')
            ticker_obj = yf.Ticker(ticker)
            
            st.write('<b>Balance Sheet</b>', unsafe_allow_html=True)
            balance_sheet = ticker_obj.balance_sheet
            st.write(balance_sheet)
            
            st.write('<b>Income Statement</b>', unsafe_allow_html=True)
            income_statement = ticker_obj.financials
            st.write(income_statement)
            
            st.write('<b>Cash Flow</b>', unsafe_allow_html=True)
            cash_flow = ticker_obj.cashflow
            st.write(cash_flow)
        
        with tab3:
            st.subheader('News')
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            for item in news:
                title = item['title']
                sentiment = analyze_sentiment(title)
                st.write(f"{title}: {item['link']} - Sentiment: {sentiment:.2f}")
    
    ### Test Alpha Vantage API in Jupyter
    #from alpha_vantage.fundamentaldata import FundamentalData
    #key = 

if __name__ == '__main__':
    stock_predictor()
    
