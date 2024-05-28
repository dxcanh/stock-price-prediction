import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from Forecast import forecast_stock

st.set_page_config(
    page_title="Dự đoán giá cổ phiếu",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
body {
    font-family: Arial, sans-serif;
    color: #333;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.stAlert {
    font-size: 1.1rem;
}
.checkbox-label {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}
.checkbox-label .checkbox-icon {
    margin-right: 5px;
    border: 2px solid #333;
    width: 18px;
    height: 18px;
    border-radius: 3px;
    cursor: pointer;
    display: inline-block;
}
.checkbox-label .checkbox-icon.checked {
    background-color: #333;
}
.stPlotlyChart {
    margin: 0 auto; 
    display: block; 
}
</style>
""",
    unsafe_allow_html=True,
)

def format_market_cap(market_cap):
    if market_cap >= 1e12:
        return f"{market_cap / 1e12:.2f} nghìn tỷ USD"
    elif market_cap >= 1e9:
        return f"{market_cap / 1e9:.2f} tỷ USD"
    elif market_cap >= 1e6:
        return f"{market_cap / 1e6:.2f} triệu USD"
    elif market_cap >= 1e3:
        return f"{market_cap / 1e3:.2f} nghìn USD"
    else:
        return f"{market_cap:.2f}"

def hide_st_cache_deprecation_warning(func):
    def wrapper(*args, **kwargs):
        with st.empty():
            return func(*args, **kwargs)
    return wrapper

@hide_st_cache_deprecation_warning
@st.cache_data
def load_data(stock, start, end):
    return yf.download(stock, start=start, end=end)

@hide_st_cache_deprecation_warning
@st.cache
def predict_and_visualize(selected_stock, stock_data):
    stock_data = stock_data.asfreq('D').ffill()
    ts = stock_data[['Close']]

    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()

    stock_data['Predicted_Close'] = model_fit.predict(start=0, end=len(ts)-1)

    last_date = stock_data.index[-1]
    next_n_days = pd.date_range(last_date + timedelta(days=1), periods=1, freq="B")

    predictions_n_days = []
    last_train_series = list(ts.Close)
    for _ in range(1):
        model = ARIMA(last_train_series, order=(5, 1, 0))
        model_fit = model.fit()
        next_day_prediction = model_fit.forecast(steps=1)[0]
        
        random_factor = np.random.normal(0, 1)
        next_day_prediction += random_factor
        
        predictions_n_days.append(next_day_prediction)
        last_train_series.append(next_day_prediction)

    for i in range(len(next_n_days)):
        stock_data.loc[next_n_days[i], "Predicted_Close"] = predictions_n_days[i]
        stock_data.loc[next_n_days[i], "Close"] = predictions_n_days[i]
        for col in stock_data.columns:
            if col != "Predicted_Close" and col != "Close":
                stock_data.loc[next_n_days[i], col] = np.nan 
    
    # Dự đoán và tính toán độ chính xác
    forecast_results = forecast_stock(selected_stock, period="2y", interval='1d')
    accuracy = forecast_results['accuracy']
    
    return stock_data, accuracy

end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 20)

st.title("Ứng dụng dự đoán giá cổ phiếu")

st.sidebar.header("Lựa chọn cổ phiếu")
selected_stocks = st.sidebar.text_area("Nhập mã cổ phiếu (cách nhau bởi dấu phẩy):", "")

if selected_stocks:
    stocks = [stock.strip() for stock in selected_stocks.split(',')]
else:
    stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NFLX", "NVDA", "BABA", "V"]

data = []
for stock in stocks:
    stock_data = load_data(stock, start_date, end_date)
    if stock_data.empty:
        continue  

    predicted_stock_data, accuracy = predict_and_visualize(stock, stock_data)
    
    recent_prices = predicted_stock_data['Close'][-4:-1].values
    recent_dates = predicted_stock_data.index[-4:-1].strftime('%Y-%m-%d').tolist()
    predicted_price = predicted_stock_data['Predicted_Close'].iloc[-1]
    volume = predicted_stock_data['Volume'].iloc[-2]  

    data.append([stock, *recent_prices, predicted_price, accuracy, volume])

recent_dates_headers = predicted_stock_data.index[-4:].strftime('%d-%m-%Y').tolist()

df = pd.DataFrame(data, columns=['Mã cổ phiếu', recent_dates_headers[0], recent_dates_headers[1], recent_dates_headers[2], recent_dates_headers[3], 'Độ chính xác', 'Volume'])
df = df.sort_values(by='Độ chính xác', ascending=False)
df.index = range(1, len(df) + 1)

st.subheader("Bảng thông tin cổ phiếu")
def color_comparison(row):
    color = 'white'
    if row['27-05-2024'] > row['24-05-2024']:
        color = '#00FF00'
    else:
        color = 'red'
    return ['color: white'] * (len(row) - 1) + [f'color: {color}']

styled_df = df.style.apply(color_comparison, axis=1, subset=['24-05-2024', '27-05-2024'])
st.dataframe(styled_df)

selected_stock = st.selectbox("Chọn mã cổ phiếu để xem chi tiết:", [""] + list(df['Mã cổ phiếu'].unique()))

if selected_stock:
    stock_data = load_data(selected_stock, start_date, end_date)
    
    if stock_data.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã {selected_stock}. Vui lòng kiểm tra lại.")
    else:
        predicted_stock_data, accuracy = predict_and_visualize(selected_stock, stock_data)

        st.subheader(f"Thông tin về {selected_stock}")
        stock_info = yf.Ticker(selected_stock).info
        st.write(f"**Tên công ty:** {stock_info['longName']}")
        st.write(f"**Ngành:** {stock_info['industry']}")
        st.write(f"**Vốn hóa:** {format_market_cap(stock_info['marketCap'])}")

        with st.expander("Xem dữ liệu"):
            st.dataframe(predicted_stock_data.round(2))

        st.subheader(f"Biểu đồ giá {selected_stock}")

        ma_options = [50, 100, 200]
        line_visibility = {}
        for ma in ma_options:
            col_name = f"MA_{ma}"
            predicted_stock_data[col_name] = predicted_stock_data["Close"].rolling(window=ma, min_periods=1).mean()
            line_visibility[col_name] = st.checkbox(f"Hiển thị {col_name}", value=True)

        ma_colors = ['#E6C300', '#EC5840', '#5CD07F']

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=predicted_stock_data.index,
            open=predicted_stock_data['Open'],
            high=predicted_stock_data['High'],
            low=predicted_stock_data['Low'],
            close=predicted_stock_data['Close'],
            name="Giá",
            increasing_line_color='green', 
            decreasing_line_color='red',
            visible=True,
        ))

        for i, ma in enumerate(ma_options):
            col_name = f"MA_{ma}"
            if line_visibility[col_name]:
                fig.add_trace(go.Scatter(
                    x=predicted_stock_data.index,
                    y=predicted_stock_data[col_name],
                    mode='lines',
                    name=col_name,
                    line=dict(color=ma_colors[i], width=1)
                ))

        fig.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Giá (USD)",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            width=1200,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
            yaxis=dict(
                fixedrange=False
            )
        )

        st.plotly_chart(fig)
else:
    st.warning("Vui lòng chọn mã cổ phiếu để xem chi tiết.")
