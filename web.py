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
    initial_sidebar_state="collapsed",
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
@st.cache(persist=True)
def predict_stock_prices(stocks, start_date, end_date):
    predicted_data = {}
    for stock in stocks:
        stock_data = yf.download(stock, start=start_date, end=end_date)
        if stock_data.empty:
            continue

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
        forecast_results = forecast_stock(stock, period="2y", interval='1d')
        accuracy = forecast_results['accuracy']

        predicted_data[stock] = (stock_data, accuracy)

    return predicted_data

end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 20)

st.title("Ứng dụng dự đoán giá cổ phiếu")

st.sidebar.header("Lựa chọn cổ phiếu")
selected_stocks = st.sidebar.text_area("Nhập mã cổ phiếu (cách nhau bởi dấu phẩy):", "")

if selected_stocks:
    stocks = [stock.strip() for stock in selected_stocks.split(',')]
else:
    stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NFLX", "NVDA", "BABA", "V"]

predicted_data = predict_stock_prices(stocks, start_date, end_date)

# Hiển thị bảng thông tin cổ phiếu
data = []
for stock, (stock_data, accuracy) in predicted_data.items():
    recent_prices = stock_data['Close'][-4:-1].values
    recent_dates = stock_data.index[-4:-1].strftime('%Y-%m-%d').tolist()
    predicted_price = stock_data['Predicted_Close'].iloc[-1]
    volume= stock_data['Volume'].iloc[-2]  

    data.append([stock, *recent_prices, predicted_price, accuracy, volume])

recent_dates_headers = stock_data.index[-4:].strftime('%d-%m-%Y').tolist()

df = pd.DataFrame(data, columns=['Mã cổ phiếu', recent_dates_headers[0], recent_dates_headers[1], recent_dates_headers[2], recent_dates_headers[3], 'Độ chính xác', 'Volume'])
df = df.sort_values(by='Độ chính xác', ascending=False)
df.index = range(1, len(df) + 1)


st.subheader("Bảng thông tin cổ phiếu")

def color_comparison(row):
    color = 'white'
    
    if row[recent_dates_headers[3]] > row[recent_dates_headers[2]]:
        color = '#00FF00'
    else:
        color = 'red'
        
    return ['color: white'] * (len(row) - 1) + [f'color: {color}']

styled_df = df.style.apply(color_comparison, axis=1, subset=[recent_dates_headers[2], recent_dates_headers[3]])

show_styled_df = styled_df
st.dataframe(show_styled_df.format({
    recent_dates_headers[0]: "{:.2f}",
    recent_dates_headers[1]: "{:.2f}",
    recent_dates_headers[2]: "{:.2f}",
    recent_dates_headers[3]: "{:.2f}",
    'Độ chính xác': "{:.3f}",
    'Volume': "{:.2f}"
}))

selected_stock = st.selectbox("Chọn mã cổ phiếu để xem chi tiết:", [""] + list(df['Mã cổ phiếu'].unique()))

def update_y_axis_range(trace, layout_update):
    if 'xaxis.range[0]' in layout_update or 'xaxis.range[1]' in layout_update:
        x_range = fig.layout.xaxis.range
        if x_range is not None:
            filtered_data = stock_data.loc[(stock_data.index >= x_range[0]) & (stock_data.index <= x_range[1])]
            y_min = filtered_data[['Low', 'MA_50', 'MA_100', 'MA_200']].min().min()
            y_max = filtered_data[['High', 'MA_50', 'MA_100', 'MA_200']].max().max()
            
            y_range = y_max - y_min
            
            y_mean = (y_min + y_max) / 2
            
            y_distance = max(y_mean - y_min, y_max - y_mean)
            
            y_min_new = y_mean - y_distance * 1.1
            y_max_new = y_mean + y_distance * 1.1
            
            fig.update_yaxes(range=[y_min_new, y_max_new])


if selected_stock:
    stock_data, accuracy = predicted_data[selected_stock]
    
    if stock_data.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã {selected_stock}. Vui lòng thử lại")
    else:
        st.subheader(f"Thông tin về {selected_stock}")
        stock_info = yf.Ticker(selected_stock).info
        st.write(f"**Tên công ty:** {stock_info['longName']}")
        st.write(f"**Ngành:** {stock_info['industry']}")
        st.write(f"**Vốn hóa:** {format_market_cap(stock_info['marketCap'])}")

        with st.expander("Xem dữ liệu"):
            st.dataframe(stock_data.round(2))

        st.subheader(f"Biểu đồ giá đóng cửa của {selected_stock} (bao gồm dự đoán 1 ngày)")
        
        ma_options = [50, 100, 200]
        line_visibility = {}
        for ma in ma_options:
            col_name = f"MA_{ma}"
            stock_data[col_name] = stock_data["Close"].rolling(window=ma, min_periods=1).mean()
            line_visibility[col_name] = st.checkbox(f"Hiển thị {col_name}", value=True)

        ma_colors = ['#E6C300', '#EC5840', '#5CD07F']

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Giá",
            increasing_line_color='green', 
            decreasing_line_color='red',
            visible=True,
        ))

        for i, ma in enumerate(ma_options):
            col_name = f"MA_{ma}"
            if line_visibility[col_name]:
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data[col_name],
                    mode='lines',
                    name=col_name,
                    line=dict(color=ma_colors[i], width=1)
                ))

        fig.update_layout(
            title='',
            xaxis_title="Ngày",
            yaxis_title="Giá (USD)",
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
                fixedrange=False,
                range=[(end_date - timedelta(days=365)).strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
            ),
            yaxis=dict(
                autorange=True,  
                fixedrange=False,
                anchor="x",
            ),
            dragmode='pan',  
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            width=1200,
            height=800,  
        )
        
        fig.layout.on_change(update_y_axis_range, 'xaxis.range')

        st.plotly_chart(fig, config={'editable': True}, use_container_width=True)
else:
    st.warning("Vui lòng chọn mã cổ phiếu để xem chi tiết.")
