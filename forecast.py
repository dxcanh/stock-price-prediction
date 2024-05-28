import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import yfinance as yf

def forecast_stock(stock, period="2y", interval='1d'):
    try:
        df = yf.download(stock, period=period, interval=interval)
        df = df.asfreq('D').ffill()
        ts = df[['Close']]

        # Tìm bộ tham số tối ưu (p, d, q) cho mô hình ARIMA
        model_autoARIMA = pm.auto_arima(ts, start_p=0, start_q=0,
                                      test='adf',       # kiểm tra tính dừng của dữ liệu
                                      max_p=3, max_q=3, # giá trị tối đa của p và q
                                      m=1,              # tần suất của chuỗi thời gian (hàng ngày = 1)
                                      d=None,           # để auto_arima tự tìm giá trị d tối ưu
                                      seasonal=False,   # không có tính mùa vụ
                                      start_P=0, 
                                      D=0, 
                                      trace=True,
                                      error_action='ignore',  
                                      suppress_warnings=True, 
                                      stepwise=True)

        # Huấn luyện mô hình ARIMA với tham số tối ưu
        model = ARIMA(ts, order=model_autoARIMA.order)
        model_fit = model.fit()

        # Dự đoán giá đóng cửa cho ngày tiếp theo
        forecast = model_fit.forecast(steps=1)

        # Tính toán độ chính xác của mô hình (MAE)
        mae = mean_absolute_error(ts, model_fit.predict())
        
        return {'predictions': forecast, 'accuracy': mae}

    except Exception as e:
        print(f"Lỗi khi dự đoán {stock}: {e}")
        return {'predictions': [], 'accuracy': None}
