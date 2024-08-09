# Import necessary libraries (Nhập các thư viện cần thiết)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from datetime import datetime
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Main function (Hàm chính)
def main():
    # Cities (Các thành phố)
    city_names = ["basel", "budapest", "de_bilt", "dresden", "dusseldorf", "heathrow", "kassel", "ljubljana", "maastricht", "malmo", "montelimar", "muenchen", "oslo", "perpignan", "roma", "sonnblick", "stockholm", "tours"]
    
    # Group data by month for each city
    for name in city_names:
        globals()[name] = data_group_by_month(globals()[name])
    
    try:
        selected_city = input("\033[96mSelect a city:\033[0m ").lower().strip()
        if selected_city in city_names:
            selected_city = selected_city.upper()
            selected_thermodynamic_variables = ["cloud_cover","humidity","pressure","global_radiation","precipitation","sunshine","temp_mean","temp_min","temp_max"]
            print(selected_thermodynamic_variables)
            variable = input("\033[96mSelect one thermodynamic variable above:\033[0m ")
            try:
                if variable in selected_thermodynamic_variables:
                    pass
            except ValueError:
                print("\033[91mSelect one thermodynamic variable:\033[0m", selected_thermodynamic_variables)
            prophet_data, column = prepare_prophet_data(globals()[selected_city.lower()], f"{selected_city}_{variable}")
            target_date = input("\033[96mDate (YYYYMMDD):\033[0m ")
            try:
                if datetime.strptime(target_date , "%Y%m%d") == True:
                    pass
            except ValueError:
                print("Input date with format: \033[91mYYYYMMDD\033[0m!")
            model, forecast, train, test = training(prophet_data, column, target_date)
            evaluate_model(model, forecast, train, test)
        else:
            raise ValueError
    except ValueError:
        print("\033[91mSelect one city in this list:\033[0m\n", city_names)
        return
    
# Preparing and cleaning data: Chuẩn bị và làm sạch dữ liệu
    # Load data: CSV file: "weather_prediction_dataset.csv" (Nhập file dữ liệu)
data = pd.read_csv("weather_prediction_dataset.csv")
print("\033[96mQuick look:\033[0m\n", data.head(5))
print("\033[96mData information:\033[0m")
data.info()
print("\033[96mDescribe data:\033[0m\n", data.describe())

    # Clean data (làm sạch dữ liệu)
data = data.dropna().drop_duplicates()  # Removing missing values and duplicates (Loại bỏ giá trị bị thiếu và các bản sao)

# Split data by city and group by month (Chia dữ liệu theo thành phố và nhóm theo tháng)
basel = data.iloc[:, 0:11]
budapest = pd.merge(data.iloc[:, :2], data.iloc[:, 11:19], left_index=True, right_index=True)
de_bilt = pd.merge(data.iloc[:, :2], data.iloc[:, 19:30], left_index=True, right_index=True)
dresden = pd.merge(data.iloc[:, :2], data.iloc[:, 30:40], left_index=True, right_index=True)
dusseldorf = pd.merge(data.iloc[:, :2], data.iloc[:, 40:51], left_index=True, right_index=True)
heathrow = pd.merge(data.iloc[:, :2], data.iloc[:, 51:60], left_index=True, right_index=True)
kassel = pd.merge(data.iloc[:, :2], data.iloc[:, 60:70], left_index=True, right_index=True)
ljubljana = pd.merge(data.iloc[:, :2], data.iloc[:, 70:80], left_index=True, right_index=True)
maastricht = pd.merge(data.iloc[:, :2], data.iloc[:, 80:91], left_index=True, right_index=True)
malmo = pd.merge(data.iloc[:, :2], data.iloc[:, 91:96], left_index=True, right_index=True)
montelimar = pd.merge(data.iloc[:, :2], data.iloc[:, 96:104], left_index=True, right_index=True)
muenchen = pd.merge(data.iloc[:, :2], data.iloc[:, 104:115], left_index=True, right_index=True)
oslo = pd.merge(data.iloc[:, :2], data.iloc[:, 115:126], left_index=True, right_index=True)
perpignan = pd.merge(data.iloc[:, :2], data.iloc[:, 126:134], left_index=True, right_index=True)
roma = pd.merge(data.iloc[:, :2], data.iloc[:, 134:143], left_index=True, right_index=True)
sonnblick = pd.merge(data.iloc[:, :2], data.iloc[:, 143:150], left_index=True, right_index=True)
stockholm = pd.merge(data.iloc[:, :2], data.iloc[:, 150:157], left_index=True, right_index=True)
tours = pd.merge(data.iloc[:, :2], data.iloc[:, 157:166], left_index=True, right_index=True)

def data_group_by_month(name):
  for _, group in name.groupby("MONTH"):
    summary = group.describe()
    Q1 = summary.loc["25%"]
    Q3 = summary.loc["75%"]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for columns in group.columns[1:]:
        group = group[(group[columns] >= lower_bound[columns]) & (group[columns] <= upper_bound[columns])]
  return name

    # Prepare data for Prophet model (Chuẩn bị dữ liệu cho mô hình Prophet)
def prepare_prophet_data(name, column):
    name["DATE"] = pd.to_datetime(name["DATE"], format = "%Y%m%d")
    name = name.sort_values("DATE")
    prophet_data = pd.merge(name["DATE"], name[column], left_index = True, right_index = True)
    prophet_data = prophet_data.reset_index().rename(columns={"DATE": "ds", column: "y"})
    return prophet_data, column

def training(prophet_data ,column, target_date):
    # Split train and test set: 80/20 (Chia dữ liệu thành tập huấn luyện và tập thử nghiệm: 80/20)
    train_size = int(len(basel) * 0.8)
    train = prophet_data.iloc[:train_size]
    test = prophet_data.iloc[train_size:]
 
    # Apply Prophet model: daily seasonality (Áp dụng mô hình Prophet với chu kỳ: hàng ngày)
    model = Prophet(daily_seasonality=True)
    model.fit(train)

    # Predict (Dự đoán)
    target_date = pd.to_datetime(target_date, format='%Y%m%d')
    end_train_date = pd.to_datetime(train.iloc[-1,1], format='%Y%m%d')
    period = target_date - end_train_date
    future = model.make_future_dataframe(periods=period.days)
    forecast = model.predict(future)

    # Graph (Đồ thị)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(prophet_data['ds'][:train_size], prophet_data['y'][:train_size], label='Train', color = "green")
    plt.plot(prophet_data['ds'][train_size:], prophet_data['y'][train_size:], label='Test', color = "orange")
    model.plot(forecast, ax=ax)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.xlabel("DATE")
    plt.title(column)
    plt.legend(loc = "upper left")
    plt.xticks(rotation = 45)
    plt.legend()
    plt.show()

    return model, forecast, train, test

# Model evaluation (Đánh giá mô hình)
def evaluate_model(model,forecast,train, test):
    # MSE
    mse_train = mean_squared_error(train["y"], model.predict(train)["yhat"])
    print("\033[96mMean Squared Error (Train):\033[0m", mse_train)
    forecast_test = forecast.iloc[-len(test):]["yhat"]
    mse_test = mean_squared_error(test["y"], forecast_test)
    print("\033[96mMean Squared Error (Test):\033[0m", mse_test)

if __name__ == "__main__":
    main()
