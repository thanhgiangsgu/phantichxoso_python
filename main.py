import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDateEdit, QLineEdit, QPushButton
from PyQt5.QtCore import QDate, Qt, QDateTime
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.decomposition import PCA
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



import csv


class LotteryPrediction(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set the main window size
        self.setGeometry(100, 100, 550, 400)
        self.setWindowTitle("Phân tích và dự đoán xổ số")

        # Create a label for the date picker
        date_label = QLabel("Chọn ngày:", self)
        date_label.move(20, 10)

        # Create a QDateEdit widget for selecting the date
        self.date_picker = QDateEdit(self)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setMaximumDate(QDate.currentDate())
        self.date_picker.setGeometry(20, 40, 200, 30)

        # Create a label and line edit for entering the number
        number_label = QLabel("Enter a number:", self)
        number_label.move(260, 10)
        self.number_edit = QLineEdit(self)
        self.number_edit.setGeometry(260, 40, 100, 30)
        self.number_edit.setText("30")

        # Create a button to submit the data
        submit_button = QPushButton("Submit", self)
        submit_button.setGeometry(400, 40, 100, 30)
        submit_button.clicked.connect(self.submit_data)

        submit_button = QPushButton("Biểu đồ chẳn lẻ", self)
        submit_button.setGeometry(200, 90, 100, 30)
        submit_button.clicked.connect(self.submit_parity_chart)

        submit_button = QPushButton("Số xuất hiện nhiều nhất", self)
        submit_button.setGeometry(30, 90, 150, 30)
        submit_button.clicked.connect(self.submit_appearing_numbers)

        # Create a label and line edit for entering the number
        number_label = QLabel("Số dự đoán:", self)
        number_label.move(320, 90)
        self.prediction_number = QLineEdit(self)
        self.prediction_number.setGeometry(400, 90, 50, 30)
        self.prediction_number.setText("12345")



        submit_button = QPushButton("Linea Regression ", self)
        submit_button.setGeometry(30, 150, 150, 30)
        submit_button.clicked.connect(self.Linear_Regression)

        submit_button = QPushButton("Decision Tree ", self)
        submit_button.setGeometry(200, 150, 150, 30)
        submit_button.clicked.connect(self.Decision_Tree)

        submit_button = QPushButton("Random Forest ", self)
        submit_button.setGeometry(370, 150, 150, 30)
        submit_button.clicked.connect(self.Random_Forest)

        submit_button = QPushButton("ARIMA ", self)
        submit_button.setGeometry(30, 210, 150, 30)
        submit_button.clicked.connect(self.ARIMA)

        submit_button = QPushButton("KMeans ", self)
        submit_button.setGeometry(200, 210, 150, 30)
        submit_button.clicked.connect(self.KMeans)

        submit_button = QPushButton("Apriori  ", self)
        submit_button.setGeometry(370, 210, 150, 30)
        submit_button.clicked.connect(self.Apriori)

        submit_button = QPushButton("PCA  ", self)
        submit_button.setGeometry(30, 270, 150, 30)
        submit_button.clicked.connect(self.PCA)

        submit_button = QPushButton("Neural   ", self)
        submit_button.setGeometry(200, 270, 150, 30)
        submit_button.clicked.connect(self.Neural)

        submit_button = QPushButton("DecisionTreeClassifier   ", self)
        submit_button.setGeometry(370, 270, 150, 30)
        submit_button.clicked.connect(self.DecisionTreeClassifier)

        submit_button = QPushButton("Gradient Boosting   ", self)
        submit_button.setGeometry(30, 330, 150, 30)
        submit_button.clicked.connect(self.Gradient_Boosting)

        submit_button = QPushButton("SVM", self)
        submit_button.setGeometry(200, 330, 150, 30)
        submit_button.clicked.connect(self.SVM)


    #

    def SVM(self):
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])

        X = data[['Đặc biệt', 'Giải nhất']]
        y = data['Ngày']

        # Chuyển đổi Ngày sang số nguyên
        y = pd.to_datetime(y, format='%d-%m-%Y').apply(lambda x: x.toordinal())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Khởi tạo mô hình SVM
        svm = SVC(kernel='linear', C=1)

        # Huấn luyện mô hình trên tập huấn luyện
        svm.fit(X_train, y_train)

        # Dự đoán giá trị trên tập kiểm tra
        y_pred = svm.predict(X_test)

        # Tính toán độ chính xác trên tập kiểm tra
        accuracy = accuracy_score(y_test, y_pred)
        print("Áp dụng thuật toán SVM . !")
        print("Độ chính xác trên tập kiểm tra:", accuracy)

    def Gradient_Boosting(self):
        # Đọc dữ liệu từ tập tin csv
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])

        # Tiền xử lý dữ liệu
        X = data[['Đặc biệt', 'Giải nhất']]
        y = pd.to_datetime(data['Ngày'], format='%d-%m-%Y').apply(lambda x: x.weekday())

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Xây dựng mô hình Gradient Boosting
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


        # Huấn luyện mô hình trên tập huấn luyện
        model.fit(X_train, y_train)

        # Dự đoán nhãn trên tập kiểm tra
        y_pred = model.predict(X_test)

        # Tính độ chính xác của mô hình trên tập kiểm tra
        accuracy = accuracy_score(y_test, y_pred)
        print("Áp dụng thuật toán Gradient Boosting")
        print('Độ chính xác của mô hình trên tập kiểm tra: {:.2f}%'.format(accuracy * 100))

    def DecisionTreeClassifier(self):
        # Đọc dữ liệu từ tập tin csv
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])

        # Tiền xử lý dữ liệu
        X = data[['Đặc biệt', 'Giải nhất']]
        y = pd.to_datetime(data['Ngày'], format='%d-%m-%Y').apply(lambda x: x.weekday())


        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Xây dựng mô hình Cây quyết định
        model = DecisionTreeClassifier()

        # Huấn luyện mô hình trên tập huấn luyện
        model.fit(X_train, y_train)

        # Đánh giá độ chính xác của mô hình trên tập kiểm tra
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Độ chính xác của mô hình trên tập kiểm tra: {:.2f}%'.format(accuracy * 100))


    def Neural(self):
        # Đọc dữ liệu từ tập tin csv
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])
        # Tiền xử lý dữ liệu
        X = data[['Đặc biệt', 'Giải nhất']]
        y = pd.to_datetime(data['Ngày'], format='%d-%m-%Y').apply(lambda x: x.weekday())
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # Xây dựng mô hình
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # Huấn luyện mô hình
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
        # Đánh giá mô hình trên tập kiểm tra
        test_loss = model.evaluate(X_test, y_test)
        # Dự đoán giá trị trên tập kiểm tra
        y_pred = model.predict(X_test)
        print(y_pred)

    def PCA(self):
        # Chuẩn bị dữ liệu
        # data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])
        X = data[['Đặc biệt', 'Giải nhất']]
        y = data['Ngày']
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Áp dụng PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X_scaled)

        # Hiển thị kết quả
        principalDf = pd.DataFrame(data=principalComponents, columns=['PC 1', 'PC 2'])
        finalDf = pd.concat([principalDf, y], axis=1)
        print(finalDf)

        plt.scatter(finalDf['PC 1'], finalDf['PC 2'])
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('Scatter Plot of Data on Two Principal Components')
        plt.show()


    def Apriori(self):
        # Load dữ liệu
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'])
        print(data)

        # Tạo bảng dữ liệu dạng one-hot encoding
        data_encoded = pd.get_dummies(data, columns=['Đặc biệt', 'Giải nhất'])
        data_encoded = data_encoded.set_index('Ngày')
        # Áp dụng thuật toán Apriori
        frequent_itemsets = apriori(data_encoded, min_support=0.001 , use_colnames=True)

        # Tìm các luật kết hợp
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

        # In kết quả
        print(rules)
    def KMeans(self):
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        X = data[['Đặc biệt', 'Giải nhất']]

        # Áp dụng thuật toán KMeans
        kmeans = KMeans(n_clusters=5, random_state=0)
        kmeans.fit(X)

        # Vẽ biểu đồ các cluster
        plt.scatter(X['Đặc biệt'], X['Giải nhất'], c=kmeans.labels_)
        plt.xlabel('Đặc biệt')
        plt.ylabel('Giải nhất')
        plt.show()

    def Random_Forest(self):
        # Chuẩn bị dữ liệu
        input_number = int(self.prediction_number.text())
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        X = data[['Đặc biệt']]
        y = data[['Giải nhất']]

        # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Xây dựng mô hình Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X_train, y_train)

        # Đánh giá mô hình
        score = rf.score(X_test, y_test)

        # Dự đoán
        prediction = rf.predict([[input_number]])

        print("Score: ", score)
        print("Prediction: ", prediction)

    def Decision_Tree(self):
        input_number = int(self.prediction_number.text())
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        X = data[['Đặc biệt']]
        y = data['Giải nhất']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        score = r2_score(y_test, y_pred)
        print("Score: ", score)
        prediction = dt.predict([[input_number]])
        print("Prediction: ", prediction)

    def Linear_Regression(self):

        # Chuẩn bị dữ liệu
        input_number = int(self.prediction_number.text())
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        X = data[['Đặc biệt']]
        X.columns = ['DB']  # Gán tên cho cột Đặc biệt
        y = data[['Giải nhất']]

        # Xây dựng mô hình Linear Regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        score = lr.score(X_test, y_test)

        # Dự đoán
        prediction = lr.predict([[input_number]])

        print("Score: ", score)
        print("Prediction: ", prediction)

    def ARIMA(self):
        print(123)
        # Đọc dữ liệu từ file CSV và chuyển cột Ngày thành index
        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')

        # Tạo 2 series riêng biệt cho Giải đặc biệt và Giải nhất
        data_db = data['Đặc biệt']
        data_gn = data['Giải nhất']
        print(data_db)
        print(data_gn)

        # Áp dụng mô hình ARIMA cho Giải đặc biệt
        model_db = ARIMA(data_db, order=(2, 1, 2))  # Tham số order là (p, d, q)
        model_db_fit = model_db.fit()
        prediction_db = model_db_fit.forecast(steps=7)  # Dự đoán 7 giá trị tiếp theo

        # Áp dụng mô hình ARIMA cho Giải nhất
        model_gn = ARIMA(data_gn, order=(2, 1, 2))
        model_gn_fit = model_gn.fit()
        prediction_gn = model_gn_fit.forecast(steps=7)

        # In kết quả dự đoán
        print("Dự đoán Giải đặc biệt trong 7 ngày tới:\n", prediction_db)
        print("\nDự đoán Giải nhất trong 7 ngày tới:\n", prediction_gn)
    def submit_appearing_numbers(self):

        data = pd.read_csv('XSMB.csv', parse_dates=['Ngày'], index_col='Ngày')
        special_prize = data['Đặc biệt'].astype(str).str[-2:].value_counts().sort_values(ascending=False).head(10)
        first_prize = data['Giải nhất'].astype(str).str[-2:].value_counts().sort_values(ascending=False).head(10)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        special_prize.plot(kind='bar', ax=ax[0])
        ax[0].set_title('10 số xuất hiện nhiều nhất của Giải đặc biệt')

        first_prize.plot(kind='bar', ax=ax[1])
        ax[1].set_title('10 số xuất hiện nhiều nhất của Giải nhất')

        plt.tight_layout()
        plt.show()

    def submit_parity_chart(self):
        # Đọc dữ liệu từ file CSV vào DataFrame
        data = pd.read_csv('XSMB.csv')

        # Lấy ra cột "Giải đặc biệt" và "Giải nhất"
        giai_db = data['Đặc biệt']
        giai_nhat = data['Giải nhất']

        # Tạo hàm đếm số lượng số chẵn và số lẻ
        def count_even_odd(series):
            even = sum(series % 2 == 0)
            odd = sum(series % 2 != 0)
            return even, odd

        # Đếm số lượng số chẵn và số lẻ của "Giải đặc biệt" và "Giải nhất"
        giai_db_even, giai_db_odd = count_even_odd(giai_db)
        giai_nhat_even, giai_nhat_odd = count_even_odd(giai_nhat)

        # Tạo biểu đồ số chẵn và số lẻ của "Giải đặc biệt" và "Giải nhất"
        labels = ['Even', 'Odd']
        giai_db_values = [giai_db_even, giai_db_odd]
        giai_nhat_values = [giai_nhat_even, giai_nhat_odd]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.pie(giai_db_values, labels=labels, autopct='%1.1f%%')
        ax1.set_title('Giải đặc biệt')
        ax2.pie(giai_nhat_values, labels=labels, autopct='%1.1f%%')
        ax2.set_title('Giải nhất')

        plt.show()
    def submit_data(self):
        # Lấy giá trị ngày t QDateEdit
        selected_date = self.date_picker.date()
        # Lấy giá trị trong input
        input_number = int(self.number_edit.text())

        # Chuyển đổi QdateTime thành datetime
        date_time = QDateTime(selected_date)
        selected_datetime = date_time.toPyDateTime()

        # Khai báo biến broser
        browser = webdriver.Chrome(executable_path="./chromedriver")
        data = []
        # Read utl
        idx = 0
        while True:
            url = 'https://www.thantai.net/so-ket-qua'
            browser.get(url)

            # Set date
            end = browser.find_element(By.ID, "end")
            end.clear()
            end.send_keys("{}-{}-{}".format(selected_datetime.day, selected_datetime.month, selected_datetime.year))

            btn = browser.find_element(By.XPATH, "/html/body/div[3]/main/div/form/div[2]/div/button[9]")
            btn.click()
            # Lấy thông tin ngày
            result = browser.find_elements(By.CLASS_NAME, "d-inline-block")
            # with open('date_list.txt', 'w') as f:
            dateList = []
            for row in result:
                dateList.append(row.text)
            with open("date_list.txt", "a") as f:
                for date in dateList[2:]:
                    f.write(date + "\n")
                    idx += 1
            # Lấy thông tin kết quả xổ số của Đặc biệt và Giải nhất
            result = browser.find_elements(By.CLASS_NAME, "font-weight-bold.col-12.d-block.p-1.m-0")
            for row in result:
                data.append(row.text)
            # Tách các phần tử ở vị trí lẻ và ghi vào file data2_odd.txt
            with open("dataG1.txt", "w") as f:
                for i in range(len(data)):
                    if i % 2 == 1:  # Nếu vị trí là lẻ
                        f.write(str(data[i]) + "\n")

            # Tách các phần tử ở vị trí chẵn và ghi vào file data2_even.txt
            with open("dataDB.txt", "w") as f:
                for i in range(len(data)):
                    if i % 2 == 0:  # Nếu vị trí là chẵn
                        f.write(str(data[i]) + "\n")

            selected_datetime -= timedelta(days=300)
            if idx >= input_number:
                break

        with open("date_list.txt", "r") as f:
            date_list = f.read().splitlines()

        with open("dataDB.txt", "r") as f:
            data_db = f.read().splitlines()

        with open("dataG1.txt", "r") as f:
            data_g1 = f.read().splitlines()

        data = []
        for i in range(len(date_list)):
            row = [date_list[i], data_db[i], data_g1[i]]
            data.append(row)
        print(data)

        df = pd.DataFrame(data, columns=['Ngày', 'Đặc biệt', 'Giải nhất'])
        df.to_csv("XSMB.csv", index=False)

        # browser.close()


# Create an instance of the LotteryPrediction class and show the GUI
app = QApplication(sys.argv)
lottery_prediction = LotteryPrediction()
lottery_prediction.show()
sys.exit(app.exec_())
