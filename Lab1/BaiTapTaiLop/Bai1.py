# Bài 1. Giả sử có một tập dữ liệu giả định với các cột Diện tích nhà (m2) và Giá bán (triệu VND).
# 1. Cần xây dựng một mô hình hồi quy tuyến tính đơn giản để dự đoán Giá bán dựa trên
# Diện tích nhà.
# 2. Thêm nhiều (số lượng) dữ liệu và độ nhiễu để xem mô hình hoạt động như thế nào.
# 3. Thêm các biến đầu vào khác để nâng cao tính phức tạp của mô hình.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import pandas as pd

def DuLieuIt():
    # 1. Tạo tập dữ liệu giả định nhỏ
    # X: Diện tích nhà (m2), y: Giá bán (triệu VND)
    X_small = np.array([30, 45, 55, 65, 80, 100, 120]).reshape(-1, 1)
    y_small = np.array([900, 1300, 1600, 1900, 2300, 2900, 3500])

    # 2. Xây dựng và huấn luyện mô hình Hồi quy tuyến tính
    model_simple = LinearRegression()
    model_simple.fit(X_small, y_small)

    # 3. Dự đoán
    y_small_pred = model_simple.predict(X_small)

    # In kết quả
    print("--- YÊU CẦU 1: HỒI QUY TUYẾN TÍNH ĐƠN GIẢN ---")
    print(f"Hệ số góc (Trọng số w1): {model_simple.coef_[0]:.2f}")
    print(f"Hệ số chặn (Bias w0): {model_simple.intercept_:.2f}")

    # Trực quan hóa
    plt.figure(figsize=(8, 5))
    plt.scatter(X_small, y_small, color='blue', label='Dữ liệu thực tế')
    plt.plot(X_small, y_small_pred, color='red', label='Đường hồi quy')
    plt.title('Dự đoán Giá bán theo Diện tích (Dữ liệu ít)')
    plt.xlabel('Diện tích (m2)')
    plt.ylabel('Giá bán (Triệu VND)')
    plt.legend()
    plt.grid(True)
    plt.show()

def ThemDuLieu():
    # 1. Tạo 500 mẫu dữ liệu diện tích từ 30m2 đến 200m2
    np.random.seed(42) # Cố định seed để kết quả không đổi mỗi lần chạy
    n_samples = 500
    X_large = np.random.uniform(30, 200, n_samples).reshape(-1, 1)

    # print("X_Large = ",X_large)

    # Thêm độ nhiễu (noise) bằng phân phối chuẩn
    noise = np.random.normal(0, 300, n_samples) 

    # Giả định công thức: Giá = 25 * Diện tích + 100 + nhiễu
    y_large = 25 * X_large.flatten() + 100 + noise

    # print("y_Large = ",y_large)

    # print("X_Large.len = ",X_large.__len__())
    # print("Y_Large.len = ",y_large.__len__())

    # 2. Chia tập dữ liệu thành Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

    # 3. Xây dựng và huấn luyện mô hình
    model_noisy = LinearRegression()
    model_noisy.fit(X_train, y_train)

    # 4. Dự đoán và Đánh giá trên tập Test
    y_pred_noisy = model_noisy.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_noisy)
    r2 = r2_score(y_test, y_pred_noisy)

    print("\n--- YÊU CẦU 2: NHIỀU DỮ LIỆU & CÓ ĐỘ NHIỄU ---")
    print(f"Hệ số góc (Trọng số w1): {model_noisy.coef_[0]:.2f}")
    print(f"Hệ số chặn (Bias w0): {model_noisy.intercept_:.2f}")
    print(f"R-squared (R2): {r2:.4f} (Càng gần 1 càng tốt)")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # Trực quan hóa
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='lightgray', label='Dữ liệu Train', alpha=0.6)
    plt.scatter(X_test, y_test, color='blue', label='Dữ liệu Test', alpha=0.8)
    plt.plot(X_test, y_pred_noisy, color='red', linewidth=2, label='Đường dự đoán')
    plt.title('Dự đoán Giá bán với nhiều dữ liệu & Nhiễu')
    plt.xlabel('Diện tích (m2)')
    plt.ylabel('Giá bán (Triệu VND)')
    plt.legend()
    plt.grid(True)
    plt.show()

def DuLieuLon():
    n_samples = 500
    # 1. Tạo dữ liệu đa biến
    # Diện tích: 30 - 200 m2
    Area = np.random.uniform(30, 200, n_samples)
    # Số phòng ngủ: 1 - 5 phòng
    Bedrooms = np.random.randint(1, 6, n_samples)
    # Tuổi của nhà: 0 - 30 năm (nhà càng cũ giá càng giảm)
    Age = np.random.uniform(0, 30, n_samples)

    # Công thức giá nhà: 
    # Mỗi m2 tăng 25 triệu, mỗi phòng ngủ tăng 150 triệu, mỗi năm tuổi giảm 30 triệu + Nhiễu
    noise_multi = np.random.normal(0, 200, n_samples)
    y_multi = 25 * Area + 150 * Bedrooms - 30 * Age + 100 + noise_multi

    # Gộp thành DataFrame để dễ xử lý
    df = pd.DataFrame({'DienTich': Area, 'PhongNgu': Bedrooms, 'TuoiNha': Age, 'GiaBan': y_multi})
    X_multi = df[['DienTich', 'PhongNgu', 'TuoiNha']]
    y_multi_target = df['GiaBan']

    # 2. Chia tập Train/Test
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi_target, test_size=0.2, random_state=42)

    # 3. Huấn luyện mô hình đa biến
    model_multi = LinearRegression()
    model_multi.fit(X_train_m, y_train_m)

    # 4. Đánh giá
    y_pred_m = model_multi.predict(X_test_m)
    r2_m = r2_score(y_test_m, y_pred_m)

    print("\n--- YÊU CẦU 3: HỒI QUY TUYẾN TÍNH BỘI (NHIỀU BIẾN) ---")
    print(f"Hệ số góc của (Diện tích, Phòng ngủ, Tuổi nhà): {model_multi.coef_}")
    print(f"Hệ số chặn (Bias w0): {model_multi.intercept_}")
    print(f"R-squared (R2) mô hình đa biến: {r2_m:.4f}")

    # Dự đoán thử 1 căn nhà: 80m2, 2 phòng ngủ, nhà mới xây (1 năm)
    sample_house = pd.DataFrame({'DienTich': [80], 'PhongNgu': [2], 'TuoiNha': [1]})
    predicted_price = model_multi.predict(sample_house)
    print(f"Dự đoán nhà 80m2, 2 phòng ngủ, 1 năm tuổi: {predicted_price[0]:.2f} Triệu VND")

if __name__ == "__main__":
    print("Dữ liệu ít")
    DuLieuIt()
    print("Dữ liệu vừa")
    ThemDuLieu()
    print("Dữ liệu lớn")
    DuLieuLon()