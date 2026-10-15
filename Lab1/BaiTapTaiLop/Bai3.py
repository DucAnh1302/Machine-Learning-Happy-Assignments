# Bài 3: Hồi quy tuyến tính đa biến trên bộ dữ liệu California Housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Đọc dữ liệu từ file csv
data_path = Path(__file__).parent / 'dataset_Bai3' / 'california_housing_train.csv'
df = pd.read_csv(data_path)

# Hiển thị cấu trúc dữ liệu cơ bản
print("--- Thông tin bộ dữ liệu California Housing ---")
print(df)

# 2. Lựa chọn đặc trưng (Features) và Biến mục tiêu (Target)
# Chúng ta dùng các đặc trưng trực quan: median_income, housing_median_age, và total_rooms để dự đoán median_house_value
features = ['median_income', 'housing_median_age', 'total_rooms']
X = df[features]
y = df['median_house_value']

# 3. Chia tập dữ liệu thành Train và Test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính đa biến
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# 5. Dự đoán trên tập kiểm tra
y_pred = reg_model.predict(X_test)

# Đánh giá mô hình bằng MSE và R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Kết quả huấn luyện mô hình ---")
print(f"Hệ số chặn (Intercept): {reg_model.intercept_:.2f}")
for feature, coef in zip(features, reg_model.coef_):
    print(f"Hệ số hồi quy cho {feature}: {coef:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score (Độ tương thích của mô hình): {r2:.4f} ({r2*100:.2f}%)")

# 6. Trực quan hóa kết quả: So sánh Giá trị thực tế vs Giá trị dự đoán
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='purple', label='Dữ liệu dự đoán')

# Vẽ đường phân giác hoàn hảo (nơi thực tế bằng dự đoán)
lims = [0, max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Đường dự đoán lý tưởng')

plt.title('So sánh Giá trị nhà thực tế và Giá trị nhà dự đoán (California)', fontsize=14)
plt.xlabel('Giá trị nhà thực tế ($)', fontsize=12)
plt.ylabel('Giá trị nhà dự đoán ($)', fontsize=12)
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()