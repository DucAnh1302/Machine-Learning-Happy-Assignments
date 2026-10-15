# Bài 2. Dự đoán khả năng mắc bệnh dựa trên chỉ số BMI
# Giả sử có một tập dữ liệu với các thông tin sau:
# - BMI: Chỉ số khối cơ thể của bệnh nhân (Body Mass Index).
# - Mắc bệnh: Nhãn mục tiêu là 1 nếu bệnh nhân mắc bệnh và 0 nếu không mắc bệnh.
# Yêu cầu:
# 1. - Tạo dữ liệu mẫu giả lập với đặc trưng BMI và nhãn Mắc bệnh dựa trên mối liên hệ logic
# (BMI cao có thể dẫn đến khả năng mắc bệnh cao hơn).
# 2. - Sử dụng thư viện phù hợp (HD: scikit-learn) để xây dựng một mô hình hồi quy logistic.
# 3. - Huấn luyện mô hình trên tập dữ liệu mẫu.
# 4. - Dự đoán xác suất và nhãn mắc bệnh cho một tập dữ liệu kiểm tra.
# 5. - Đánh giá hiệu quả của mô hình thông qua độ chính xác và ma trận nhầm lẫn.
# 6. - Trực quan hóa dữ liệu và đường hồi quy logistic.

# Bài 2: Dự đoán khả năng mắc bệnh dựa trên chỉ số BMI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Tạo dữ liệu mẫu giả lập (BMI từ 15 đến 35)
np.random.seed(42)
n_samples = 150
BMI = np.random.uniform(15, 35, n_samples)

# Xác suất mắc bệnh tăng theo BMI (sử dụng hàm sigmoid giả lập)
# BMI càng cao thì khả năng mắc bệnh (y = 1) càng lớn
prob = 1 / (1 + np.exp(-(0.4 * (BMI - 25))))
y = np.random.binomial(1, prob)

# Định dạng lại dữ liệu đầu vào cho sklearn (X cần là mảng 2 chiều)
X = BMI.reshape(-1, 1)

# Chia dữ liệu: 80% huấn luyện và 20% kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2 & 3. Xây dựng và huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Dự đoán xác suất và nhãn trên tập kiểm tra
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Xác suất mắc bệnh (nhãn 1)

# 5. Đánh giá hiệu quả mô hình
accuracy = accuracy_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
print("Ma trận nhầm lẫn (Confusion Matrix):")
print(conf_matrix)
print("Báo cáo phân loại (Classification Report):")
print(class_report)

# 6. Trực quan hóa dữ liệu và đường cong hồi quy logistic
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Dữ liệu thực tế (Mắc bệnh: 1, Khỏe mạnh: 0)')

# Tạo trục X mịn để vẽ đường cong sigmoid mượt mà
X_curve = np.linspace(15, 35, 300).reshape(-1, 1)
y_curve_prob = model.predict_proba(X_curve)[:, 1]

plt.plot(X_curve, y_curve_prob, color='red', linewidth=2, label='Đường hồi quy Logistic (Xác suất)')
plt.axhline(0.5, color='gray', linestyle='--', label='Ranh giới quyết định (0.5)')
plt.title('Hồi quy Logistic: Dự đoán khả năng mắc bệnh theo BMI')
plt.xlabel('Chỉ số BMI')
plt.ylabel('Xác suất mắc bệnh')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()