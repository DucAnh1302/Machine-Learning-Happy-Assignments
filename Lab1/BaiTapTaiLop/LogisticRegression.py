# Thư viện
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Tải dữ liệu hoa Iris trong thư viện sklearn
iris = load_iris()

X = iris['data'] # Các đặc trưng (features)
y = iris['target'] # Các nhãn (target)

# Chọn ra hai loại hoa (0 và 1) để thực hiện phân loại nhị phân
X = X[y != 2] # Loại bỏ lớp 2
y = y[y != 2] # Chỉ giữ lại lớp 0 và 1

# Chia dữ liệu (minh họa chia 80% huấn luyên và 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Logistic Regression
model = LogisticRegression()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# In kết quả
print(f"Độ chính xác (Accuracy): {accuracy}")
print("Ma trận nhầm lẫn (Confusion Matrix):")
print(conf_matrix)
print("Báo cáo phân loại (Classification Report):")
print(class_report)