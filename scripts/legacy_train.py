import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, classification_report
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PIPELINE PHÂN LOẠI HOA IRIS")
print("="*60)

# 1. ĐỌC DỮ LIỆU
print("\n1. ĐỌC DỮ LIỆU")
print("-" * 30)

data = pd.read_csv('data/Iris.csv')
print(f"Dữ liệu đã được tải thành công với {data.shape[0]} mẫu và {data.shape[1]} cột")
print(f"Các cột: {list(data.columns)}")

# 2. KIỂM TRA VÀ KHÁM PHÁ DỮ LIỆU
print("\n2. KIỂM TRA VÀ KHÁM PHÁ DỮ LIỆU")
print("-" * 30)

# Thông tin tổng quan
print("\nThông tin tổng quan về dữ liệu:")
print(data.info())

print("\nThống kê mô tả:")
print(data.describe())

# Kiểm tra missing values
print("\nKiểm tra missing values:")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("✓ Không có missing values trong dữ liệu")
else:
    print("⚠ Có missing values cần xử lý")

# Phân bố các lớp
print("\nPhân bố các lớp:")
print(data['Species'].value_counts())

# Visualize phân bố lớp
plt.figure(figsize=(15, 10))

# Biểu đồ phân bố lớp
plt.subplot(2, 3, 1)
data['Species'].value_counts().plot(kind='bar')
plt.title('Phân bố các lớp Species')
plt.xlabel('Species')
plt.ylabel('Số lượng')
plt.xticks(rotation=45)

# Pair plot cho các đặc trưng
plt.subplot(2, 3, 2)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+2)
    for species in data['Species'].unique():
        subset = data[data['Species'] == species]
        plt.hist(subset[feature], alpha=0.7, label=species)
    plt.title(f'Phân bố {feature}')
    plt.xlabel(feature)
    plt.ylabel('Tần suất')
    plt.legend()

plt.tight_layout()
plt.show()

# 3. TIỀN XỬ LÝ DỮ LIỆU
print("\n3. TIỀN XỬ LÝ DỮ LIỆU")
print("-" * 30)

# Loại bỏ cột Id không cần thiết
data_processed = data.drop('Id', axis=1)
print("✓ Đã loại bỏ cột 'Id'")

# Tách features và target
X = data_processed.drop('Species', axis=1)
y = data_processed['Species']

print(f"✓ Tách features (X): {X.shape}")
print(f"✓ Tách target (y): {y.shape}")

# Mã hóa nhãn (Label Encoding)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"✓ Mã hóa nhãn:")
for i, species in enumerate(label_encoder.classes_):
    print(f"  {species} -> {i}")

# 4. CHIA DỮ LIỆU THÀNH TẬP HUẤN LUYỆN VÀ KIỂM TRA
print("\n4. CHIA DỮ LIỆU")
print("-" * 30)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"✓ Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"✓ Tập kiểm tra: {X_test.shape[0]} mẫu")
print(f"✓ Tỷ lệ chia: {X_train.shape[0]/X.shape[0]:.1%} - {X_test.shape[0]/X.shape[0]:.1%}")

# 5. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
print("\n5. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH")
print("-" * 30)

# Tạo pipeline với chuẩn hóa dữ liệu
def create_pipeline(model):
    """Tạo pipeline với chuẩn hóa dữ liệu và mô hình"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Khởi tạo các mô hình
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Huấn luyện và đánh giá từng mô hình
results = {}
trained_pipelines = {}

for name, model in models.items():
    print(f"\n--- Huấn luyện {name} ---")
    
    # Tạo pipeline
    pipeline = create_pipeline(model)
    
    # Huấn luyện
    pipeline.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = pipeline.predict(X_test)
    
    # Lưu pipeline và kết quả
    trained_pipelines[name] = pipeline
    results[name] = {
        'predictions': y_pred,
        'pipeline': pipeline
    }
    
    print(f"✓ Hoàn thành huấn luyện {name}")

# 6. ĐÁNH GIÁ MÔ HÌNH
print("\n6. ĐÁNH GIÁ MÔ HÌNH")
print("-" * 30)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    """Đánh giá mô hình với các chỉ số"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n--- KẾT QUẢ {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Đánh giá từng mô hình
evaluation_results = {}
for name, result in results.items():
    evaluation_results[name] = evaluate_model(y_test, result['predictions'], name)

# 7. MA TRẬN NHẦM LẪN VÀ BÁO CÁO PHÂN LOẠI
print("\n7. MA TRẬN NHẦM LẪN VÀ BÁO CÁO PHÂN LOẠI")
print("-" * 30)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(15, 6))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 2, i+1)
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, result['predictions'])
    
    # Vẽ heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Ma trận nhầm lẫn - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# In báo cáo phân loại chi tiết
for name, result in results.items():
    print(f"\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT: {name} ---")
    print(classification_report(y_test, result['predictions'], 
                              target_names=label_encoder.classes_))

# 8. SO SÁNH KẾT QUẢ
print("\n8. SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
print("-" * 30)

# Tạo DataFrame để so sánh
comparison_df = pd.DataFrame(evaluation_results).T
print(comparison_df.round(4))

# Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 6))

# Biểu đồ so sánh các chỉ số
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(metrics))
width = 0.35

for i, (name, scores) in enumerate(evaluation_results.items()):
    values = [scores[metric] for metric in metrics]
    plt.bar(x + i*width, values, width, label=name, alpha=0.8)

plt.xlabel('Chỉ số đánh giá')
plt.ylabel('Giá trị')
plt.title('So sánh hiệu suất các mô hình')
plt.xticks(x + width/2, metrics)
plt.legend()
plt.ylim(0, 1)

# Thêm giá trị lên các cột
for i, (name, scores) in enumerate(evaluation_results.items()):
    values = [scores[metric] for metric in metrics]
    for j, v in enumerate(values):
        plt.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 9. CHỌN MÔ HÌNH TỐT NHẤT VÀ LƯU PIPELINE
print("\n9. CHỌN MÔ HÌNH TỐT NHẤT VÀ LƯU PIPELINE")
print("-" * 30)

# Tìm mô hình tốt nhất dựa trên accuracy
best_model_name = max(evaluation_results.keys(), 
                     key=lambda x: evaluation_results[x]['accuracy'])
best_pipeline = trained_pipelines[best_model_name]

print(f"✓ Mô hình tốt nhất: {best_model_name}")
print(f"✓ Accuracy: {evaluation_results[best_model_name]['accuracy']:.4f}")

# Lưu pipeline tốt nhất
joblib.dump(best_pipeline, 'best_iris_pipeline.pkl')
print("✓ Đã lưu pipeline tốt nhất vào file 'best_iris_pipeline.pkl'")

# Lưu label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✓ Đã lưu label encoder vào file 'label_encoder.pkl'")

# Lưu thông tin mô hình
model_info = {
    'best_model': best_model_name,
    'accuracy': evaluation_results[best_model_name]['accuracy'],
    'features': list(X.columns),
    'classes': list(label_encoder.classes_),
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

joblib.dump(model_info, 'model_info.pkl')
print("✓ Đã lưu thông tin mô hình vào file 'model_info.pkl'")

# 10. DEMO SỬ DỤNG PIPELINE ĐÃ LƯU
print("\n10. DEMO SỬ DỤNG PIPELINE ĐÃ LƯU")
print("-" * 30)

# Tải lại pipeline
loaded_pipeline = joblib.load('best_iris_pipeline.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

# Lấy một vài mẫu từ tập test để demo
demo_samples = X_test.iloc[:5]
demo_predictions = loaded_pipeline.predict(demo_samples)

print("Demo dự đoán với pipeline đã lưu:")
print("-" * 40)
for i, (idx, sample) in enumerate(demo_samples.iterrows()):
    predicted_class = loaded_label_encoder.inverse_transform([demo_predictions[i]])[0]
    actual_class = loaded_label_encoder.inverse_transform([y_test[i]])[0]
    
    print(f"Mẫu {i+1}:")
    print(f"  Đặc trưng: {sample.values}")
    print(f"  Dự đoán: {predicted_class}")
    print(f"  Thực tế: {actual_class}")
    print(f"  Chính xác: {'✓' if predicted_class == actual_class else '✗'}")
    print()

print("="*60)
print("HOÀN THÀNH PIPELINE!")
print("="*60)
print(f"✓ Mô hình tốt nhất: {best_model_name}")
print(f"✓ Accuracy: {evaluation_results[best_model_name]['accuracy']:.4f}")
print("✓ Files đã lưu:")
print("  - best_iris_pipeline.pkl (pipeline hoàn chỉnh)")
print("  - label_encoder.pkl (encoder cho nhãn)")
print("  - model_info.pkl (thông tin mô hình)")
print("="*60)



