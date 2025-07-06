import pandas as pd
import numpy as np
import joblib

def load_model():
    """Tải mô hình đã lưu"""
    try:
        pipeline = joblib.load('best_iris_pipeline.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        model_info = joblib.load('model_info.pkl')
        return pipeline, label_encoder, model_info
    except FileNotFoundError:
        print("❌ Không tìm thấy file mô hình. Vui lòng chạy train.py trước.")
        return None, None, None

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Dự đoán loài hoa iris dựa trên các đặc trưng
    
    Args:
        sepal_length: Chiều dài đài hoa (cm)
        sepal_width: Chiều rộng đài hoa (cm)
        petal_length: Chiều dài cánh hoa (cm)
        petal_width: Chiều rộng cánh hoa (cm)
    
    Returns:
        dict: Chứa kết quả dự đoán và xác suất
    """
    # Tải mô hình
    pipeline, label_encoder, model_info = load_model()
    
    if pipeline is None or label_encoder is None:
        return None
    
    # Tạo dữ liệu đầu vào
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Dự đoán
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0]
    
    # Chuyển đổi nhãn về tên loài
    species_name = label_encoder.inverse_transform([prediction])[0]
    
    # Tạo dictionary xác suất cho từng loài
    species_probabilities = {}
    for i, species in enumerate(label_encoder.classes_):
        species_probabilities[species] = prediction_proba[i]
    
    return {
        'predicted_species': species_name,
        'confidence': max(prediction_proba),
        'probabilities': species_probabilities,
        'input_features': {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
    }

def main():
    """Chương trình demo"""
    print("="*60)
    print("DEMO DỰ ĐOÁN LOÀI HOA IRIS")
    print("="*60)
    
    # Hiển thị thông tin mô hình
    _, _, model_info = load_model()
    if model_info:
        print(f"✓ Mô hình: {model_info['best_model']}")
        print(f"✓ Độ chính xác: {model_info['accuracy']:.4f}")
        print(f"✓ Số mẫu huấn luyện: {model_info['training_samples']}")
        print(f"✓ Các loài có thể dự đoán: {', '.join(model_info['classes'])}")
    
    print("\n" + "="*60)
    print("DEMO VỚI CÁC MẪU THỰC TẾ")
    print("="*60)
    
    # Một số mẫu demo
    demo_samples = [
        {"name": "Mẫu Iris-setosa", "features": [5.1, 3.5, 1.4, 0.2]},
        {"name": "Mẫu Iris-versicolor", "features": [6.0, 2.9, 4.5, 1.5]},
        {"name": "Mẫu Iris-virginica", "features": [6.3, 3.3, 6.0, 2.5]},
        {"name": "Mẫu biên giới", "features": [5.9, 3.0, 4.2, 1.5]}
    ]
    
    for sample in demo_samples:
        print(f"\n--- {sample['name']} ---")
        sepal_length, sepal_width, petal_length, petal_width = sample['features']
        
        result = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)
        
        if result:
            print(f"Đặc trưng: SepalL={sepal_length}, SepalW={sepal_width}, PetalL={petal_length}, PetalW={petal_width}")
            print(f"Dự đoán: {result['predicted_species']}")
            print(f"Độ tin cậy: {result['confidence']:.3f}")
            print("Xác suất từng loài:")
            for species, prob in result['probabilities'].items():
                print(f"  {species}: {prob:.3f}")
    
    print("\n" + "="*60)
    print("DEMO NHẬP TỪ NGƯỜI DÙNG")
    print("="*60)
    
    try:
        while True:
            print("\nNhập đặc trưng hoa iris (hoặc 'quit' để thoát):")
            
            # Nhập dữ liệu
            try:
                sepal_length = float(input("Chiều dài đài hoa (cm): "))
                sepal_width = float(input("Chiều rộng đài hoa (cm): "))
                petal_length = float(input("Chiều dài cánh hoa (cm): "))
                petal_width = float(input("Chiều rộng cánh hoa (cm): "))
            except ValueError:
                print(" Vui lòng nhập số hợp lệ!")
                continue
            
            # Kiểm tra giá trị hợp lệ
            if any(x <= 0 for x in [sepal_length, sepal_width, petal_length, petal_width]):
                print(" Tất cả giá trị phải lớn hơn 0!")
                continue
            
            # Dự đoán
            result = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)
            
            if result:
                print(f"\n KẾT QUẢ DỰ ĐOÁN:")
                print(f"   Loài dự đoán: {result['predicted_species']}")
                print(f"   Độ tin cậy: {result['confidence']:.3f}")
                print(f"   Xác suất chi tiết:")
                for species, prob in result['probabilities'].items():
                    confidence_bar = "█" * int(prob * 20)
                    print(f"     {species:<15}: {prob:.3f} {confidence_bar}")
            
            # Hỏi có tiếp tục không
            continue_choice = input("\nBạn có muốn tiếp tục? (y/n): ").lower()
            if continue_choice not in ['y', 'yes']:
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Tạm biệt!")
    
    print("\n" + "="*60)
    print("CẢM ÔN BẠN ĐÃ SỬ DỤNG!")
    print("="*60)

if __name__ == "__main__":
    main() 