import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Iris ML Pipeline",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B6B;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"

# Utility functions
def check_api_health():
    """Kiểm tra API có hoạt động không"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_model_info():
    """Lấy thông tin model"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def predict_classification(features):
    """Gọi API để dự đoán phân loại"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/classification",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def predict_regression(features):
    """Gọi API để dự đoán hồi quy"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/regression",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def predict_batch(data):
    """Gọi API để dự đoán hàng loạt"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Iris ML Pipeline</h1>', unsafe_allow_html=True)
    
    # Kiểm tra API health
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("API đang hoạt động bình thường")
        if health_data:
            with st.expander("Thông tin API"):
                st.json(health_data)
    else:
        st.error("Không thể kết nối đến API. Vui lòng kiểm tra server.")
        st.info("Hướng dẫn khởi động API: `uvicorn iris_pipeline.api.server:app --reload`")
        return
    
    # Sidebar
    st.sidebar.title("Tùy chọn")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Dự đoán đơn lẻ", "Dự đoán hồi quy", "Dự đoán hàng loạt"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Dự đoán phân loại loài hoa</h2>', unsafe_allow_html=True)
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Nhập thông số hoa iris")
            
            sepal_length = st.number_input(
                "Chiều dài đài hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=5.1,
                step=0.1,
                help="Chiều dài của đài hoa tính bằng cm"
            )
            
            sepal_width = st.number_input(
                "Chiều rộng đài hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=3.5,
                step=0.1,
                help="Chiều rộng của đài hoa tính bằng cm"
            )
            
            petal_length = st.number_input(
                "Chiều dài cánh hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=1.4,
                step=0.1,
                help="Chiều dài của cánh hoa tính bằng cm"
            )
            
            petal_width = st.number_input(
                "Chiều rộng cánh hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=0.2,
                step=0.1,
                help="Chiều rộng của cánh hoa tính bằng cm"
            )
            
            submit_cls = st.button("Dự đoán loài hoa", type="primary")
        
        with col2:
            st.markdown("### Kết quả dự đoán")
            
            if submit_cls:
                features = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width
                }
                
                with st.spinner("Đang dự đoán..."):
                    result = predict_classification(features)
                
                if result:
                    # Hiển thị kết quả
                    predicted_species = result['predicted_species']
                    confidence = result['confidence']
                    
                    # Species color map
                    colors = {
                        'Iris-setosa': '#FF6B6B',
                        'Iris-versicolor': '#4ECDC4', 
                        'Iris-virginica': '#45B7D1'
                    }
                    
                    # Main result
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>Loài dự đoán: <span style="color: {colors.get(predicted_species, '#000')}">{predicted_species}</span></h3>
                        <h4>Độ tin cậy: {confidence:.2%}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    probabilities = result['probabilities']
                    prob_df = pd.DataFrame({
                        'Species': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Species',
                        y='Probability',
                        title='Xác suất dự đoán cho từng loài',
                        color='Species',
                        color_discrete_map=colors
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail info
                    with st.expander("Chi tiết kết quả"):
                        st.json(result)
                else:
                    st.error("Không thể dự đoán. Vui lòng kiểm tra API.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Dự đoán chiều dài đài hoa</h2>', unsafe_allow_html=True)
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Nhập thông số để dự đoán")
            
            sepal_width_reg = st.number_input(
                "Chiều rộng đài hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=3.5,
                step=0.1,
                key="reg_sepal_width"
            )
            
            petal_length_reg = st.number_input(
                "Chiều dài cánh hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=1.4,
                step=0.1,
                key="reg_petal_length"
            )
            
            petal_width_reg = st.number_input(
                "Chiều rộng cánh hoa (cm)",
                min_value=0.0,
                max_value=20.0,
                value=0.2,
                step=0.1,
                key="reg_petal_width"
            )
            
            submit_reg = st.button("Dự đoán chiều dài", type="primary")
        
        with col2:
            st.markdown("### Kết quả dự đoán")
            
            if submit_reg:
                features = {
                    "sepal_width": sepal_width_reg,
                    "petal_length": petal_length_reg,
                    "petal_width": petal_width_reg
                }
                
                with st.spinner("Đang dự đoán..."):
                    result = predict_regression(features)
                
                if result:
                    predicted_length = result['predicted_sepal_length']
                    
                    # Main result
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>Chiều dài đài hoa dự đoán: {predicted_length:.2f} cm</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence interval if available
                    if result.get('confidence_interval'):
                        ci = result['confidence_interval']
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Point estimate
                        fig.add_trace(go.Scatter(
                            x=[predicted_length],
                            y=[0],
                            mode='markers',
                            marker=dict(size=15, color='red'),
                            name='Dự đoán'
                        ))
                        
                        # Confidence interval
                        if 'lower_bound' in ci and 'upper_bound' in ci:
                            fig.add_trace(go.Scatter(
                                x=[ci['lower_bound'], ci['upper_bound']],
                                y=[0, 0],
                                mode='lines',
                                line=dict(color='blue', width=5),
                                name='Khoảng tin cậy 95%'
                            ))
                        
                        fig.update_layout(
                            title="Dự đoán với khoảng tin cậy",
                            xaxis_title="Chiều dài đài hoa (cm)",
                            yaxis=dict(showticklabels=False),
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail info
                    with st.expander("Chi tiết kết quả"):
                        st.json(result)
                else:
                    st.error("Không thể dự đoán. Vui lòng kiểm tra API.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Dự đoán hàng loạt</h2>', unsafe_allow_html=True)
        
        # File upload
        st.markdown("### Tải lên file CSV")
        uploaded_file = st.file_uploader(
            "Chọn file CSV chứa dữ liệu hoa iris",
            type=['csv'],
            help="File CSV cần có các cột: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.markdown("### Dữ liệu đã tải lên")
                st.dataframe(df.head())
                
                # Validate columns
                required_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Thiếu các cột: {', '.join(missing_columns)}")
                else:
                    # Predict button
                    if st.button("Dự đoán hàng loạt", type="primary"):
                        with st.spinner("Đang xử lý dự đoán hàng loạt..."):
                            # Prepare data for API
                            data_for_api = df[required_columns].to_dict('records')  # type: ignore
                            
                            # Call batch prediction API
                            results = predict_batch(data_for_api)
                    
                    # Display results
                    if results:
                        results_df = pd.DataFrame(results)
                        
                        st.markdown("### Kết quả dự đoán")
                        st.dataframe(results_df)
                        
                        # Summary
                        st.markdown("### Tóm tắt kết quả")
                        summary = results_df['Predicted_Species'].value_counts()
                        
                        fig = px.pie(
                            values=summary.values,
                            names=summary.index,
                            title="Phân bố dự đoán loài"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Tải xuống kết quả",
                            data=csv,
                            file_name=f"iris_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {str(e)}")
        
        # Sample data download
        st.markdown("### File mẫu")
        sample_data = pd.DataFrame({
            'SepalLengthCm': [5.1, 6.0, 6.3, 4.9, 5.8],
            'SepalWidthCm': [3.5, 2.9, 3.3, 3.0, 2.7],
            'PetalLengthCm': [1.4, 4.5, 6.0, 1.4, 5.1],
            'PetalWidthCm': [0.2, 1.5, 2.5, 0.2, 1.9]
        })
        
        st.dataframe(sample_data)
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="Tải xuống file mẫu",
            data=csv_sample,
            file_name="iris_sample.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 