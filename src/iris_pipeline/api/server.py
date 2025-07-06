from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(
    title="Iris ML Pipeline API",
    description="API để predict loài hoa Iris và các thuộc tính với Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models cho request/response
class IrisFeatures(BaseModel):
    """Model cho input features của Iris"""
    sepal_length: float = Field(..., ge=0, le=10, description="Chiều dài đài hoa (cm)")
    sepal_width: float = Field(..., ge=0, le=10, description="Chiều rộng đài hoa (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="Chiều dài cánh hoa (cm)")
    petal_width: float = Field(..., ge=0, le=10, description="Chiều rộng cánh hoa (cm)")
    
    @validator('*', pre=True)
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('Tất cả giá trị phải >= 0')
        return v

class RegressionFeatures(BaseModel):
    """Model cho regression input (dự đoán SepalLength)"""
    sepal_width: float = Field(..., ge=0, le=10, description="Chiều rộng đài hoa (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="Chiều dài cánh hoa (cm)")
    petal_width: float = Field(..., ge=0, le=10, description="Chiều rộng cánh hoa (cm)")
    
    @validator('*', pre=True)
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('Tất cả giá trị phải >= 0')
        return v

class ClassificationResponse(BaseModel):
    """Response cho classification"""
    predicted_species: str
    confidence: float
    probabilities: Dict[str, float]
    input_features: IrisFeatures
    model_used: str
    prediction_time: str

class RegressionResponse(BaseModel):
    """Response cho regression"""
    predicted_sepal_length: float
    input_features: RegressionFeatures
    model_used: str
    prediction_time: str
    confidence_interval: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """Response cho health check"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str

class ModelInfo(BaseModel):
    """Thông tin về models"""
    classification_model: str
    regression_model: str
    classification_accuracy: float
    regression_r2_score: float
    training_date: str
    model_files: List[str]

# Global variables để lưu models
models = {
    'classification': None,
    'regression': None,
    'label_encoder': None,
    'model_info': None,
    'scaler_classification': None,
    'scaler_regression': None
}

def load_models():
    """Load tất cả models cần thiết"""
    try:
        # Load classification model
        if os.path.exists('models/best_classification_model.pkl'):
            models['classification'] = joblib.load('models/best_classification_model.pkl')
            logger.info("✓ Loaded classification model")
        else:
            logger.warning("⚠️ Classification model not found")
        
        # Load regression model  
        if os.path.exists('models/best_regression_model.pkl'):
            models['regression'] = joblib.load('models/best_regression_model.pkl')
            logger.info("✓ Loaded regression model")
        else:
            logger.warning("⚠️ Regression model not found")
        
        # Load label encoder
        if os.path.exists('models/label_encoder_advanced.pkl'):
            models['label_encoder'] = joblib.load('models/label_encoder_advanced.pkl')
            logger.info("✓ Loaded label encoder")
        else:
            logger.warning("⚠️ Label encoder not found")
        
        # Load model info
        if os.path.exists('models/advanced_model_info.pkl'):
            models['model_info'] = joblib.load('models/advanced_model_info.pkl')
            logger.info("✓ Loaded model info")
        else:
            logger.warning("⚠️ Model info not found")
        
        # Load scalers
        if os.path.exists('models/scaler_classification.pkl'):
            models['scaler_classification'] = joblib.load('models/scaler_classification.pkl')
            logger.info("✓ Loaded classification scaler")
        else:
            logger.warning("⚠️ Classification scaler not found")
            
        if os.path.exists('models/scaler_regression.pkl'):
            models['scaler_regression'] = joblib.load('models/scaler_regression.pkl')
            logger.info("✓ Loaded regression scaler")
        else:
            logger.warning("⚠️ Regression scaler not found")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise

def create_custom_features(features_dict):
    """Tạo custom features giống như trong training"""
    # Tạo DataFrame từ dict
    df = pd.DataFrame([features_dict])
    
    # Tạo ratios
    df['sepal_ratio'] = df.iloc[:, 0] / (df.iloc[:, 1] + 1e-8)
    df['petal_ratio'] = df.iloc[:, 2] / (df.iloc[:, 3] + 1e-8)
    
    # Tạo areas
    df['sepal_area'] = df.iloc[:, 0] * df.iloc[:, 1]
    df['petal_area'] = df.iloc[:, 2] * df.iloc[:, 3]
    
    # Tạo interactions
    df['total_length'] = df.iloc[:, 0] + df.iloc[:, 2]
    df['total_width'] = df.iloc[:, 1] + df.iloc[:, 3]
    
    return df

def check_models_loaded():
    """Dependency để kiểm tra models đã được load"""
    if models['classification'] is None or models['regression'] is None:
        raise HTTPException(
            status_code=503, 
            detail="Models chưa được load. Vui lòng huấn luyện models trước."
        )
    return True

def prepare_features_for_classification(features: IrisFeatures) -> np.ndarray:
    """Chuẩn bị features cho classification"""
    # Tạo dict từ features
    features_dict = {
        'SepalLengthCm': features.sepal_length,
        'SepalWidthCm': features.sepal_width,
        'PetalLengthCm': features.petal_length,
        'PetalWidthCm': features.petal_width
    }
    
    # Apply feature engineering
    df_engineered = create_custom_features(features_dict)
    
    # Scale features nếu có scaler
    if models['scaler_classification'] is not None:
        feature_vector = models['scaler_classification'].transform(df_engineered)
    else:
        feature_vector = df_engineered.values
    
    return feature_vector

def prepare_features_for_regression(features: RegressionFeatures) -> np.ndarray:
    """Chuẩn bị features cho regression"""
    # Regression model dự đoán SepalLength từ SepalWidth, PetalLength, PetalWidth
    # Nhưng trong training, model chỉ dùng 3 features gốc, không có feature engineering
    feature_vector = np.array([[
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Scale features nếu có scaler
    if models['scaler_regression'] is not None:
        feature_vector = models['scaler_regression'].transform(feature_vector)
    
    return feature_vector

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Iris ML Pipeline API",
        "version": "1.0.0",
        "description": "API để predict loài hoa Iris và thuộc tính",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_status = {
        "classification": models['classification'] is not None,
        "regression": models['regression'] is not None,
        "label_encoder": models['label_encoder'] is not None,
        "model_info": models['model_info'] is not None
    }
    
    overall_status = "healthy" if all(models_status.values()) else "partial"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        models_loaded=models_status,
        api_version="1.0.0"
    )

@app.get("/models/info", response_model=ModelInfo)
async def get_model_info(check_models: bool = Depends(check_models_loaded)):
    """Lấy thông tin về models"""
    if models['model_info'] is None:
        raise HTTPException(status_code=404, detail="Model info not found")
    
    info = models['model_info']
    
    # Tìm các file models
    model_files = []
    for filename in ['best_classification_model.pkl', 'best_regression_model.pkl', 
                     'label_encoder_advanced.pkl', 'advanced_model_info.pkl']:
        if os.path.exists(f'models/{filename}'):
            model_files.append(f'models/{filename}')
    
    # Lấy accuracy từ classification results
    classification_accuracy = 0.0
    if 'classification_results' in info:
        best_cls_model = info['best_classification_model']
        if best_cls_model in info['classification_results']:
            classification_accuracy = info['classification_results'][best_cls_model]['accuracy']
    
    # Lấy R2 score từ regression results
    regression_r2 = 0.0
    if 'regression_results' in info:
        best_reg_model = info['best_regression_model']
        if best_reg_model in info['regression_results']:
            regression_r2 = info['regression_results'][best_reg_model]['r2_score']
    
    return ModelInfo(
        classification_model=info.get('best_classification_model', 'Unknown'),
        regression_model=info.get('best_regression_model', 'Unknown'),
        classification_accuracy=classification_accuracy,
        regression_r2_score=regression_r2,
        training_date=datetime.now().strftime("%Y-%m-%d"),
        model_files=model_files
    )

@app.post("/predict/classification", response_model=ClassificationResponse)
async def predict_species(
    features: IrisFeatures,
    check_models: bool = Depends(check_models_loaded)
):
    """Predict loài hoa Iris"""
    try:
        # Chuẩn bị features
        feature_vector = prepare_features_for_classification(features)
        
        # Predict
        classifier = models['classification']
        assert classifier is not None  # Guaranteed by check_models_loaded
        prediction = classifier.predict(feature_vector)[0]
        probabilities = classifier.predict_proba(feature_vector)[0]
        
        # Chuyển đổi prediction về tên loài
        if models['label_encoder'] is not None:
            species_name = models['label_encoder'].inverse_transform([prediction])[0]
            species_classes = models['label_encoder'].classes_
        else:
            # Fallback nếu không có label encoder
            species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
            species_name = species_map.get(prediction, f'Class_{prediction}')
            species_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        # Tạo probability dict
        prob_dict = {
            species_classes[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Lấy model name
        model_name = models['model_info']['best_classification_model'] if models['model_info'] else 'Unknown'
        
        return ClassificationResponse(
            predicted_species=species_name,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            input_features=features,
            model_used=model_name,
            prediction_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Classification prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/regression", response_model=RegressionResponse)
async def predict_sepal_length(
    features: RegressionFeatures,
    check_models: bool = Depends(check_models_loaded)
):
    """Predict chiều dài đài hoa (SepalLength)"""
    try:
        # Chuẩn bị features
        feature_vector = prepare_features_for_regression(features)
        
        # Predict
        regressor = models['regression']
        assert regressor is not None  # Guaranteed by check_models_loaded
        prediction = regressor.predict(feature_vector)[0]
        
        # Lấy model name
        model_name = models['model_info']['best_regression_model'] if models['model_info'] else 'Unknown'
        
        # Estimate confidence interval (simplified)
        confidence_interval = None
        if hasattr(regressor, 'predict') and hasattr(regressor, 'estimators_'):
            # Nếu là Random Forest, có thể tính standard deviation
            try:
                predictions = [tree.predict(feature_vector)[0] for tree in getattr(regressor, 'estimators_', [])]
                std = np.std(predictions)
                confidence_interval = {
                    "lower_bound": float(prediction - 1.96 * std),
                    "upper_bound": float(prediction + 1.96 * std),
                    "std_dev": float(std)
                }
            except:
                pass
        
        return RegressionResponse(
            predicted_sepal_length=float(prediction),
            input_features=features,
            model_used=model_name,
            prediction_time=datetime.now().isoformat(),
            confidence_interval=confidence_interval
        )
        
    except Exception as e:
        logger.error(f"Regression prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch/classification")
async def predict_batch_classification(
    features_list: List[IrisFeatures],
    check_models: bool = Depends(check_models_loaded)
):
    """Batch prediction cho classification"""
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size không được vượt quá 100")
    
    results = []
    for features in features_list:
        try:
            result = await predict_species(features, check_models)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "input": features})
    
    return {"predictions": results, "count": len(results)}

@app.post("/predict/batch/regression")
async def predict_batch_regression(
    features_list: List[RegressionFeatures],
    check_models: bool = Depends(check_models_loaded)
):
    """Batch prediction cho regression"""
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size không được vượt quá 100")
    
    results = []
    for features in features_list:
        try:
            result = await predict_sepal_length(features, check_models)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "input": features})
    
    return {"predictions": results, "count": len(results)}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models khi start server"""
    logger.info("🚀 Starting Iris ML Pipeline API...")
    try:
        load_models()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        logger.warning("⚠️ API will start but predictions may fail")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup khi shutdown"""
    logger.info("🛑 Shutting down Iris ML Pipeline API...")

if __name__ == "__main__":
    import uvicorn
    
    # Chạy server
    print("🚀 Starting Iris ML Pipeline API Server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("ℹ️ Model Info: http://localhost:8000/models/info")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 