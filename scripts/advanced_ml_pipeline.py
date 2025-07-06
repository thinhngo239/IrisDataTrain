import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import modules đã tạo
from data_validation import DataValidator, IRIS_SCHEMA, validate_iris_data
from feature_engineering import FeatureEngineer, CustomFeatureCreator
from advanced_eda import AdvancedEDA

class AdvancedMLPipeline:
    """Pipeline ML nâng cao cho cả classification và regression"""
    
    def __init__(self):
        self.classification_models = {}
        self.regression_models = {}
        self.classification_results = {}
        self.regression_results = {}
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, df):
        """Chuẩn bị dữ liệu cho cả classification và regression"""
        print("=" * 60)
        print("CHUẨN BỊ DỮ LIỆU")
        print("=" * 60)
        
        # Loại bỏ cột Id nếu có
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        # Tạo dữ liệu cho classification
        X_classification = df.drop('Species', axis=1)
        y_classification = df['Species']
        
        # Tạo dữ liệu cho regression (dự đoán SepalLengthCm từ các features khác)
        X_regression = df.drop(['SepalLengthCm', 'Species'], axis=1)
        y_regression = df['SepalLengthCm']
        
        print(f"✓ Classification data: {X_classification.shape[0]} samples, {X_classification.shape[1]} features")
        print(f"✓ Regression data: {X_regression.shape[0]} samples, {X_regression.shape[1]} features")
        print(f"✓ Classification target: {y_classification.nunique()} classes")
        print(f"✓ Regression target: {y_regression.min():.2f} - {y_regression.max():.2f}")
        
        return X_classification, y_classification, X_regression, y_regression
    
    def validate_data(self, df):
        """Validate dữ liệu"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        
        results = validate_iris_data(df)
        
        if results['is_valid']:
            print("✅ Data validation: PASS")
        else:
            print("❌ Data validation: FAIL")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        return results
    
    def perform_eda(self, df):
        """Thực hiện EDA"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Loại bỏ cột Id nếu có
        if 'Id' in df.columns:
            df_eda = df.drop('Id', axis=1)
        else:
            df_eda = df.copy()
        
        eda = AdvancedEDA(df_eda, target_column='Species')
        eda.basic_info()
        eda.statistical_summary()
        eda.correlation_analysis()
        eda.outlier_detection()
        
        # Tạo insights
        insights = eda.generate_insights()
        
        return insights
    
    def engineer_features(self, X_train, X_test, y_train, task_type='classification'):
        """Feature engineering"""
        print(f"\n" + "=" * 60)
        print(f"FEATURE ENGINEERING - {task_type.upper()}")
        print("=" * 60)
        
        fe = FeatureEngineer(task_type=task_type)
        
        # Tạo custom features
        print("1. Tạo custom features...")
        X_train_custom = fe.create_interaction_features(X_train)
        X_test_custom = fe.create_interaction_features(X_test)
        
        # Feature selection
        print("2. Feature selection...")
        X_train_selected, selector = fe.select_features_importance(X_train_custom, y_train, threshold=0.01)
        
        # Áp dụng selector cho test set
        selected_features = X_train_selected.columns.tolist()
        X_test_selected = X_test_custom[selected_features]
        
        # Scaling
        print("3. Feature scaling...")
        X_train_scaled, X_test_scaled, scaler = fe.scale_features(X_train_selected, X_test_selected, method='standard')
        
        print(f"✓ Original features: {X_train.shape[1]}")
        print(f"✓ After custom features: {X_train_custom.shape[1]}")
        print(f"✓ After selection: {X_train_selected.shape[1]}")
        print(f"✓ Selected features: {selected_features}")
        
        return X_train_scaled, X_test_scaled, selector, scaler
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Huấn luyện các mô hình classification"""
        print("\n" + "=" * 60)
        print("TRAINING CLASSIFICATION MODELS")
        print("=" * 60)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Định nghĩa models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Huấn luyện
            model.fit(X_train, y_train_encoded)
            
            # Dự đoán
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Tính metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted')
            recall = recall_score(y_test_encoded, y_pred, average='weighted')
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
            }
            
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ Precision: {precision:.4f}")
            print(f"✓ Recall: {recall:.4f}")
            print(f"✓ F1-score: {f1:.4f}")
        
        self.classification_models = {name: result['model'] for name, result in results.items()}
        self.classification_results = results
        
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Huấn luyện các mô hình regression"""
        print("\n" + "=" * 60)
        print("TRAINING REGRESSION MODELS")
        print("=" * 60)
        
        # Định nghĩa models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Huấn luyện
            model.fit(X_train, y_train)
            
            # Dự đoán
            y_pred = model.predict(X_test)
            
            # Tính metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'rmse': rmse,
                'predictions': y_pred
            }
            
            print(f"✓ MSE: {mse:.4f}")
            print(f"✓ MAE: {mae:.4f}")
            print(f"✓ R² Score: {r2:.4f}")
            print(f"✓ RMSE: {rmse:.4f}")
        
        self.regression_models = {name: result['model'] for name, result in results.items()}
        self.regression_results = results
        
        return results
    
    def evaluate_models(self, y_test_classification, y_test_regression):
        """Đánh giá và so sánh models"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION & COMPARISON")
        print("=" * 60)
        
        # So sánh classification models
        print("\n📊 CLASSIFICATION MODELS COMPARISON")
        print("-" * 50)
        
        classification_comparison = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            for name, results in self.classification_results.items()
        }).T
        
        print(classification_comparison.round(4))
        
        # Tìm best classification model
        best_classification_model = classification_comparison['Accuracy'].idxmax()
        print(f"\n🏆 Best Classification Model: {best_classification_model}")
        
        # So sánh regression models
        print("\n📈 REGRESSION MODELS COMPARISON")
        print("-" * 50)
        
        regression_comparison = pd.DataFrame({
            name: {
                'MSE': results['mse'],
                'MAE': results['mae'],
                'R² Score': results['r2_score'],
                'RMSE': results['rmse']
            }
            for name, results in self.regression_results.items()
        }).T
        
        print(regression_comparison.round(4))
        
        # Tìm best regression model
        best_regression_model = regression_comparison['R² Score'].idxmax()
        print(f"\n🏆 Best Regression Model: {best_regression_model}")
        
        return best_classification_model, best_regression_model
    
    def visualize_results(self, y_test_classification, y_test_regression):
        """Visualize kết quả"""
        print("\n" + "=" * 60)
        print("VISUALIZATION")
        print("=" * 60)
        
        # Classification results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion matrices
        for i, (name, results) in enumerate(self.classification_results.items()):
            if i < 2:  # Chỉ hiển thị 2 models đầu tiên
                ax = axes[0, i]
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Model comparison
        ax = axes[1, 0]
        classification_metrics = pd.DataFrame({
            name: [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
            for name, results in self.classification_results.items()
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        classification_metrics.plot(kind='bar', ax=ax)
        ax.set_title('Classification Models Comparison')
        ax.set_ylabel('Score')
        ax.legend(title='Models')
        ax.tick_params(axis='x', rotation=45)
        
        # Regression results
        ax = axes[1, 1]
        
        # Scatter plot: Actual vs Predicted cho best regression model
        best_reg_name = max(self.regression_results.keys(), 
                           key=lambda x: self.regression_results[x]['r2_score'])
        best_reg_results = self.regression_results[best_reg_name]
        
        ax.scatter(y_test_regression, best_reg_results['predictions'], alpha=0.7)
        ax.plot([y_test_regression.min(), y_test_regression.max()], 
                [y_test_regression.min(), y_test_regression.max()], 'r--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs Predicted - {best_reg_name}')
        
        plt.tight_layout()
        plt.show()
        
        # Regression metrics comparison
        plt.figure(figsize=(12, 6))
        
        regression_metrics = pd.DataFrame({
            name: [results['mse'], results['mae'], results['r2_score']]
            for name, results in self.regression_results.items()
        }, index=['MSE', 'MAE', 'R² Score'])
        
        regression_metrics.plot(kind='bar', figsize=(10, 6))
        plt.title('Regression Models Comparison')
        plt.ylabel('Score')
        plt.legend(title='Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_models(self, best_classification_model, best_regression_model):
        """Lưu models"""
        print("\n" + "=" * 60)
        print("SAVING MODELS")
        print("=" * 60)
        
        # Lưu classification model
        classification_model = self.classification_models[best_classification_model]
        joblib.dump(classification_model, 'best_classification_model.pkl')
        print(f"✓ Saved classification model: {best_classification_model}")
        
        # Lưu regression model
        regression_model = self.regression_models[best_regression_model]
        joblib.dump(regression_model, 'best_regression_model.pkl')
        print(f"✓ Saved regression model: {best_regression_model}")
        
        # Lưu label encoder
        joblib.dump(self.label_encoder, 'label_encoder_advanced.pkl')
        print("✓ Saved label encoder")
        
        # Lưu model info
        model_info = {
            'best_classification_model': best_classification_model,
            'best_regression_model': best_regression_model,
            'classification_results': {
                name: {k: v for k, v in results.items() if k != 'model'}
                for name, results in self.classification_results.items()
            },
            'regression_results': {
                name: {k: v for k, v in results.items() if k != 'model'}
                for name, results in self.regression_results.items()
            }
        }
        
        joblib.dump(model_info, 'advanced_model_info.pkl')
        print("✓ Saved model info")
    
    def run_full_pipeline(self, df):
        """Chạy toàn bộ pipeline"""
        print("=" * 80)
        print("ADVANCED MACHINE LEARNING PIPELINE")
        print("=" * 80)
        
        # 1. Data Validation
        validation_results = self.validate_data(df)
        
        # 2. EDA
        eda_insights = self.perform_eda(df)
        
        # 3. Data Preparation
        X_classification, y_classification, X_regression, y_regression = self.prepare_data(df)
        
        # 4. Train/Test Split
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_classification, y_classification, test_size=0.2, random_state=42, stratify=y_classification
        )
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_regression, y_regression, test_size=0.2, random_state=42
        )
        
        # 5. Feature Engineering
        X_train_cls_eng, X_test_cls_eng, cls_selector, cls_scaler = self.engineer_features(
            X_train_cls, X_test_cls, y_train_cls, task_type='classification'
        )
        
        X_train_reg_eng, X_test_reg_eng, reg_selector, reg_scaler = self.engineer_features(
            X_train_reg, X_test_reg, y_train_reg, task_type='regression'
        )
        
        # 6. Model Training
        classification_results = self.train_classification_models(
            X_train_cls_eng, X_test_cls_eng, y_train_cls, y_test_cls
        )
        
        regression_results = self.train_regression_models(
            X_train_reg_eng, X_test_reg_eng, y_train_reg, y_test_reg
        )
        
        # 7. Model Evaluation
        best_cls_model, best_reg_model = self.evaluate_models(y_test_cls, y_test_reg)
        
        # 8. Visualization
        self.visualize_results(y_test_cls, y_test_reg)
        
        # 9. Save Models
        self.save_models(best_cls_model, best_reg_model)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"🏆 Best Classification Model: {best_cls_model}")
        print(f"🏆 Best Regression Model: {best_reg_model}")
        print("=" * 80)
        
        return {
            'validation_results': validation_results,
            'eda_insights': eda_insights,
            'best_classification_model': best_cls_model,
            'best_regression_model': best_reg_model,
            'classification_results': classification_results,
            'regression_results': regression_results
        }

def main():
    """Demo advanced ML pipeline"""
    try:
        # Đọc dữ liệu
        df = pd.read_csv('data/Iris.csv')
        print(f"✓ Loaded data: {df.shape}")
        
        # Tạo và chạy pipeline
        pipeline = AdvancedMLPipeline()
        results = pipeline.run_full_pipeline(df)
        
        print("\n🎉 Pipeline completed successfully!")
        
    except FileNotFoundError:
        print("❌ File data/Iris.csv not found")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 