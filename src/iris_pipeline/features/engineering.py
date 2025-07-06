import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class CustomFeatureCreator(BaseEstimator, TransformerMixin):
    """Tạo custom features cho dữ liệu Iris"""
    
    def __init__(self, create_ratios=True, create_areas=True, create_interactions=True):
        self.create_ratios = create_ratios
        self.create_areas = create_areas
        self.create_interactions = create_interactions
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Giả sử các cột theo thứ tự: SepalLength, SepalWidth, PetalLength, PetalWidth
        if X.shape[1] >= 4:
            col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
            if hasattr(X, 'columns'):
                col_names = X.columns.tolist()[:4]
            
            # Tạo ratios
            if self.create_ratios:
                X_new[f'{col_names[0]}_to_{col_names[1]}_ratio'] = X.iloc[:, 0] / (X.iloc[:, 1] + 1e-8)
                X_new[f'{col_names[2]}_to_{col_names[3]}_ratio'] = X.iloc[:, 2] / (X.iloc[:, 3] + 1e-8)
                X_new['sepal_to_petal_length_ratio'] = X.iloc[:, 0] / (X.iloc[:, 2] + 1e-8)
                X_new['sepal_to_petal_width_ratio'] = X.iloc[:, 1] / (X.iloc[:, 3] + 1e-8)
            
            # Tạo areas (diện tích)
            if self.create_areas:
                X_new['sepal_area'] = X.iloc[:, 0] * X.iloc[:, 1]
                X_new['petal_area'] = X.iloc[:, 2] * X.iloc[:, 3]
                X_new['total_area'] = X_new['sepal_area'] + X_new['petal_area']
                X_new['area_ratio'] = X_new['sepal_area'] / (X_new['petal_area'] + 1e-8)
            
            # Tạo interactions
            if self.create_interactions:
                # Combinations
                X_new['sepal_length_x_width'] = X.iloc[:, 0] * X.iloc[:, 1]
                X_new['petal_length_x_width'] = X.iloc[:, 2] * X.iloc[:, 3]
                X_new['diagonal_sepal'] = np.sqrt(X.iloc[:, 0]**2 + X.iloc[:, 1]**2)
                X_new['diagonal_petal'] = np.sqrt(X.iloc[:, 2]**2 + X.iloc[:, 3]**2)
                
                # Sums and differences
                X_new['total_length'] = X.iloc[:, 0] + X.iloc[:, 2]
                X_new['total_width'] = X.iloc[:, 1] + X.iloc[:, 3]
                X_new['length_diff'] = X.iloc[:, 0] - X.iloc[:, 2]
                X_new['width_diff'] = X.iloc[:, 1] - X.iloc[:, 3]
        
        return X_new

class FeatureEngineer:
    """Class chính để feature engineering"""
    
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.feature_names = []
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.feature_importance = {}
        
    def create_polynomial_features(self, X, degree=2, include_bias=False):
        """Tạo polynomial features"""
        print(f"Tạo polynomial features với degree={degree}")
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Lấy tên features
        if hasattr(X, 'columns'):
            feature_names = poly.get_feature_names_out(X.columns)
        else:
            feature_names = poly.get_feature_names_out([f'feature_{i}' for i in range(X.shape[1])])
            
        print(f"Tạo được {X_poly.shape[1]} features từ {X.shape[1]} features gốc")
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def create_interaction_features(self, X):
        """Tạo interaction features"""
        print("Tạo interaction features")
        
        creator = CustomFeatureCreator()
        X_transformed = creator.fit_transform(X)
        
        print(f"Tạo được {X_transformed.shape[1]} features từ {X.shape[1]} features gốc")
        
        return X_transformed
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """Scale features"""
        print(f"Scaling features với method={method}")
        
        if method not in self.scalers:
            raise ValueError(f"Method {method} không hợp lệ. Chọn: {list(self.scalers.keys())}")
        
        scaler = self.scalers[method]
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Chuyển về DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler
    
    def select_features_univariate(self, X, y, k: Union[int, str] = 'all'):
        """Feature selection sử dụng univariate statistics"""
        print(f"Feature selection với k={k}")
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=k)  # type: ignore
        else:
            selector = SelectKBest(score_func=f_regression, k=k)  # type: ignore
        
        X_selected = selector.fit_transform(X, y)
        
        # Lấy tên features được chọn
        support_mask = selector.get_support()
        if hasattr(X, 'columns') and X.columns is not None:
            selected_features = X.columns[support_mask].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X.shape[1]) if support_mask is not None and support_mask[i]]
        
        # Lưu feature importance
        self.feature_importance['univariate'] = dict(zip(
            X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
            selector.scores_
        ))
        
        print(f"Chọn được {len(selected_features)} features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selector  # type: ignore
    
    def select_features_rfe(self, X, y, n_features_to_select=None):
        """Feature selection sử dụng RFE"""
        print(f"RFE feature selection với n_features={n_features_to_select}")
        
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        if n_features_to_select is None:
            # Sử dụng RFECV để tự động chọn số features
            selector = RFECV(estimator, step=1, cv=5, scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error')
        else:
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        
        X_selected = selector.fit_transform(X, y)
        
        # Lấy tên features được chọn
        support_mask = selector.get_support()
        if hasattr(X, 'columns') and X.columns is not None:
            selected_features = X.columns[support_mask].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X.shape[1]) if support_mask is not None and support_mask[i]]
        
        # Lưu feature importance
        if hasattr(selector.estimator_, 'feature_importances_'):
            self.feature_importance['rfe'] = dict(zip(
                X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
                selector.estimator_.feature_importances_  # type: ignore
            ))
        
        print(f"Chọn được {len(selected_features)} features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selector  # type: ignore
    
    def select_features_importance(self, X, y, threshold=0.01):
        """Feature selection dựa trên importance"""
        print(f"Feature selection với importance threshold={threshold}")
        
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Lấy feature importance
        importances = model.feature_importances_
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Chọn features có importance > threshold
        selected_indices = importances > threshold
        selected_features = [name for name, selected in zip(feature_names, selected_indices) if selected]
        
        X_selected = X.iloc[:, selected_indices] if hasattr(X, 'iloc') else X[:, selected_indices]
        
        # Lưu feature importance
        self.feature_importance['importance'] = dict(zip(feature_names, importances))
        
        print(f"Chọn được {len(selected_features)} features: {selected_features}")
        
        return X_selected, selected_features
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """Áp dụng PCA"""
        if n_components is None:
            # Tự động chọn số components để giữ được variance_threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        print(f"Áp dụng PCA với {n_components} components")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Tạo tên cột
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(data=X_pca, columns=pca_columns, index=X.index)  # type: ignore
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca_df, pca
    
    def create_feature_pipeline(self, config):
        """Tạo pipeline cho feature engineering"""
        print("Tạo Feature Engineering Pipeline")
        
        steps = []
        
        # Custom feature creation
        if config.get('create_custom_features', True):
            steps.append(('custom_features', CustomFeatureCreator(
                create_ratios=config.get('create_ratios', True),
                create_areas=config.get('create_areas', True),
                create_interactions=config.get('create_interactions', True)
            )))
        
        # Polynomial features
        if config.get('polynomial_degree', 1) > 1:
            steps.append(('polynomial', PolynomialFeatures(
                degree=config.get('polynomial_degree', 2),
                include_bias=False
            )))
        
        # Scaling
        scaler_method = config.get('scaler', 'standard')
        if scaler_method in self.scalers:
            steps.append(('scaler', self.scalers[scaler_method]))
        
        # Feature selection
        if config.get('feature_selection', False):
            if self.task_type == 'classification':
                steps.append(('feature_selection', SelectKBest(
                    score_func=f_classif,
                    k=config.get('k_features', 'all')
                )))
            else:
                steps.append(('feature_selection', SelectKBest(
                    score_func=f_regression,
                    k=config.get('k_features', 'all')
                )))
        
        # PCA
        if config.get('apply_pca', False):
            steps.append(('pca', PCA(
                n_components=config.get('n_components', None),
                random_state=42
            )))
        
        pipeline = Pipeline(steps)
        print(f"Pipeline tạo với {len(steps)} bước: {[step[0] for step in steps]}")
        
        return pipeline
    
    def visualize_feature_importance(self, importance_dict, top_n=20):
        """Visualize feature importance"""
        plt.figure(figsize=(12, 8))
        
        # Sắp xếp theo importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Lấy top N features
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def compare_feature_sets(self, X_original, X_engineered, y, cv=5):
        """So sánh hiệu suất giữa feature sets"""
        print("So sánh hiệu suất feature sets")
        
        from sklearn.model_selection import cross_val_score
        
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        # Đánh giá với features gốc
        scores_original = cross_val_score(model, X_original, y, cv=cv, scoring=scoring)
        
        # Đánh giá với features engineered
        scores_engineered = cross_val_score(model, X_engineered, y, cv=cv, scoring=scoring)
        
        print(f"Original features: {scores_original.mean():.4f} (±{scores_original.std():.4f})")
        print(f"Engineered features: {scores_engineered.mean():.4f} (±{scores_engineered.std():.4f})")
        
        improvement = scores_engineered.mean() - scores_original.mean()
        print(f"Improvement: {improvement:.4f}")
        
        return {
            'original': scores_original,
            'engineered': scores_engineered,
            'improvement': improvement
        }

def main():
    """Demo feature engineering"""
    print("=" * 80)
    print("DEMO FEATURE ENGINEERING")
    print("=" * 80)
    
    try:
        # Đọc dữ liệu
        df = pd.read_csv('data/Iris.csv')
        print(f"✓ Đọc dữ liệu thành công: {df.shape}")
        
        # Loại bỏ cột Id
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        # Tách features và target
        X = df.drop('Species', axis=1)
        y = df['Species']
        
        print(f"Features gốc: {X.shape[1]} features")
        print(f"Features: {list(X.columns)}")
        
        # Tạo Feature Engineer
        fe = FeatureEngineer(task_type='classification')
        
        # 1. Tạo custom features
        print("\n" + "="*50)
        print("1. TẠO CUSTOM FEATURES")
        print("="*50)
        X_custom = fe.create_interaction_features(X)
        print(f"Sau khi tạo custom features: {X_custom.shape[1]} features")
        
        # 2. Tạo polynomial features
        print("\n" + "="*50)
        print("2. TẠO POLYNOMIAL FEATURES")
        print("="*50)
        X_poly = fe.create_polynomial_features(X, degree=2)
        print(f"Polynomial features: {X_poly.shape[1]} features")
        
        # 3. Feature selection
        print("\n" + "="*50)
        print("3. FEATURE SELECTION")
        print("="*50)
        
        # Univariate selection
        X_selected, selector = fe.select_features_univariate(X_custom, y, k=10)
        
        # Importance-based selection
        X_important, important_features = fe.select_features_importance(X_custom, y, threshold=0.01)
        
        # 4. Scaling
        print("\n" + "="*50)
        print("4. FEATURE SCALING")
        print("="*50)
        result = fe.scale_features(X_selected, method='standard')
        X_scaled, scaler = result[0], result[1]
        
        # 5. So sánh hiệu suất
        print("\n" + "="*50)
        print("5. SO SÁNH HIỆU SUẤT")
        print("="*50)
        comparison = fe.compare_feature_sets(X, X_custom, y)
        
        # 6. Visualize feature importance
        print("\n" + "="*50)
        print("6. FEATURE IMPORTANCE")
        print("="*50)
        if 'importance' in fe.feature_importance:
            fe.visualize_feature_importance(fe.feature_importance['importance'])
        
        # 7. Tạo pipeline
        print("\n" + "="*50)
        print("7. TẠO PIPELINE")
        print("="*50)
        
        config = {
            'create_custom_features': True,
            'polynomial_degree': 2,
            'scaler': 'standard',
            'feature_selection': True,
            'k_features': 10,
            'apply_pca': False
        }
        
        pipeline = fe.create_feature_pipeline(config)
        
        # Test pipeline
        X_transformed = pipeline.fit_transform(X, y)
        print(f"Pipeline output: {X_transformed.shape}")
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file data/Iris.csv")
        return
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH FEATURE ENGINEERING")
    print("=" * 80)

if __name__ == "__main__":
    main() 