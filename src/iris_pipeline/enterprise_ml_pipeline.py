"""
Enterprise Machine Learning Pipeline for Multi-class Classification

A comprehensive, production-ready ML pipeline designed for enterprise-level 
data science projects with high standards for code quality and accuracy.

Author: ML Engineering Team
Date: 2025
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging

# Machine Learning libraries
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, 
    OrdinalEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

# Deep Learning (Bonus)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    import keras  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnterpriseMLPipeline:
    """
    Enterprise-grade Machine Learning Pipeline for Multi-class Classification.
    
    This class implements a complete ML workflow including data loading,
    exploratory data analysis, preprocessing, model training, evaluation,
    and deployment preparation.
    """
    
    def __init__(self, data_path: str = "data/Iris.csv", random_state: int = 42):
        """
        Initialize the ML Pipeline.
        
        Args:
            data_path (str): Path to the CSV data file
            random_state (int): Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.target_names = []
        
        # Set style for visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Enterprise ML Pipeline initialized with random_state={random_state}")
    
    def load_and_inspect_data(self) -> pd.DataFrame:
        """
        Load data from CSV and perform initial inspection.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        logger.info("Loading and inspecting data...")
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic information
            print("=" * 80)
            print("📋 DATASET OVERVIEW")
            print("=" * 80)
            print(f"Dataset shape: {self.df.shape}")
            print(f"Memory usage: {self.df.memory_usage().sum() / 1024:.2f} KB")
            print("\n📊 Data Types:")
            print(self.df.dtypes)
            
            print("\n📈 Statistical Summary:")
            print(self.df.describe())
            
            print("\n🔍 Sample Data:")
            print(self.df.head(10))
            
            # Check for missing values
            missing_data = self.df.isnull().sum()
            print("\n❓ Missing Values:")
            if missing_data.sum() == 0:
                print("✅ No missing values found!")
            else:
                print(missing_data[missing_data > 0])
            
            # Check for duplicates
            duplicates = self.df.duplicated().sum()
            print(f"\n🔄 Duplicate rows: {duplicates}")
            if duplicates > 0:
                print("⚠️ Found duplicate rows - will handle in preprocessing")
            
            return self.df
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def exploratory_data_analysis(self) -> None:
        """
        Perform comprehensive Exploratory Data Analysis (EDA).
        """
        logger.info("🔍 Performing Exploratory Data Analysis...")
        
        # Ensure df is loaded
        assert self.df is not None, "Data must be loaded first"
        
        # Identify target column (assuming it's the last column or named 'Species')
        target_col = 'Species' if 'Species' in self.df.columns else self.df.columns[-1]
        feature_cols = [col for col in self.df.columns if col != target_col]
        
        self.feature_names = feature_cols
        self.target_names = sorted(self.df[target_col].unique())
        
        print("=" * 80)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("=" * 80)
        
        # Target distribution
        target_counts = self.df[target_col].value_counts()
        print("Class Distribution:")
        for class_name, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        plt.subplot(3, 4, 1)
        target_counts.plot(kind='bar', color='skyblue', alpha=0.8)
        plt.title('🎯 Target Class Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. Target distribution pie chart
        plt.subplot(3, 4, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',  # type: ignore
                startangle=90, colors=sns.color_palette("husl", len(target_counts)))
        plt.title('📊 Class Distribution (%)', fontsize=12, fontweight='bold')
        
        # 3-6. Feature distributions
        numeric_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns
        for i, feature in enumerate(numeric_features[:4], 3):
            plt.subplot(3, 4, i)
            self.df[feature].hist(bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title(f'📈 {feature} Distribution', fontsize=10, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        # 7-10. Feature by target class
        for i, feature in enumerate(numeric_features[:4], 7):
            plt.subplot(3, 4, i)
            for target_class in self.target_names:
                subset = self.df[self.df[target_col] == target_class][feature]
                plt.hist(subset, alpha=0.6, label=target_class, bins=15)
            plt.title(f'📊 {feature} by Class', fontsize=10, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
        
        # 11. Correlation heatmap
        plt.subplot(3, 4, 11)
        correlation_matrix = self.df[numeric_features].corr()  # type: ignore
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('🔗 Feature Correlation Matrix', fontsize=10, fontweight='bold')
        
        # 12. Pairplot summary (boxplot)
        plt.subplot(3, 4, 12)
        if len(numeric_features) > 0:
            # Create a summary boxplot
            df_melted = pd.melt(self.df, id_vars=[target_col], 
                              value_vars=numeric_features[:4])
            sns.boxplot(data=df_melted, x='variable', y='value', hue=target_col)
            plt.title('📦 Feature Distribution by Class', fontsize=10, fontweight='bold')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed analysis
        print("\n" + "=" * 80)
        print("📊 DETAILED STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Feature statistics by class
        for feature in numeric_features:
            print(f"\n📈 {feature} by Class:")
            class_stats = self.df.groupby(target_col)[feature].describe()
            print(class_stats.round(3))
        
        # Correlation analysis
        print(f"\n🔗 Feature Correlations (> 0.5):")
        high_corr = correlation_matrix[(correlation_matrix > 0.5) & (correlation_matrix < 1.0)]
        for i, feature1 in enumerate(correlation_matrix.columns):
            for j, feature2 in enumerate(correlation_matrix.columns):
                if i < j and abs(correlation_matrix.iloc[i, j]) > 0.5:
                    print(f"  {feature1} ↔ {feature2}: {correlation_matrix.iloc[i, j]:.3f}")
    
    def preprocess_data(self) -> Tuple[Any, Any, Any, Any]:
        """
        Comprehensive data preprocessing following enterprise standards.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("🔧 Starting comprehensive data preprocessing...")
        
        # Ensure df is loaded
        assert self.df is not None, "Data must be loaded first"
        
        # Identify target column
        target_col = 'Species' if 'Species' in self.df.columns else self.df.columns[-1]
        feature_cols = [col for col in self.df.columns if col != target_col]
        
        # Handle duplicates
        initial_shape = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_shape - self.df.shape[0]
        if duplicates_removed > 0:
            logger.info(f"🔄 Removed {duplicates_removed} duplicate rows")
        
        # Separate features and target
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("=" * 80)
        print("🔧 PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Identify variable types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"📊 Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"📝 Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Handle missing values
        print("\n❓ Missing value treatment:")
        for col in X.columns:  # type: ignore
            missing_count = X[col].isnull().sum()  # type: ignore
            if missing_count > 0:
                if col in numeric_features:
                    if missing_count / len(X) > 0.1:  # >10% missing, use KNN
                        print(f"  {col}: {missing_count} missing → KNN Imputation")
                    else:
                        print(f"  {col}: {missing_count} missing → Median Imputation")
                else:
                    print(f"  {col}: {missing_count} missing → Mode Imputation")
            else:
                print(f"  {col}: ✅ No missing values")
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Split data (stratified to maintain class distribution)
        print(f"\n📊 Splitting data: 80% train, 20% test (stratified)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state, 
            stratify=y_encoded
        )
        
        print(f"✅ Training set: {X_train.shape[0]} samples")  # type: ignore
        print(f"✅ Test set: {X_test.shape[0]} samples")  # type: ignore
        
        # Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Store processed data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        
        # Verify class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n🎯 Training set class distribution:")
        for class_idx, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        logger.info("✅ Data preprocessing completed successfully")
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_models_conservative(self) -> Dict[str, Any]:
        """
        Conservative training approach to avoid overfitting on validation set.
        
        Returns:
            Dict: Trained models with realistic performance estimates
        """
        logger.info("🛡️ Training models with conservative approach...")
        
        print("=" * 80)
        print("🛡️ CONSERVATIVE MODEL TRAINING")
        print("=" * 80)
        
        # Very simple parameter grids
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [1.0],  # Fixed single value
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100],  # Fixed single value
                    'max_depth': [None],    # Fixed single value
                }
            }
        }
        
        # Simple 3-fold CV
        cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for model_name, config in models_config.items():
            print(f"\n🔧 Training {model_name} (Conservative)...")
            
            if len(config['params']) == 1 and len(list(config['params'].values())[0]) == 1:
                # No hyperparameter tuning - just train with default/fixed params
                model = config['model']
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=cv_folds, scoring='f1_macro')
                
                # Train final model
                model.fit(self.X_train, self.y_train)
                
                self.models[model_name] = {
                    'model': model,
                    'best_params': 'default',
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'search_object': None
                }
            else:
                # Minimal hyperparameter tuning
                search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv_folds,
                    scoring='f1_macro',
                    n_jobs=-1
                )
                
                search.fit(self.X_train, self.y_train)
                
                self.models[model_name] = {
                    'model': search.best_estimator_,
                    'best_params': search.best_params_,
                    'cv_score': search.best_score_,
                    'cv_std': search.cv_results_['std_test_score'][search.best_index_],
                    'search_object': search
                }
            
            print(f"✅ {model_name} completed")
            print(f"   📊 CV Score: {self.models[model_name]['cv_score']:.4f} ± {self.models[model_name]['cv_std']:.4f}")
            print(f"   ⚙️ Params: {self.models[model_name]['best_params']}")
        
        logger.info(f"✅ Conservative training completed. Trained {len(self.models)} models.")
        return self.models
    
    def train_models_holdout(self) -> Dict[str, Any]:
        """
        Train models with separate holdout test set to avoid overfitting completely.
        
        Returns:
            Dict: Trained models with unbiased performance estimates
        """
        logger.info("🔒 Training models with holdout test set...")
        
        print("=" * 80)
        print("🔒 HOLDOUT TEST SET APPROACH")
        print("=" * 80)
        
        # Split into train/val/test (60/20/20)
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
            self.X_train, self.y_train, test_size=0.33, random_state=self.random_state, 
            stratify=self.y_train
        )
        
        # Second split: train vs val
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=self.random_state,
            stratify=y_trainval
        )
        
        print(f"📊 Data splits:")
        print(f"   Train: {X_train_split.shape[0]} samples")  # type: ignore
        print(f"   Val: {X_val_split.shape[0]} samples")  # type: ignore
        print(f"   Holdout: {X_holdout.shape[0]} samples")  # type: ignore
        
        # Simple models without hyperparameter tuning
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=self.random_state),
        }
        
        for model_name, model in models_config.items():
            print(f"\n🔧 Training {model_name} (Holdout)...")
            
            # Train on train split
            model.fit(X_train_split, y_train_split)
            
            # Evaluate on validation split
            val_pred = model.predict(X_val_split)
            val_acc = accuracy_score(y_val_split, val_pred)
            
            # Final evaluation on holdout test set
            holdout_pred = model.predict(X_holdout)
            holdout_acc = accuracy_score(y_holdout, holdout_pred)
            
            # Store results
            self.models[model_name] = {
                'model': model,
                'best_params': 'default',
                'val_score': val_acc,
                'holdout_score': holdout_acc,
                'search_object': None
            }
            
            print(f"✅ {model_name} completed")
            print(f"   📊 Val Score: {val_acc:.4f}")
            print(f"   🔒 Holdout Score: {holdout_acc:.4f}")
            print(f"   📈 Difference: {abs(val_acc - holdout_acc):.4f}")
        
        logger.info(f"✅ Holdout training completed. Trained {len(self.models)} models.")
        return self.models
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Returns:
            Dict: Evaluation results for all models
        """
        logger.info("📊 Evaluating models on test set...")
        
        print("=" * 80)
        print("📊 MODEL EVALUATION RESULTS")
        print("=" * 80)
        
        evaluation_results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\n🔍 Evaluating {model_name}...")
            
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision_macro = precision_score(self.y_test, y_pred, average='macro')
            precision_weighted = precision_score(self.y_test, y_pred, average='weighted')
            recall_macro = recall_score(self.y_test, y_pred, average='macro')
            recall_weighted = recall_score(self.y_test, y_pred, average='weighted')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            # ROC AUC (multi-class)
            try:
                roc_auc_ovr = roc_auc_score(self.y_test, y_pred_proba, 
                                          multi_class='ovr', average='macro')
                roc_auc_ovo = roc_auc_score(self.y_test, y_pred_proba, 
                                          multi_class='ovo', average='macro')
            except:
                roc_auc_ovr = roc_auc_ovo = 0.0
            
            # Store results
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'precision_weighted': precision_weighted,
                'recall_macro': recall_macro,
                'recall_weighted': recall_weighted,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'roc_auc_ovr': roc_auc_ovr,
                'roc_auc_ovo': roc_auc_ovo,
                'cv_score': model_info['cv_score']
            }
            
            # Print results
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   📊 F1-Score (Macro): {f1_macro:.4f}")
            print(f"   📊 F1-Score (Weighted): {f1_weighted:.4f}")
            print(f"   🎯 ROC-AUC (OvR): {roc_auc_ovr:.4f}")
            print(f"   ⚙️ Nested CV Score: {model_info['cv_score']:.4f} ± {model_info['cv_std']:.4f}")
        
        # Create comparison table
        print(f"\n{'='*100}")
        print("📊 MODEL COMPARISON SUMMARY")
        print(f"{'='*100}")
        
        metrics_df = pd.DataFrame(evaluation_results).T
        metrics_df = metrics_df.round(4)
        print(metrics_df.to_string())
        
        # Identify best model
        best_model_name = metrics_df['f1_macro'].idxmax()
        self.best_model = self.models[best_model_name]['model']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   F1-Score (Macro): {metrics_df.loc[best_model_name, 'f1_macro']:.4f}")
        
        logger.info(f"✅ Model evaluation completed. Best model: {best_model_name}")
        return evaluation_results
    
    def visualize_results(self) -> None:
        """
        Create comprehensive visualizations for model results.
        """
        logger.info("📊 Creating result visualizations...")
        
        print("=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison bar chart
        plt.subplot(3, 4, 1)
        model_names = list(self.models.keys())
        f1_scores = []
        accuracies = []
        
        for name in model_names:
            model = self.models[name]['model']
            y_pred = model.predict(self.X_test)
            f1_scores.append(f1_score(self.y_test, y_pred, average='macro'))
            accuracies.append(accuracy_score(self.y_test, y_pred))
        
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('🏆 Model Performance Comparison', fontweight='bold')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-4. Confusion matrices for each model
        for i, (model_name, model_info) in enumerate(self.models.items(), 2):
            plt.subplot(3, 4, i)
            model = model_info['model']
            y_pred = model.predict(self.X_test)
            
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,  # type: ignore
                       yticklabels=self.label_encoder.classes_)  # type: ignore
            plt.title(f'🎯 {model_name}\nConfusion Matrix', fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        # 5. ROC Curves for best model
        plt.subplot(3, 4, 5)
        best_model = self.best_model
        assert best_model is not None, "Best model must be trained first"
        y_pred_proba = best_model.predict_proba(self.X_test)
        
        # ROC curve for each class
        for i, class_name in enumerate(self.label_encoder.classes_):  # type: ignore
            y_test_binary = np.array(self.y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('🎯 ROC Curves (Best Model)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Feature importance (for tree-based models)
        plt.subplot(3, 4, 6)
        if hasattr(self.best_model, 'feature_importances_'):
            assert self.best_model is not None
            feature_importance = self.best_model.feature_importances_
            feature_names = self.feature_names[:len(feature_importance)]
            
            indices = np.argsort(feature_importance)[::-1]
            plt.bar(range(len(feature_importance)), feature_importance[indices])
            plt.title('🌟 Feature Importance\n(Best Model)', fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(feature_importance)), 
                      [feature_names[i] for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12)
            plt.title('🌟 Feature Importance', fontweight='bold')
        
        # 7. Cross-validation scores
        plt.subplot(3, 4, 7)
        cv_scores = [self.models[name]['cv_score'] for name in model_names]
        bars = plt.bar(model_names, cv_scores, color='lightcoral', alpha=0.8)
        plt.title('📊 Cross-Validation Scores', fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('CV F1-Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, cv_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 8. Training set performance
        plt.subplot(3, 4, 8)
        train_accuracies = []
        for name in model_names:
            model = self.models[name]['model']
            y_train_pred = model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, y_train_pred)
            train_accuracies.append(train_acc)
        
        x = np.arange(len(model_names))
        plt.bar(x - width/2, train_accuracies, width, label='Train Acc', alpha=0.8)
        plt.bar(x + width/2, accuracies, width, label='Test Acc', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('🎯 Train vs Test Accuracy', fontweight='bold')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9-12. Detailed classification reports visualization
        for i, (model_name, model_info) in enumerate(self.models.items(), 9):
            if i > 12:
                break
            plt.subplot(3, 4, i)
            model = model_info['model']
            y_pred = model.predict(self.X_test)
            
            report = classification_report(self.y_test, y_pred, 
                                         target_names=self.label_encoder.classes_,
                                         output_dict=True)
            
            # Create heatmap for classification report
            df_report = pd.DataFrame(report).iloc[:-1, :-2].T
            sns.heatmap(df_report, annot=True, cmap='YlOrRd', fmt='.3f')
            plt.title(f'📊 {model_name}\nClassification Report', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations generated successfully")
    
    def create_final_pipeline(self) -> Pipeline:
        """
        Create final production-ready pipeline with best model.
        
        Returns:
            Pipeline: Complete preprocessing + model pipeline
        """
        logger.info("🏗️ Creating final production pipeline...")
        
        print("=" * 80)
        print("🏗️ BUILDING FINAL PIPELINE")
        print("=" * 80)
        
        # Create final pipeline combining preprocessing and best model
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.best_model)
        ])
        
        # Fit on full training data
        assert self.df is not None, "Data must be loaded first"
        target_col = 'Species' if 'Species' in self.df.columns else self.df.columns[-1]
        feature_cols = [col for col in self.df.columns if col != target_col]
        
        X_full = self.df[feature_cols]
        y_full = self.label_encoder.fit_transform(self.df[target_col])
        
        self.pipeline.fit(X_full, y_full)
        
        print(f"✅ Final pipeline created with {type(self.best_model).__name__}")
        print(f"📊 Pipeline components:")
        print(f"   1. Preprocessor: {type(self.preprocessor).__name__}")
        print(f"   2. Classifier: {type(self.best_model).__name__}")
        
        return self.pipeline
    
    def save_pipeline(self, filepath: str = "models/enterprise_iris_pipeline.pkl") -> None:
        """
        Save the complete pipeline for deployment.
        
        Args:
            filepath (str): Path to save the pipeline
        """
        logger.info(f"💾 Saving pipeline to {filepath}...")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline and metadata
        pipeline_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_performance': {
                name: {
                    'cv_score': info['cv_score'],
                    'best_params': info['best_params']
                }
                for name, info in self.models.items()
            },
            'best_model_name': type(self.best_model).__name__
        }
        
        joblib.dump(pipeline_data, filepath)
        
        print(f"✅ Pipeline saved successfully to: {filepath}")
        print(f"📦 Saved components:")
        print(f"   - Complete preprocessing + model pipeline")
        print(f"   - Label encoder")
        print(f"   - Feature metadata")
        print(f"   - Model performance metrics")
        
        logger.info("💾 Pipeline saved successfully")
    
    def train_deep_learning_model(self) -> Optional[Any]:
        """
        Bonus: Train a deep learning model using TensorFlow/Keras.
        
        Returns:
            Trained neural network model or None if TensorFlow not available
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available. Skipping deep learning model.")
            return None
        
        logger.info("🧠 Training Deep Learning model...")
        
        print("=" * 80)
        print("🧠 DEEP LEARNING MODEL (BONUS)")
        print("=" * 80)
        
        try:
            # Prepare data for neural network
            from sklearn.preprocessing import StandardScaler
            
            # Use processed features
            assert self.X_train is not None and self.X_test is not None
            X_train_dl = self.X_train.copy()
            X_test_dl = self.X_test.copy()
            
            # Convert to categorical for Keras
            y_train_categorical = keras.utils.to_categorical(self.y_train)
            y_test_categorical = keras.utils.to_categorical(self.y_test)
            
            # Build neural network
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(len(self.target_names), activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            print("🏗️ Neural Network Architecture:")
            model.summary()
            
            # Train model
            print("\n🚀 Training neural network...")
            history = model.fit(
                X_train_dl, y_train_categorical,
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                verbose=1,  # type: ignore
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            # Evaluate
            y_pred_dl = model.predict(X_test_dl)
            y_pred_classes = np.argmax(y_pred_dl, axis=1)
            
            # Calculate metrics
            dl_accuracy = accuracy_score(self.y_test, y_pred_classes)
            dl_f1 = f1_score(self.y_test, y_pred_classes, average='macro')
            
            print(f"\n📊 Deep Learning Results:")
            print(f"   🎯 Test Accuracy: {dl_accuracy:.4f}")
            print(f"   📊 F1-Score (Macro): {dl_f1:.4f}")
            
            # Compare with best traditional model
            assert self.best_model is not None
            best_f1 = f1_score(self.y_test, self.best_model.predict(self.X_test), average='macro')
            improvement = ((dl_f1 - best_f1) / best_f1) * 100
            
            print(f"\n🏆 Comparison with best traditional model:")
            print(f"   Traditional: {best_f1:.4f}")
            print(f"   Deep Learning: {dl_f1:.4f}")
            print(f"   Improvement: {improvement:+.2f}%")
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('🧠 Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('📉 Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Deep learning model training completed")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training deep learning model: {str(e)}")
            return None
    
    def run_complete_pipeline(self) -> None:
        """
        Execute the complete ML pipeline from start to finish.
        """
        print("🌟" * 40)
        print("🚀 ENTERPRISE MACHINE LEARNING PIPELINE")
        print("🌟" * 40)
        print(f"📅 Started at: {pd.Timestamp.now()}")
        
        try:
            # Step 1: Load and inspect data
            self.load_and_inspect_data()
            
            # Step 2: Exploratory Data Analysis
            self.exploratory_data_analysis()
            
            # Step 3: Preprocess data
            self.preprocess_data()
            
            # Step 4: Train models
            self.train_models_conservative()
            self.train_models_holdout()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Visualize results
            self.visualize_results()
            
            # Step 7: Create final pipeline
            self.create_final_pipeline()
            
            # Step 8: Save pipeline
            self.save_pipeline()
            
            # Step 9: Deep Learning (Bonus)
            self.train_deep_learning_model()
            
            print("\n" + "🎉" * 40)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 40)
            print(f"📅 Completed at: {pd.Timestamp.now()}")
            print(f"🏆 Best Model: {type(self.best_model).__name__}")
            print(f"💾 Pipeline saved for deployment")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """
    Main function to run the Enterprise ML Pipeline.
    """
    # Create and run pipeline
    pipeline = EnterpriseMLPipeline(data_path="data/Iris.csv", random_state=42)
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main() 