#!/usr/bin/env python3
"""
Demo script for Enterprise ML Pipeline

Quick demonstration of the enterprise pipeline functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

print("=" * 30)
print("Enterprise ML Pipeline Demo")
print("=" * 30)

try:
    # Test basic imports
    print("Testing imports...")
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    print("Basic imports successful")
    
    # Test enterprise pipeline import
    try:
        from iris_pipeline.enterprise_ml_pipeline import EnterpriseMLPipeline
        print("Enterprise pipeline import successful")
    except ImportError as e:
        print(f"Enterprise pipeline import failed: {e}")
        print("This is expected if XGBoost/TensorFlow not installed")
    
    # Test data loading
    data_path = project_root / "data" / "Iris.csv"
    if data_path.exists():
        print(f"✅ Data file found: {data_path}")
        
        # Quick data test
        df = pd.read_csv(data_path)
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📊 Sample data:\n{df.head(3)}")
        
        # Simple ML demo
        print("\n🤖 Quick ML Demo...")
        target_col = 'Species' if 'Species' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = pd.factorize(df[target_col])[0]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Simple Logistic Regression Accuracy: {accuracy:.3f}")
        
    else:
        print(f"❌ Data file not found: {data_path}")
    
    print("\n" + "✅" * 30)
    print("🎉 Demo completed successfully!")
    print("🚀 Ready to run full enterprise pipeline:")
    print("   python scripts/run_enterprise_pipeline.py")
    print("   or: make enterprise")
    print("✅" * 30)
    
    # Run enterprise pipeline
    pipeline = EnterpriseMLPipeline("data/Iris.csv")
    pipeline.run_complete_pipeline()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Install missing packages:")
    print("   pip install pandas numpy matplotlib seaborn scikit-learn")
    
except Exception as e:
    print(f"❌ Error: {e}")
    raise 