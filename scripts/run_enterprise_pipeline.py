#!/usr/bin/env python3
"""
Runner script for Enterprise ML Pipeline

This script runs the complete enterprise-grade machine learning pipeline
for multi-class classification with the Iris dataset.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from iris_pipeline.enterprise_ml_pipeline import EnterpriseMLPipeline
    
    def main():
        """Run the Enterprise ML Pipeline."""
        print("🌟" * 50)
        print("🚀 ENTERPRISE MACHINE LEARNING PIPELINE")
        print("   Multi-class Classification with Advanced Features")
        print("🌟" * 50)
        
        # Check if data file exists
        data_path = project_root / "data" / "Iris.csv"
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure Iris.csv is in the data/ directory")
            return
        
        # Create and run pipeline
        try:
            pipeline = EnterpriseMLPipeline(
                data_path=str(data_path), 
                random_state=42
            )
            pipeline.run_complete_pipeline()
            
            print("\n" + "🎉" * 50)
            print("✅ Enterprise ML Pipeline completed successfully!")
            print("📦 Check the models/ directory for saved pipeline")
            print("🎉" * 50)
            
        except ImportError as e:
            print(f"❌ Missing dependencies: {e}")
            print("🔧 Install required packages:")
            print("   pip install xgboost tensorflow")
            print("   or run: pip install -r requirements.txt")
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            raise

    if __name__ == "__main__":
        main()
        
except ImportError:
    print("❌ Cannot import enterprise pipeline module")
    print("🔧 Make sure you're running from the project root directory")
    print("   python scripts/run_enterprise_pipeline.py") 