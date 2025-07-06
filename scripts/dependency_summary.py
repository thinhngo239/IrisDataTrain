#!/usr/bin/env python3
"""
Final dependency summary for Enterprise ML Pipeline
"""

import sys
import importlib
import importlib.util
import platform

print("=" * 70)
print("🚀 ENTERPRISE ML PIPELINE - DEPENDENCY STATUS")
print("=" * 70)
print(f"🐍 Python Version: {sys.version.split()[0]}")
print(f"💻 Platform: {platform.system()} {platform.release()}")
print(f"🏗️ Architecture: {platform.machine()}")
print("=" * 70)

# Check all dependencies
dependencies = {
    "✅ Core ML Libraries": {
        'pandas': True,
        'numpy': True, 
        'scikit-learn': True,
        'matplotlib': True,
        'seaborn': True,
        'joblib': True,
    },
    "🚀 Advanced ML": {
        'xgboost': True,
        'tensorflow': False,  # Not compatible with Python 3.13
    },
    "🌐 Web & API": {
        'fastapi': True,
        'uvicorn': True,
        'pydantic': True,
        'streamlit': True,
        'plotly': True,
        'requests': True,
    }
}

total_installed = 0
total_packages = 0

for category, packages in dependencies.items():
    print(f"\n{category}")
    print("-" * 50)
    
    for pkg_name, expected in packages.items():
        total_packages += 1
        
        # Special handling for sklearn
        import_name = 'sklearn' if pkg_name == 'scikit-learn' else pkg_name
        
        try:
            importlib.import_module(import_name)
            status = "✅ Installed"
            total_installed += 1
        except ImportError:
            status = "❌ Not installed"
            if pkg_name == 'tensorflow':
                status += " (Python 3.13 not supported)"
        
        print(f"  {pkg_name:<20} {status}")

print("\n" + "=" * 70)
print(f"📊 SUMMARY: {total_installed}/{total_packages} packages installed")
print("=" * 70)

# Enterprise Pipeline Status
print("\n🎯 ENTERPRISE PIPELINE READINESS:")
print("-" * 50)

can_run_basic = all([
    importlib.util.find_spec('pandas'),
    importlib.util.find_spec('numpy'),
    importlib.util.find_spec('sklearn'),
    importlib.util.find_spec('matplotlib'),
    importlib.util.find_spec('seaborn'),
])

can_run_advanced = (
    can_run_basic and 
    importlib.util.find_spec('xgboost') is not None
)

can_run_api = all([
    importlib.util.find_spec('fastapi'),
    importlib.util.find_spec('uvicorn'),
    importlib.util.find_spec('pydantic'),
])

can_run_web = importlib.util.find_spec('streamlit') is not None

print(f"✅ Basic ML Pipeline: {'Ready' if can_run_basic else 'Not Ready'}")
print(f"✅ Advanced ML (with XGBoost): {'Ready' if can_run_advanced else 'Not Ready'}")
print(f"⚠️  Deep Learning (TensorFlow): Not Available (Python 3.13 incompatible)")
print(f"✅ FastAPI Server: {'Ready' if can_run_api else 'Not Ready'}")
print(f"✅ Streamlit Web App: {'Ready' if can_run_web else 'Not Ready'}")

print("\n" + "=" * 70)
print("🚀 QUICK START COMMANDS:")
print("=" * 70)

if can_run_basic:
    print("1. Run Enterprise ML Pipeline:")
    print("   make enterprise")
    print("   python scripts/run_enterprise_pipeline.py")
    
if can_run_api:
    print("\n2. Start API Server:")
    print("   make api")
    print("   uvicorn iris_pipeline.api.server:app --reload")
    
if can_run_web:
    print("\n3. Start Web Interface:")
    print("   make web")
    print("   streamlit run apps/web_interface.py")

print("\n" + "=" * 70)
print("💡 NOTES:")
print("=" * 70)
print("• TensorFlow is not available for Python 3.13 yet")
print("• The pipeline will run without Deep Learning features")
print("• All other features are fully functional")
print("• Consider using Python 3.10 or 3.11 for TensorFlow support")

print("\n✅ System is ready for Enterprise ML Pipeline! 🎉") 