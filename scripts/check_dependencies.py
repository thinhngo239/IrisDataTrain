#!/usr/bin/env python3
"""
Check and install all dependencies for the Iris ML Pipeline project
"""

import subprocess
import sys
import importlib

print("🔍 Checking Python dependencies for Iris ML Pipeline")
print("=" * 60)
print(f"Python version: {sys.version}")
print("=" * 60)

# Define all required packages
packages = {
    # Core ML Libraries
    'pandas': 'pandas>=1.3.0',
    'numpy': 'numpy>=1.21.0',
    'sklearn': 'scikit-learn>=1.0.0',
    'matplotlib': 'matplotlib>=3.4.0',
    'seaborn': 'seaborn>=0.11.0',
    
    # Advanced ML Models
    'xgboost': 'xgboost>=1.6.0',
    
    # Deep Learning
    'tensorflow': 'tensorflow>=2.8.0',
    
    # Web Framework & API
    'fastapi': 'fastapi>=0.70.0',
    'uvicorn': 'uvicorn[standard]>=0.15.0',
    'pydantic': 'pydantic>=1.8.0',
    
    # Web Interface
    'streamlit': 'streamlit>=1.2.0',
    'plotly': 'plotly>=5.0.0',
    
    # Utilities
    'requests': 'requests>=2.26.0',
    'joblib': 'joblib>=1.1.0',
    'multipart': 'python-multipart>=0.0.5',
}

# Check each package
missing_packages = []
installed_packages = []

for module_name, package_spec in packages.items():
    try:
        importlib.import_module(module_name)
        installed_packages.append(module_name)
        print(f"✅ {module_name:<20} - Installed")
    except ImportError:
        missing_packages.append(package_spec)
        print(f"❌ {module_name:<20} - Not installed")

print("\n" + "=" * 60)
print(f"📊 Summary: {len(installed_packages)}/{len(packages)} packages installed")
print("=" * 60)

if missing_packages:
    print(f"\n⚠️ Missing {len(missing_packages)} packages:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    
    print("\n🔧 Installation options:")
    print("1. Install all missing packages:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("\n2. Install from requirements.txt (recommended):")
    print("   pip install -r requirements.txt")
    
    # Ask user if they want to install
    response = input("\n🤔 Do you want to install missing packages now? (y/n): ")
    if response.lower() == 'y':
        print("\n🚀 Installing missing packages...")
        try:
            # Install from requirements.txt for consistency
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("\n✅ Installation completed successfully!")
        except subprocess.CalledProcessError:
            print("\n❌ Installation failed. Please install manually:")
            print("   pip install -r requirements.txt")
else:
    print("\n✅ All dependencies are installed!")
    print("🚀 You can now run the Enterprise ML Pipeline:")
    print("   python scripts/run_enterprise_pipeline.py")
    print("   or: make enterprise") 