# Project Structure

## Overview

The Iris ML Pipeline has been reorganized into a professional, enterprise-ready structure following Python packaging best practices.

## Directory Structure

```
iris-ml-pipeline/
├── src/
│   └── iris_pipeline/              # Main package
│       ├── __init__.py
│       ├── config/                 # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py         # Centralized settings
│       ├── data/                   # Data processing
│       │   ├── __init__.py
│       │   └── validation.py       # Data validation & schemas
│       ├── features/               # Feature engineering
│       │   ├── __init__.py
│       │   └── engineering.py      # Feature creation & selection
│       ├── models/                 # Model management
│       │   ├── __init__.py
│       │   ├── training.py         # Model training pipeline
│       │   └── prediction.py       # Model inference
│       └── api/                    # API endpoints
│           ├── __init__.py
│           └── server.py           # FastAPI server
├── apps/
│   └── web_interface.py            # Streamlit web application
├── models/
│   ├── *.pkl                       # Trained model files
│   └── *.joblib                    # Serialized objects
├── data/
│   ├── Iris.csv                    # Raw dataset
│   └── database.sqlite             # Database (if any)
├── scripts/
│   ├── launcher.py                 # Convenient launcher script
│   ├── legacy_train.py             # Original training script
│   ├── demo_prediction.py          # Demo script
│   └── advanced_ml_pipeline.py     # Advanced pipeline script
├── tests/
│   └── test_*.py                   # Unit tests (to be added)
├── docs/
│   └── TECHNICAL_REPORT.md         # Technical documentation
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup (legacy)
├── pyproject.toml                  # Modern Python configuration
├── Makefile                        # Project management commands
├── env.template                    # Environment variables template
├── .gitignore                      # Git ignore patterns
└── README.md                       # Project documentation
```

## Module Organization

### `src/iris_pipeline/`

The main package containing all core functionality:

- **`config/`**: Centralized configuration management with Pydantic settings
- **`data/`**: Data validation, preprocessing, and schema definitions
- **`features/`**: Feature engineering, selection, and transformation
- **`models/`**: Model training, evaluation, and inference
- **`api/`**: REST API endpoints and server configuration

### `apps/`

Standalone applications:

- **`web_interface.py`**: Streamlit web application for interactive ML predictions

### `models/`

Trained model artifacts:

- Classification models (`.pkl` files)
- Regression models (`.pkl` files)
- Encoders and scalers
- Model metadata and information

### `scripts/`

Utility scripts and legacy code:

- **`launcher.py`**: Convenient script to start services
- **`legacy_train.py`**: Original training script (for reference)
- **`demo_prediction.py`**: Demo script for testing models
- **`advanced_ml_pipeline.py`**: Advanced pipeline script

### `docs/`

Documentation:

- **`TECHNICAL_REPORT.md`**: Comprehensive technical documentation
- Other documentation files (to be added)

## Benefits of New Structure

### **Professional Organization**
- Clear separation of concerns
- Standard Python package structure
- Enterprise-ready architecture

### **Maintainability**
- Modular design for easy updates
- Clear dependencies between components
- Standardized imports and exports

### **Deployment Ready**
- Proper package configuration
- Docker support
- CI/CD friendly structure

### **Developer Experience**
- Easy to understand and navigate
- Clear entry points
- Comprehensive documentation

## Usage Examples

### Installing the Package

```bash
# Development installation
pip install -e .

# Production installation
pip install -e .[prod]
```

### Importing Modules

```python
# Configuration
from iris_pipeline.config import settings

# Data processing
from iris_pipeline.data import DataValidator

# Feature engineering
from iris_pipeline.features import FeatureEngineer

# Model training
from iris_pipeline.models import ModelTrainer

# Model prediction
from iris_pipeline.models import ModelPredictor

# API server
from iris_pipeline.api import app
```

### Running Services

```bash
# Using Makefile
make train        # Train models
make api          # Start API server
make web          # Start web interface
make demo         # Start demo environment

# Using launcher script
python scripts/launcher.py demo
```

## Migration from Old Structure

### What Changed

1. **Files moved to proper modules**:
   - `data_validation.py` → `src/iris_pipeline/data/validation.py`
   - `feature_engineering.py` → `src/iris_pipeline/features/engineering.py`
   - `train_models_for_api.py` → `src/iris_pipeline/models/training.py`
   - `api_server.py` → `src/iris_pipeline/api/server.py`
   - `streamlit_app.py` → `apps/web_interface.py`

2. **Added configuration management**:
   - `src/iris_pipeline/config/settings.py` - centralized settings
   - `env.template` - environment variables template

3. **Added build and packaging files**:
   - `setup.py` - package setup
   - `pyproject.toml` - modern Python configuration
   - `Makefile` - project management commands

4. **Improved documentation**:
   - Updated `README.md`
   - Moved technical report to `docs/`
   - Added this structure documentation

### Import Updates

Old imports will need to be updated to use the new structure:

```python
# Old
from data_validation import DataValidator
from feature_engineering import FeatureEngineer

# New
from iris_pipeline.data import DataValidator
from iris_pipeline.features import FeatureEngineer
```

## Next Steps

1. **Update any remaining import statements** in moved files
2. **Add comprehensive unit tests** in `tests/` directory
3. **Set up CI/CD pipeline** using GitHub Actions
4. **Add Docker configuration** for containerized deployment
5. **Create proper documentation** using Sphinx or similar

---

**This structure follows Python packaging best practices and enterprise standards for maintainable, scalable ML pipelines.** 