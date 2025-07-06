"""Configuration settings for Iris ML Pipeline."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

class Settings(BaseSettings):
    """Application settings."""
    
    # Application info
    app_name: str = "Iris ML Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Data settings
    data_dir: Path = DATA_DIR
    raw_data_file: str = "Iris.csv"
    
    # Model settings
    models_dir: Path = MODELS_DIR
    classification_model_file: str = "best_classification_model.pkl"
    regression_model_file: str = "best_regression_model.pkl"
    label_encoder_file: str = "label_encoder_advanced.pkl"
    model_info_file: str = "advanced_model_info.pkl"
    
    # Feature engineering
    feature_engineering_enabled: bool = True
    polynomial_degree: int = 2
    feature_selection_k: int = 10
    scaling_method: str = "standard"  # standard, minmax, robust
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Validation settings
    max_batch_size: int = 100
    request_timeout: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    allowed_origins: List[str] = ["*"]
    api_key_header: str = "X-API-Key"
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_data_path(self) -> Path:
        """Get full path to data directory."""
        return self.data_dir
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to model file."""
        return self.models_dir / model_name
    
    def get_raw_data_path(self) -> Path:
        """Get full path to raw data file."""
        return self.data_dir / self.raw_data_file

# Global settings instance
settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.models_dir, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True) 