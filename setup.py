"""Setup configuration for Iris ML Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements = read_requirements('requirements.txt')

setup(
    name="iris-ml-pipeline",
    version="1.0.0",
    author="ML Team",
    author_email="ml-team@company.com",
    description="A complete machine learning pipeline for Iris flower classification and regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/iris-ml-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/company/iris-ml-pipeline/issues",
        "Documentation": "https://iris-ml-pipeline.readthedocs.io/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "monitoring": [
            "prometheus-client>=0.11.0",
            "structlog>=21.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iris-api=iris_pipeline.api.server:main",
            "iris-train=iris_pipeline.models.training:main",
            "iris-web=apps.web_interface:main",
        ],
    },
    include_package_data=True,
    package_data={
        "iris_pipeline": ["config/*.yaml", "config/*.json"],
    },
    zip_safe=False,
    keywords="machine-learning, iris, classification, regression, fastapi, streamlit",
) 