import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Enum cho các kiểu dữ liệu được hỗ trợ"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATE = "date"

@dataclass
class ColumnSchema:
    """Schema cho từng cột dữ liệu"""
    name: str
    data_type: DataType
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    null_allowed: bool = False
    description: str = ""

class DataValidator:
    """Class để validation dữ liệu"""
    
    def __init__(self, schema: List[ColumnSchema]):
        self.schema = {col.name: col for col in schema}
        self.validation_results = []
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate toàn bộ DataFrame
        
        Args:
            df: DataFrame cần validate
            
        Returns:
            Dict chứa kết quả validation
        """
        logger.info("Bắt đầu validation dữ liệu...")
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'schema_columns': len(self.schema)
            }
        }
        
        # Kiểm tra cột bắt buộc
        missing_columns = set(self.schema.keys()) - set(df.columns)
        if missing_columns:
            error_msg = f"Thiếu các cột bắt buộc: {list(missing_columns)}"
            results['errors'].append(error_msg)
            results['is_valid'] = False
            logger.error(error_msg)
            
        # Kiểm tra cột thừa
        extra_columns = set(df.columns) - set(self.schema.keys())
        if extra_columns:
            warning_msg = f"Có các cột không mong muốn: {list(extra_columns)}"
            results['warnings'].append(warning_msg)
            logger.warning(warning_msg)
            
        # Validate từng cột
        for col_name, col_schema in self.schema.items():
            if col_name in df.columns:
                series = df[col_name]
                assert isinstance(series, pd.Series)
                col_results = self._validate_column(series, col_schema)
                if col_results['errors']:
                    results['errors'].extend(col_results['errors'])
                    results['is_valid'] = False
                if col_results['warnings']:
                    results['warnings'].extend(col_results['warnings'])
                    
        # Kiểm tra data types
        type_results = self._validate_data_types(df)
        if type_results['errors']:
            results['errors'].extend(type_results['errors'])
            results['is_valid'] = False
            
        # Kiểm tra duplicates
        duplicate_results = self._check_duplicates(df)
        if duplicate_results['warnings']:
            results['warnings'].extend(duplicate_results['warnings'])
            
        logger.info(f"Validation hoàn thành. Kết quả: {'PASS' if results['is_valid'] else 'FAIL'}")
        return results
    
    def _validate_column(self, series: pd.Series, schema: ColumnSchema) -> Dict[str, List[str]]:
        """Validate một cột cụ thể"""
        results = {'errors': [], 'warnings': []}
        col_name = schema.name
        
        # Kiểm tra null values
        null_count = series.isnull().sum()
        if null_count > 0:
            if not schema.null_allowed:
                results['errors'].append(f"Cột '{col_name}' có {null_count} giá trị null không được phép")
            else:
                results['warnings'].append(f"Cột '{col_name}' có {null_count} giá trị null")
                
        # Chỉ validate dữ liệu không null
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return results
            
        # Kiểm tra range cho numeric data
        if schema.data_type == DataType.NUMERIC:
            if schema.min_value is not None:
                below_min = (non_null_series < schema.min_value).sum()
                if below_min > 0:
                    results['errors'].append(f"Cột '{col_name}' có {below_min} giá trị < {schema.min_value}")
                    
            if schema.max_value is not None:
                above_max = (non_null_series > schema.max_value).sum()
                if above_max > 0:
                    results['errors'].append(f"Cột '{col_name}' có {above_max} giá trị > {schema.max_value}")
                    
        # Kiểm tra allowed values
        if schema.allowed_values is not None:
            invalid_values = set(non_null_series.unique()) - set(schema.allowed_values)
            if invalid_values:
                results['errors'].append(f"Cột '{col_name}' có giá trị không hợp lệ: {list(invalid_values)}")
                
        return results
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Kiểm tra kiểu dữ liệu của các cột"""
        results = {'errors': []}
        
        for col_name, col_schema in self.schema.items():
            if col_name not in df.columns:
                continue
                
            column = df[col_name]
            
            if col_schema.data_type == DataType.NUMERIC:
                if not pd.api.types.is_numeric_dtype(column):
                    results['errors'].append(f"Cột '{col_name}' phải là kiểu số")
                    
            elif col_schema.data_type == DataType.CATEGORICAL:
                if not pd.api.types.is_object_dtype(column):
                    results['errors'].append(f"Cột '{col_name}' phải là kiểu categorical/string")
                    
        return results
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Kiểm tra dữ liệu trùng lặp"""
        results = {'warnings': []}
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Có {duplicate_count} hàng trùng lặp trong dữ liệu")
            
        return results
    
    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Tạo báo cáo validation"""
        report = []
        report.append("=" * 60)
        report.append("BÁO CÁO VALIDATION DỮ LIỆU")
        report.append("=" * 60)
        
        # Thông tin tổng quan
        summary = validation_results['summary']
        report.append(f"Tổng số hàng: {summary['total_rows']}")
        report.append(f"Tổng số cột: {summary['total_columns']}")
        report.append(f"Cột theo schema: {summary['schema_columns']}")
        
        # Kết quả validation
        status = "✅ PASS" if validation_results['is_valid'] else "❌ FAIL"
        report.append(f"Kết quả: {status}")
        
        # Lỗi
        if validation_results['errors']:
            report.append(f"\n🚨 LỖI ({len(validation_results['errors'])} lỗi):")
            for i, error in enumerate(validation_results['errors'], 1):
                report.append(f"  {i}. {error}")
                
        # Cảnh báo
        if validation_results['warnings']:
            report.append(f"\n⚠️  CẢNH BÁO ({len(validation_results['warnings'])} cảnh báo):")
            for i, warning in enumerate(validation_results['warnings'], 1):
                report.append(f"  {i}. {warning}")
                
        report.append("=" * 60)
        return "\n".join(report)

# Schema cho dữ liệu Iris
IRIS_SCHEMA = [
    ColumnSchema(
        name="SepalLengthCm",
        data_type=DataType.NUMERIC,
        min_value=0.0,
        max_value=10.0,
        description="Chiều dài đài hoa (cm)"
    ),
    ColumnSchema(
        name="SepalWidthCm",
        data_type=DataType.NUMERIC,
        min_value=0.0,
        max_value=10.0,
        description="Chiều rộng đài hoa (cm)"
    ),
    ColumnSchema(
        name="PetalLengthCm",
        data_type=DataType.NUMERIC,
        min_value=0.0,
        max_value=10.0,
        description="Chiều dài cánh hoa (cm)"
    ),
    ColumnSchema(
        name="PetalWidthCm",
        data_type=DataType.NUMERIC,
        min_value=0.0,
        max_value=10.0,
        description="Chiều rộng cánh hoa (cm)"
    ),
    ColumnSchema(
        name="Species",
        data_type=DataType.CATEGORICAL,
        allowed_values=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        description="Loài hoa iris"
    )
]

def validate_iris_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dữ liệu Iris
    
    Args:
        df: DataFrame chứa dữ liệu Iris
        
    Returns:
        Dict chứa kết quả validation
    """
    validator = DataValidator(IRIS_SCHEMA)
    return validator.validate_dataframe(df)

def main():
    """Demo validation"""
    print("=" * 60)
    print("DEMO DATA VALIDATION")
    print("=" * 60)
    
    # Đọc dữ liệu
    try:
        df = pd.read_csv('data/Iris.csv')
        print(f"✓ Đọc dữ liệu thành công: {df.shape}")
        
        # Loại bỏ cột Id nếu có
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
            print("✓ Đã loại bỏ cột Id")
            
        # Validate dữ liệu
        results = validate_iris_data(df)
        
        # Tạo validator và generate report
        validator = DataValidator(IRIS_SCHEMA)
        report = validator.generate_report(results)
        print(report)
        
        # Thống kê chi tiết
        print("\n📊 THỐNG KÊ CHI TIẾT:")
        print(f"Kiểu dữ liệu các cột:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
            
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        for col in missing.index:
            print(f"  {col}: {missing[col]}")
            
        print(f"\nGiá trị unique:")
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                print(f"  {col}: {list(unique_vals)}")
            else:
                print(f"  {col}: {df[col].nunique()} giá trị unique")
                
    except FileNotFoundError:
        print("❌ Không tìm thấy file data/Iris.csv")
        return
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH VALIDATION")
    print("=" * 60)

if __name__ == "__main__":
    main() 