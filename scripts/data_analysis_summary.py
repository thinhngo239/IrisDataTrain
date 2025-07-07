#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisSummary:
    """Tóm tắt phân tích dữ liệu"""
    
    def __init__(self, data_path='data/Iris.csv'):
        """Khởi tạo với đường dẫn dữ liệu"""
        self.data_path = data_path
        self.df = None
        self.numeric_features = None
        self.insights = []
        
    def load_data(self):
        """Tải dữ liệu"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns.tolist() if col != 'Id']
            print(f"✅ Dữ liệu đã tải: {self.df.shape[0]} mẫu, {self.df.shape[1]} cột")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            return False
    
    def analyze_data_quality(self):
        """Phân tích chất lượng dữ liệu"""
        print("\n" + "="*60)
        print("🔍 PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU")
        print("="*60)
        
        # Missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count == 0:
            print("✅ Không có missing values")
            self.insights.append("Dữ liệu sạch, không có missing values")
        else:
            print(f"⚠️ Có {missing_count} missing values")
            self.insights.append(f"Có {missing_count} missing values cần xử lý")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("✅ Không có dữ liệu trùng lặp")
            self.insights.append("Không có dữ liệu trùng lặp")
        else:
            print(f"⚠️ Có {duplicates} dòng trùng lặp")
            self.insights.append(f"Có {duplicates} dòng trùng lặp")
        
        # Data types
        print(f"📊 Kiểu dữ liệu:")
        for col, dtype in self.df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Value ranges
        print(f"\n📈 Phạm vi giá trị:")
        for feature in self.numeric_features:
            min_val = self.df[feature].min()
            max_val = self.df[feature].max()
            print(f"  {feature}: {min_val:.2f} - {max_val:.2f}")
    
    def analyze_distribution(self):
        """Phân tích phân bố dữ liệu"""
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH PHÂN BỐ DỮ LIỆU")
        print("="*60)
        
        # Class distribution
        species_counts = self.df['Species'].value_counts()
        print("🌸 Phân bố các loài:")
        for species, count in species_counts.items():
            percentage = count / len(self.df) * 100
            print(f"  {species}: {count} mẫu ({percentage:.1f}%)")
        
        if len(species_counts) == len(species_counts.unique()):
            print("✅ Phân bố cân bằng giữa các loài")
            self.insights.append("Phân bố cân bằng giữa các loài")
        else:
            print("⚠️ Phân bố không cân bằng")
            self.insights.append("Phân bố không cân bằng giữa các loài")
        
        # Feature distributions
        print(f"\n📈 Thống kê mô tả:")
        stats_df = self.df[self.numeric_features].describe()
        print(stats_df.round(3))
        
        # Normality test
        print(f"\n🔬 Kiểm định tính chuẩn (Shapiro-Wilk):")
        for feature in self.numeric_features:
            stat, p_value = stats.shapiro(self.df[feature])
            is_normal = p_value > 0.05
            status = "✅ Chuẩn" if is_normal else "❌ Không chuẩn"
            print(f"  {feature}: p-value={p_value:.4f} ({status})")
            
            if is_normal:
                self.insights.append(f"{feature} có phân bố chuẩn")
            else:
                self.insights.append(f"{feature} không có phân bố chuẩn")
    
    def analyze_correlations(self):
        """Phân tích tương quan"""
        print("\n" + "="*60)
        print("🔗 PHÂN TÍCH TƯƠNG QUAN")
        print("="*60)
        
        # Correlation matrix
        correlation_matrix = self.df[self.numeric_features].corr()
        print("📊 Ma trận tương quan:")
        print(correlation_matrix.round(3))
        
        # Strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            print(f"\n🔗 Tương quan mạnh (|r| > 0.7):")
            for corr in strong_correlations:
                print(f"  {corr['feature1']} - {corr['feature2']}: {corr['correlation']:.3f}")
                self.insights.append(f"Tương quan mạnh giữa {corr['feature1']} và {corr['feature2']} ({corr['correlation']:.3f})")
        else:
            print("✅ Không có tương quan mạnh")
            self.insights.append("Không có tương quan mạnh giữa các đặc trưng")
    
    def analyze_outliers(self):
        """Phân tích outliers"""
        print("\n" + "="*60)
        print("🔍 PHÂN TÍCH OUTLIERS")
        print("="*60)
        
        outlier_summary = []
        
        for feature in self.numeric_features:
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[feature]))
            outliers_z = len(self.df[z_scores > 3])
            
            # IQR method
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = len(self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)])
            
            outlier_summary.append({
                'feature': feature,
                'z_score_outliers': outliers_z,
                'iqr_outliers': outliers_iqr,
                'total_samples': len(self.df)
            })
            
            print(f"📊 {feature}:")
            print(f"  Z-score outliers: {outliers_z} ({outliers_z/len(self.df)*100:.1f}%)")
            print(f"  IQR outliers: {outliers_iqr} ({outliers_iqr/len(self.df)*100:.1f}%)")
        
        # Overall assessment
        total_outliers_z = sum(item['z_score_outliers'] for item in outlier_summary)
        total_outliers_iqr = sum(item['iqr_outliers'] for item in outlier_summary)
        
        if total_outliers_z == 0:
            print("✅ Không có outliers đáng kể (Z-score)")
            self.insights.append("Không có outliers đáng kể theo Z-score")
        else:
            print(f"⚠️ Tổng cộng {total_outliers_z} outliers (Z-score)")
            self.insights.append(f"Có {total_outliers_z} outliers theo Z-score")
        
        if total_outliers_iqr == 0:
            print("✅ Không có outliers đáng kể (IQR)")
            self.insights.append("Không có outliers đáng kể theo IQR")
        else:
            print(f"⚠️ Tổng cộng {total_outliers_iqr} outliers (IQR)")
            self.insights.append(f"Có {total_outliers_iqr} outliers theo IQR")
    
    def analyze_separability(self):
        """Phân tích khả năng phân tách giữa các loài"""
        print("\n" + "="*60)
        print("🎯 PHÂN TÍCH KHẢ NĂNG PHÂN TÁCH")
        print("="*60)
        
        # ANOVA test
        print("🔬 Kiểm định ANOVA (so sánh trung bình giữa các loài):")
        species_list = self.df['Species'].unique()
        
        for feature in self.numeric_features:
            groups = [self.df[self.df['Species'] == species][feature].values 
                     for species in species_list]
            f_stat, p_value = stats.f_oneway(*groups)
            
            is_significant = p_value < 0.05
            status = "✅ Có ý nghĩa" if is_significant else "❌ Không có ý nghĩa"
            print(f"  {feature}: F={f_stat:.2f}, p={p_value:.4f} ({status})")
            
            if is_significant:
                self.insights.append(f"{feature} có khả năng phân tách tốt giữa các loài")
            else:
                self.insights.append(f"{feature} không có khả năng phân tách tốt")
        
        # Effect size (eta-squared)
        print(f"\n📊 Hiệu lực phân tách (Effect size):")
        for feature in self.numeric_features:
            groups = [self.df[self.df['Species'] == species][feature].values 
                     for species in species_list]
            
            # Calculate eta-squared
            grand_mean = np.mean([np.mean(group) for group in groups])
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
            ss_total = sum((val - grand_mean)**2 for group in groups for val in group)
            eta_squared = ss_between / ss_total
            
            effect_size = "Lớn" if eta_squared > 0.14 else "Trung bình" if eta_squared > 0.06 else "Nhỏ"
            print(f"  {feature}: η² = {eta_squared:.3f} ({effect_size})")
    
    def analyze_dimensionality(self):
        """Phân tích chiều dữ liệu"""
        print("\n" + "="*60)
        print("📉 PHÂN TÍCH CHIỀU DỮ LIỆU")
        print("="*60)
        
        # PCA analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.numeric_features])
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        print("📊 Tỷ lệ phương sai giải thích:")
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        
        for i, (ratio, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_var)):
            print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%) - Tích lũy: {cum_var:.3f} ({cum_var*100:.1f}%)")
        
        # Determine optimal number of components
        optimal_components = np.argmax(cumulative_var >= 0.95) + 1
        print(f"\n🎯 Số thành phần tối ưu (95% phương sai): {optimal_components}")
        
        if optimal_components <= 2:
            print("✅ Có thể giảm chiều dữ liệu hiệu quả")
            self.insights.append(f"Có thể giảm từ {len(self.numeric_features)} xuống {optimal_components} chiều")
        else:
            print("⚠️ Khó giảm chiều dữ liệu")
            self.insights.append("Khó giảm chiều dữ liệu hiệu quả")
        
        # Feature importance in PCA
        print(f"\n🌟 Đóng góp của từng feature trong PC1:")
        for i, feature in enumerate(self.numeric_features):
            print(f"  {feature}: {pca.components_[0, i]:.3f}")
    
    def generate_recommendations(self):
        """Tạo khuyến nghị"""
        print("\n" + "="*60)
        print("💡 KHUYẾN NGHỊ")
        print("="*60)
        
        recommendations = []
        
        # Data quality recommendations
        if "missing values" not in " ".join(self.insights).lower():
            recommendations.append("✅ Dữ liệu sạch, không cần xử lý missing values")
        else:
            recommendations.append("⚠️ Cần xử lý missing values trước khi phân tích")
        
        # Feature engineering recommendations
        if any("tương quan mạnh" in insight for insight in self.insights):
            recommendations.append("🔧 Có thể tạo features mới từ các đặc trưng tương quan mạnh")
        
        # Model recommendations
        if any("phân tách tốt" in insight for insight in self.insights):
            recommendations.append("🎯 Các đặc trưng có khả năng phân tách tốt, phù hợp cho classification")
        
        if any("giảm từ" in insight for insight in self.insights):
            recommendations.append("📉 Có thể sử dụng PCA để giảm chiều dữ liệu")
        
        # Algorithm recommendations
        if "phân bố chuẩn" in " ".join(self.insights):
            recommendations.append("📊 Dữ liệu có phân bố chuẩn, phù hợp với các thuật toán parametric")
        else:
            recommendations.append("📊 Dữ liệu không chuẩn, nên sử dụng thuật toán non-parametric")
        
        # Validation recommendations
        if "cân bằng" in " ".join(self.insights):
            recommendations.append("⚖️ Phân bố cân bằng, có thể sử dụng accuracy làm metric chính")
        else:
            recommendations.append("⚖️ Phân bố không cân bằng, nên sử dụng F1-score hoặc precision/recall")
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    def create_summary_report(self):
        """Tạo báo cáo tóm tắt"""
        print("\n" + "="*60)
        print("📋 BÁO CÁO TÓM TẮT")
        print("="*60)
        
        # Basic statistics
        print(f"📊 Thống kê cơ bản:")
        print(f"  • Tổng số mẫu: {len(self.df)}")
        print(f"  • Số đặc trưng: {len(self.numeric_features)}")
        print(f"  • Số loài: {len(self.df['Species'].unique())}")
        print(f"  • Missing values: {self.df.isnull().sum().sum()}")
        print(f"  • Duplicates: {self.df.duplicated().sum()}")
        
        # Key insights
        print(f"\n🔍 Insights chính:")
        for i, insight in enumerate(self.insights[:10], 1):  # Top 10 insights
            print(f"  {i}. {insight}")
        
        # Data quality score
        quality_score = 0
        if self.df.isnull().sum().sum() == 0:
            quality_score += 25
        if self.df.duplicated().sum() == 0:
            quality_score += 25
        if len(self.df['Species'].value_counts()) == len(self.df['Species'].unique()):
            quality_score += 25
        if any("phân tách tốt" in insight for insight in self.insights):
            quality_score += 25
        
        print(f"\n📈 Điểm chất lượng dữ liệu: {quality_score}/100")
        
        if quality_score >= 80:
            print("✅ Dữ liệu chất lượng cao, phù hợp cho machine learning")
        elif quality_score >= 60:
            print("⚠️ Dữ liệu chất lượng trung bình, cần một số xử lý")
        else:
            print("❌ Dữ liệu chất lượng thấp, cần xử lý nhiều")
        
        # ML readiness
        print(f"\n🤖 Sẵn sàng cho Machine Learning:")
        readiness_factors = []
        
        if self.df.isnull().sum().sum() == 0:
            readiness_factors.append("Dữ liệu sạch")
        if len(self.df['Species'].value_counts()) == len(self.df['Species'].unique()):
            readiness_factors.append("Phân bố cân bằng")
        if any("phân tách tốt" in insight for insight in self.insights):
            readiness_factors.append("Đặc trưng phân tách tốt")
        if len(self.df) >= 100:
            readiness_factors.append("Đủ dữ liệu")
        
        if len(readiness_factors) >= 3:
            print("✅ Sẵn sàng cho machine learning")
            for factor in readiness_factors:
                print(f"  ✓ {factor}")
        else:
            print("⚠️ Cần cải thiện trước khi áp dụng machine learning")
    
    def run_complete_analysis(self):
        """Chạy phân tích toàn diện"""
        if not self.load_data():
            return
        
        print("="*80)
        print("📊 PHÂN TÍCH DỮ LIỆU IRIS - TÓM TẮT TOÀN DIỆN")
        print("="*80)
        
        # Thực hiện các phân tích
        self.analyze_data_quality()
        self.analyze_distribution()
        self.analyze_correlations()
        self.analyze_outliers()
        self.analyze_separability()
        self.analyze_dimensionality()
        self.generate_recommendations()
        self.create_summary_report()
        
        print("\n" + "="*80)
        print("✅ PHÂN TÍCH HOÀN THÀNH!")
        print("="*80)
        print("📁 Các file đã tạo:")
        print("  • visualizations/ - Thư mục chứa biểu đồ tĩnh")
        print("  • interactive_report.html - Báo cáo tương tác")
        print("  • visualizations/README.md - Báo cáo chi tiết")

def main():
    """Hàm chính"""
    analyzer = DataAnalysisSummary()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 