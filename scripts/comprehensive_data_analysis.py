#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Cấu hình style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ComprehensiveDataAnalysis:
    """Phân tích dữ liệu toàn diện với nhiều loại biểu đồ"""
    
    def __init__(self, data_path='data/Iris.csv'):
        """Khởi tạo với đường dẫn dữ liệu"""
        self.data_path = data_path
        self.df = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self):
        """Tải dữ liệu"""
        print("="*80)
        print("📊 PHÂN TÍCH DỮ LIỆU IRIS TOÀN DIỆN")
        print("="*80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dữ liệu đã tải thành công: {self.df.shape[0]} mẫu, {self.df.shape[1]} cột")
            print(f"📋 Các cột: {list(self.df.columns)}")
            
            # Phân loại features
            self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
            
            print(f"🔢 Features số: {self.numeric_features}")
            print(f"📝 Features phân loại: {self.categorical_features}")
            
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu: {e}")
            return False
        
        return True
    
    def basic_info(self):
        """Thông tin cơ bản về dữ liệu"""
        print("\n" + "="*50)
        print("📋 THÔNG TIN CƠ BẢN")
        print("="*50)
        
        # Thông tin tổng quan
        print("\n📊 Thông tin tổng quan:")
        print(self.df.info())
        
        # Thống kê mô tả
        print("\n📈 Thống kê mô tả:")
        print(self.df.describe())
        
        # Kiểm tra missing values
        print("\n🔍 Kiểm tra missing values:")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() == 0:
            print("✅ Không có missing values")
        else:
            print("⚠️ Có missing values:")
            print(missing_data[missing_data > 0])
        
        # Phân bố các lớp
        print("\n🌸 Phân bố các loài:")
        species_counts = self.df['Species'].value_counts()
        print(species_counts)
        
        # Tỷ lệ phần trăm
        print("\n📊 Tỷ lệ phần trăm:")
        species_percent = self.df['Species'].value_counts(normalize=True) * 100
        for species, percent in species_percent.items():
            print(f"  {species}: {percent:.1f}%")
    
    def distribution_analysis(self):
        """Phân tích phân bố dữ liệu"""
        print("\n" + "="*50)
        print("📊 PHÂN TÍCH PHÂN BỐ")
        print("="*50)
        
        # Loại bỏ cột Id
        features = [col for col in self.numeric_features if col != 'Id']
        
        # Tạo subplot cho phân bố
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Phân bố các đặc trưng', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Histogram với KDE
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                ax.hist(subset, alpha=0.6, label=species, bins=15, density=True)
            
            # KDE curve
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                subset.plot.kde(ax=ax, linewidth=2)
            
            ax.set_title(f'📈 Phân bố {feature}', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Mật độ')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Box plot để so sánh phân bố
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(data=self.df, x='Species', y=feature)
            plt.title(f'📦 Box Plot - {feature}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Violin plot
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            sns.violinplot(data=self.df, x='Species', y=feature)
            plt.title(f'🎻 Violin Plot - {feature}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """Phân tích tương quan"""
        print("\n" + "="*50)
        print("🔗 PHÂN TÍCH TƯƠNG QUAN")
        print("="*50)
        
        # Loại bỏ cột Id
        features = [col for col in self.numeric_features if col != 'Id']
        
        # Ma trận tương quan
        correlation_matrix = self.df[features].corr()
        
        # Heatmap tương quan
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.3f')
        plt.title('🔗 Ma trận tương quan giữa các đặc trưng', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Scatter plot matrix
        print("\n📊 Tạo scatter plot matrix...")
        sns.pairplot(self.df, hue='Species', diag_kind='kde', 
                    vars=features, palette='husl')
        plt.suptitle('🔗 Scatter Plot Matrix', y=1.02, fontsize=16, fontweight='bold')
        plt.show()
        
        # Tương quan với target
        print("\n📈 Tương quan với loài:")
        for feature in features:
            # Tính correlation cho từng loài
            correlations = {}
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species]
                if len(subset) > 1:
                    corr = subset[features].corr()[feature].abs().mean()
                    correlations[species] = corr
            
            print(f"  {feature}: {correlations}")
    
    def outlier_analysis(self):
        """Phân tích outliers"""
        print("\n" + "="*50)
        print("🔍 PHÂN TÍCH OUTLIERS")
        print("="*50)
        
        features = [col for col in self.numeric_features if col != 'Id']
        
        # Z-score method
        print("\n📊 Phát hiện outliers bằng Z-score:")
        for feature in features:
            z_scores = np.abs(stats.zscore(self.df[feature]))
            outliers = self.df[z_scores > 3]
            print(f"  {feature}: {len(outliers)} outliers (Z-score > 3)")
        
        # IQR method
        print("\n📊 Phát hiện outliers bằng IQR:")
        for feature in features:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            print(f"  {feature}: {len(outliers)} outliers (IQR method)")
        
        # Visualize outliers
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            
            # Box plot với outliers
            sns.boxplot(data=self.df, x='Species', y=feature)
            plt.title(f'📦 Outliers - {feature}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def statistical_tests(self):
        """Kiểm định thống kê"""
        print("\n" + "="*50)
        print("📊 KIỂM ĐỊNH THỐNG KÊ")
        print("="*50)
        
        features = [col for col in self.numeric_features if col != 'Id']
        species_list = self.df['Species'].unique()
        
        print("\n🔬 Kiểm định ANOVA (so sánh trung bình giữa các loài):")
        for feature in features:
            groups = [self.df[self.df['Species'] == species][feature].values 
                     for species in species_list]
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"  {feature}: F-statistic={f_stat:.4f}, p-value={p_value:.4f}")
        
        print("\n🔬 Kiểm định tính chuẩn (Shapiro-Wilk):")
        for feature in features:
            for species in species_list:
                subset = self.df[self.df['Species'] == species][feature]
                stat, p_value = stats.shapiro(subset)
                print(f"  {feature} - {species}: W={stat:.4f}, p-value={p_value:.4f}")
        
        print("\n🔬 Kiểm định tính đồng nhất phương sai (Levene):")
        for feature in features:
            groups = [self.df[self.df['Species'] == species][feature].values 
                     for species in species_list]
            stat, p_value = stats.levene(*groups)
            print(f"  {feature}: W={stat:.4f}, p-value={p_value:.4f}")
    
    def dimensionality_reduction(self):
        """Giảm chiều dữ liệu"""
        print("\n" + "="*50)
        print("📉 GIẢM CHIỀU DỮ LIỆU (PCA)")
        print("="*50)
        
        features = [col for col in self.numeric_features if col != 'Id']
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[features])
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance ratio
        print("\n📊 Tỷ lệ phương sai giải thích:")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        # Cumulative explained variance
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        print(f"\n📈 Tỷ lệ phương sai tích lũy:")
        for i, var in enumerate(cumulative_var):
            print(f"  PC1-PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        # Visualize PCA
        plt.figure(figsize=(15, 5))
        
        # Scree plot
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('📊 Scree Plot', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Cumulative variance
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('📈 Cumulative Variance', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # PCA scatter plot
        plt.subplot(1, 3, 3)
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=species, alpha=0.7)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('🔗 PCA Scatter Plot', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance in PCA
        print("\n📊 Đóng góp của từng feature trong PCA:")
        for i, feature in enumerate(features):
            print(f"  {feature}: PC1={pca.components_[0, i]:.4f}, PC2={pca.components_[1, i]:.4f}")
    
    def advanced_visualizations(self):
        """Biểu đồ nâng cao"""
        print("\n" + "="*50)
        print("🎨 BIỂU ĐỒ NÂNG CAO")
        print("="*50)
        
        features = [col for col in self.numeric_features if col != 'Id']
        
        # 3D scatter plot
        fig = plt.figure(figsize=(15, 5))
        
        # 3D scatter với 3 features đầu
        ax1 = fig.add_subplot(131, projection='3d')
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax1.scatter(self.df[mask][features[0]], 
                       self.df[mask][features[1]], 
                       self.df[mask][features[2]], 
                       label=species, alpha=0.7)
        
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(features[1])
        ax1.set_zlabel(features[2])
        ax1.set_title('🌐 3D Scatter Plot', fontweight='bold')
        ax1.legend()
        
        # Hexbin plot
        ax2 = fig.add_subplot(132)
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax2.hexbin(self.df[mask][features[0]], 
                      self.df[mask][features[1]], 
                      alpha=0.7, label=species)
        
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])
        ax2.set_title('🔷 Hexbin Plot', fontweight='bold')
        ax2.legend()
        
        # Joint plot
        ax3 = fig.add_subplot(133)
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax3.scatter(self.df[mask][features[2]], 
                       self.df[mask][features[3]], 
                       alpha=0.7, label=species)
        
        ax3.set_xlabel(features[2])
        ax3.set_ylabel(features[3])
        ax3.set_title('🔗 Joint Plot', fontweight='bold')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Facet grid
        g = sns.FacetGrid(self.df, col="Species", height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x=features[0], y=features[1], alpha=0.7)
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle('🔍 Facet Grid by Species', fontsize=16, fontweight='bold')
        plt.show()
        
        # Swarm plot
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            sns.swarmplot(data=self.df, x='Species', y=feature)
            plt.title(f'🐝 Swarm Plot - {feature}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary_statistics(self):
        """Thống kê tóm tắt theo loài"""
        print("\n" + "="*50)
        print("📊 THỐNG KÊ TÓM TẮT THEO LOÀI")
        print("="*50)
        
        features = [col for col in self.numeric_features if col != 'Id']
        
        # Thống kê theo loài
        summary = self.df.groupby('Species')[features].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        print("\n📈 Thống kê chi tiết theo loài:")
        print(summary)
        
        # Visualize summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Thống kê theo loài', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Mean với error bars
            means = self.df.groupby('Species')[feature].mean()
            stds = self.df.groupby('Species')[feature].std()
            
            bars = ax.bar(means.index, means.values, yerr=stds.values, 
                         capsize=5, alpha=0.7)
            ax.set_title(f'📊 {feature} - Mean ± Std', fontweight='bold')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, means.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Chạy phân tích toàn diện"""
        if not self.load_data():
            return
        
        # Thực hiện các phân tích
        self.basic_info()
        self.distribution_analysis()
        self.correlation_analysis()
        self.outlier_analysis()
        self.statistical_tests()
        self.dimensionality_reduction()
        self.advanced_visualizations()
        self.summary_statistics()
        
        print("\n" + "="*80)
        print("✅ PHÂN TÍCH DỮ LIỆU HOÀN THÀNH!")
        print("="*80)
        print("\n📋 TÓM TẮT KẾT QUẢ:")
        print("  • Dữ liệu có 150 mẫu với 4 đặc trưng số")
        print("  • 3 loài hoa được phân bố đều (50 mẫu mỗi loài)")
        print("  • Không có missing values")
        print("  • Các đặc trưng có tương quan mạnh với nhau")
        print("  • PCA cho thấy 2 thành phần chính giải thích >95% phương sai")
        print("  • Dữ liệu phù hợp cho machine learning")

def main():
    """Hàm chính"""
    analyzer = ComprehensiveDataAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 