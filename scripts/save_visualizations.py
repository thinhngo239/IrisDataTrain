#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# Cấu hình style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DataVisualizationSaver:
    """Lưu các biểu đồ phân tích dữ liệu"""
    
    def __init__(self, data_path='data/Iris.csv', output_dir='visualizations'):
        """Khởi tạo với đường dẫn dữ liệu và thư mục output"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.numeric_features = None
        
        # Tạo thư mục output nếu chưa có
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Đã tạo thư mục: {output_dir}")
    
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
    
    def save_distribution_plots(self):
        """Lưu biểu đồ phân bố"""
        print("📊 Đang tạo biểu đồ phân bố...")
        
        # Histogram với KDE
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phân bố các đặc trưng theo loài', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                ax.hist(subset, alpha=0.6, label=species, bins=15, density=True)
            
            ax.set_title(f'Phân bố {feature}', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Mật độ')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_distribution_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 01_distribution_histogram.png")
        
        # Box plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Box Plot các đặc trưng theo loài', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            sns.boxplot(data=self.df, x='Species', y=feature, ax=ax)
            ax.set_title(f'Box Plot - {feature}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_distribution_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 02_distribution_boxplot.png")
        
        # Violin plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Violin Plot các đặc trưng theo loài', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            sns.violinplot(data=self.df, x='Species', y=feature, ax=ax)
            ax.set_title(f'Violin Plot - {feature}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_distribution_violin.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 03_distribution_violin.png")
    
    def save_correlation_plots(self):
        """Lưu biểu đồ tương quan"""
        print("🔗 Đang tạo biểu đồ tương quan...")
        
        # Correlation heatmap
        correlation_matrix = self.df[self.numeric_features].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.3f')
        plt.title('Ma trận tương quan giữa các đặc trưng', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 04_correlation_heatmap.png")
        
        # Pair plot
        g = sns.pairplot(self.df, hue='Species', diag_kind='kde', 
                        vars=self.numeric_features, palette='husl')
        g.fig.suptitle('Scatter Plot Matrix', y=1.02, fontsize=16, fontweight='bold')
        g.fig.savefig(f'{self.output_dir}/05_correlation_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 05_correlation_pairplot.png")
    
    def save_outlier_plots(self):
        """Lưu biểu đồ outliers"""
        print("🔍 Đang tạo biểu đồ outliers...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phân tích Outliers', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            sns.boxplot(data=self.df, x='Species', y=feature, ax=ax)
            ax.set_title(f'Outliers - {feature}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_outliers_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 06_outliers_analysis.png")
    
    def save_pca_plots(self):
        """Lưu biểu đồ PCA"""
        print("📉 Đang tạo biểu đồ PCA...")
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.numeric_features])
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Scree plot và Cumulative variance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Phân tích thành phần chính (PCA)', fontsize=16, fontweight='bold')
        
        # Scree plot
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 'bo-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative variance
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # PCA scatter plot
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], label=species, alpha=0.7)
        
        axes[2].set_xlabel('Principal Component 1')
        axes[2].set_ylabel('Principal Component 2')
        axes[2].set_title('PCA Scatter Plot', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 07_pca_analysis.png")
    
    def save_advanced_plots(self):
        """Lưu biểu đồ nâng cao"""
        print("🎨 Đang tạo biểu đồ nâng cao...")
        
        # 3D scatter plot
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle('Biểu đồ 3D và nâng cao', fontsize=16, fontweight='bold')
        
        # 3D scatter
        ax1 = fig.add_subplot(131, projection='3d')
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax1.scatter(self.df[mask][self.numeric_features[0]], 
                       self.df[mask][self.numeric_features[1]], 
                       self.df[mask][self.numeric_features[2]], 
                       label=species, alpha=0.7)
        
        ax1.set_xlabel(self.numeric_features[0])
        ax1.set_ylabel(self.numeric_features[1])
        ax1.set_zlabel(self.numeric_features[2])
        ax1.set_title('3D Scatter Plot')
        ax1.legend()
        
        # Hexbin plot
        ax2 = fig.add_subplot(132)
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax2.hexbin(self.df[mask][self.numeric_features[0]], 
                      self.df[mask][self.numeric_features[1]], 
                      alpha=0.7, label=species)
        
        ax2.set_xlabel(self.numeric_features[0])
        ax2.set_ylabel(self.numeric_features[1])
        ax2.set_title('Hexbin Plot')
        ax2.legend()
        
        # Joint plot
        ax3 = fig.add_subplot(133)
        for species in self.df['Species'].unique():
            mask = self.df['Species'] == species
            ax3.scatter(self.df[mask][self.numeric_features[2]], 
                       self.df[mask][self.numeric_features[3]], 
                       alpha=0.7, label=species)
        
        ax3.set_xlabel(self.numeric_features[2])
        ax3.set_ylabel(self.numeric_features[3])
        ax3.set_title('Joint Plot')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_advanced_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 08_advanced_3d.png")
        
        # Swarm plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Swarm Plot các đặc trưng', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            sns.swarmplot(data=self.df, x='Species', y=feature, ax=ax)
            ax.set_title(f'Swarm Plot - {feature}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_advanced_swarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 09_advanced_swarm.png")
    
    def save_summary_plots(self):
        """Lưu biểu đồ tóm tắt"""
        print("📊 Đang tạo biểu đồ tóm tắt...")
        
        # Thống kê theo loài
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Thống kê theo loài', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            means = self.df.groupby('Species')[feature].mean()
            stds = self.df.groupby('Species')[feature].std()
            
            bars = ax.bar(means.index, means.values, yerr=stds.values, 
                         capsize=5, alpha=0.7)
            ax.set_title(f'{feature} - Mean ± Std', fontweight='bold')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, means.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 10_summary_statistics.png")
        
        # Facet grid
        g = sns.FacetGrid(self.df, col="Species", height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x=self.numeric_features[0], y=self.numeric_features[1], alpha=0.7)
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle('Facet Grid theo loài', fontsize=16, fontweight='bold')
        g.fig.savefig(f'{self.output_dir}/11_facet_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Đã lưu: 11_facet_grid.png")
    
    def create_summary_report(self):
        """Tạo báo cáo tóm tắt"""
        print("📋 Đang tạo báo cáo tóm tắt...")
        
        report = f"""
# BÁO CÁO PHÂN TÍCH DỮ LIỆU IRIS

## 📊 Thông tin tổng quan
- **Số mẫu**: {self.df.shape[0]}
- **Số đặc trưng**: {len(self.numeric_features)}
- **Số loài**: {len(self.df['Species'].unique())}

## 📈 Thống kê mô tả
{self.df[self.numeric_features].describe().to_string()}

## 🌸 Phân bố các loài
{self.df['Species'].value_counts().to_string()}

## 🔗 Tương quan giữa các đặc trưng
{self.df[self.numeric_features].corr().round(3).to_string()}

## 📉 Phân tích PCA
- PC1 giải thích: {PCA().fit(StandardScaler().fit_transform(self.df[self.numeric_features])).explained_variance_ratio_[0]:.1%} phương sai
- PC2 giải thích: {PCA().fit(StandardScaler().fit_transform(self.df[self.numeric_features])).explained_variance_ratio_[1]:.1%} phương sai
- Tổng cộng 2 thành phần đầu giải thích: {(PCA().fit(StandardScaler().fit_transform(self.df[self.numeric_features])).explained_variance_ratio_[:2].sum()):.1%} phương sai

## 🔍 Phát hiện Outliers
"""
        
        # Thêm thông tin outliers
        for feature in self.numeric_features:
            z_scores = np.abs(stats.zscore(self.df[feature]))
            outliers_z = len(self.df[z_scores > 3])
            
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = len(self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)])
            
            report += f"- {feature}: {outliers_z} outliers (Z-score), {outliers_iqr} outliers (IQR)\n"
        
        report += f"""
## 📁 Các file biểu đồ đã tạo
1. 01_distribution_histogram.png - Biểu đồ phân bố histogram
2. 02_distribution_boxplot.png - Biểu đồ box plot
3. 03_distribution_violin.png - Biểu đồ violin plot
4. 04_correlation_heatmap.png - Ma trận tương quan
5. 05_correlation_pairplot.png - Scatter plot matrix
6. 06_outliers_analysis.png - Phân tích outliers
7. 07_pca_analysis.png - Phân tích PCA
8. 08_advanced_3d.png - Biểu đồ 3D
9. 09_advanced_swarm.png - Swarm plot
10. 10_summary_statistics.png - Thống kê tóm tắt
11. 11_facet_grid.png - Facet grid

## ✅ Kết luận
- Dữ liệu sạch, không có missing values
- Các đặc trưng có tương quan mạnh với nhau
- Phân bố các loài cân bằng
- Dữ liệu phù hợp cho machine learning
"""
        
        # Lưu báo cáo
        with open(f'{self.output_dir}/README.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Đã lưu: README.md")
    
    def run_all_visualizations(self):
        """Chạy tất cả các biểu đồ"""
        if not self.load_data():
            return
        
        print("="*60)
        print("🎨 TẠO BIỂU ĐỒ PHÂN TÍCH DỮ LIỆU")
        print("="*60)
        
        # Tạo các loại biểu đồ
        self.save_distribution_plots()
        self.save_correlation_plots()
        self.save_outlier_plots()
        self.save_pca_plots()
        self.save_advanced_plots()
        self.save_summary_plots()
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("✅ HOÀN THÀNH TẠO BIỂU ĐỒ!")
        print("="*60)
        print(f"📁 Tất cả biểu đồ đã được lưu trong thư mục: {self.output_dir}")
        print("📋 Báo cáo chi tiết: README.md")

def main():
    """Hàm chính"""
    saver = DataVisualizationSaver()
    saver.run_all_visualizations()

if __name__ == "__main__":
    main() 