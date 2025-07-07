#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InteractiveReportGenerator:
    """Tạo báo cáo HTML tương tác"""
    
    def __init__(self, data_path='data/Iris.csv', output_file='interactive_report.html'):
        """Khởi tạo với đường dẫn dữ liệu và file output"""
        self.data_path = data_path
        self.output_file = output_file
        self.df = None
        self.numeric_features = None
        
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
    
    def create_distribution_plots(self):
        """Tạo biểu đồ phân bố tương tác"""
        print("📊 Đang tạo biểu đồ phân bố tương tác...")
        
        # Histogram với KDE
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Phân bố {feature}' for feature in self.numeric_features],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2 + 1, i % 2 + 1
            
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=subset,
                        name=f'{species} (Hist)',
                        opacity=0.7,
                        showlegend=False if i > 0 else True
                    ),
                    row=row, col=col
                )
                
                # KDE
                kde_x = np.linspace(subset.min(), subset.max(), 100)
                kde_y = stats.gaussian_kde(subset)(kde_x)
                
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        name=f'{species} (KDE)',
                        mode='lines',
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=row, col=col, secondary_y=True
                )
        
        fig.update_layout(
            title="Phân bố các đặc trưng theo loài",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_plots(self):
        """Tạo biểu đồ tương quan tương tác"""
        print("🔗 Đang tạo biểu đồ tương quan tương tác...")
        
        # Correlation heatmap
        correlation_matrix = self.df[self.numeric_features].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Ma trận tương quan giữa các đặc trưng"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_scatter_matrix(self):
        """Tạo scatter plot matrix tương tác"""
        print("📊 Đang tạo scatter plot matrix tương tác...")
        
        fig = px.scatter_matrix(
            self.df,
            dimensions=self.numeric_features,
            color="Species",
            title="Scatter Plot Matrix",
            opacity=0.7
        )
        
        fig.update_layout(height=800)
        
        return fig
    
    def create_box_plots(self):
        """Tạo box plot tương tác"""
        print("📦 Đang tạo box plot tương tác...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Box Plot - {feature}' for feature in self.numeric_features]
        )
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2 + 1, i % 2 + 1
            
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                
                fig.add_trace(
                    go.Box(
                        y=subset,
                        name=species,
                        boxpoints='outliers',
                        showlegend=False if i > 0 else True
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Box Plot các đặc trưng theo loài",
            height=800
        )
        
        return fig
    
    def create_violin_plots(self):
        """Tạo violin plot tương tác"""
        print("🎻 Đang tạo violin plot tương tác...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Violin Plot - {feature}' for feature in self.numeric_features]
        )
        
        for i, feature in enumerate(self.numeric_features):
            row, col = i // 2 + 1, i % 2 + 1
            
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                
                fig.add_trace(
                    go.Violin(
                        y=subset,
                        name=species,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False if i > 0 else True
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Violin Plot các đặc trưng theo loài",
            height=800
        )
        
        return fig
    
    def create_pca_plots(self):
        """Tạo biểu đồ PCA tương tác"""
        print("📉 Đang tạo biểu đồ PCA tương tác...")
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.numeric_features])
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Tạo DataFrame cho PCA
        pca_df = pd.DataFrame(
            X_pca[:, :2],
            columns=['PC1', 'PC2']
        )
        pca_df['Species'] = self.df['Species'].values
        
        # Scatter plot PCA
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Species',
            title=f"PCA Scatter Plot (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})",
            opacity=0.7
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_3d_scatter(self):
        """Tạo biểu đồ 3D scatter tương tác"""
        print("🌐 Đang tạo biểu đồ 3D scatter tương tác...")
        
        fig = px.scatter_3d(
            self.df,
            x=self.numeric_features[0],
            y=self.numeric_features[1],
            z=self.numeric_features[2],
            color='Species',
            title="3D Scatter Plot",
            opacity=0.7
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_summary_statistics(self):
        """Tạo biểu đồ thống kê tóm tắt tương tác"""
        print("📊 Đang tạo biểu đồ thống kê tóm tắt tương tác...")
        
        # Tính thống kê theo loài
        summary_stats = []
        for feature in self.numeric_features:
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                summary_stats.append({
                    'Feature': feature,
                    'Species': species,
                    'Mean': subset.mean(),
                    'Std': subset.std(),
                    'Min': subset.min(),
                    'Max': subset.max(),
                    'Median': subset.median()
                })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Bar plot với error bars
        fig = px.bar(
            summary_df,
            x='Species',
            y='Mean',
            error_y='Std',
            color='Feature',
            barmode='group',
            title="Thống kê trung bình và độ lệch chuẩn theo loài"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_outlier_analysis(self):
        """Tạo biểu đồ phân tích outliers tương tác"""
        print("🔍 Đang tạo biểu đồ phân tích outliers tương tác...")
        
        # Tính outliers bằng IQR
        outlier_data = []
        for feature in self.numeric_features:
            for species in self.df['Species'].unique():
                subset = self.df[self.df['Species'] == species][feature]
                
                Q1 = subset.quantile(0.25)
                Q3 = subset.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
                
                outlier_data.append({
                    'Feature': feature,
                    'Species': species,
                    'Outliers_Count': len(outliers),
                    'Total_Count': len(subset),
                    'Outlier_Percentage': len(outliers) / len(subset) * 100
                })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        # Bar plot outliers
        fig = px.bar(
            outlier_df,
            x='Feature',
            y='Outlier_Percentage',
            color='Species',
            title="Tỷ lệ outliers theo đặc trưng và loài (%)",
            barmode='group'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_feature_importance(self):
        """Tạo biểu đồ feature importance tương tác"""
        print("🌟 Đang tạo biểu đồ feature importance tương tác...")
        
        # Tính correlation với target (sử dụng label encoding)
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.df['Species'])
        
        feature_importance = []
        for feature in self.numeric_features:
            correlation = np.corrcoef(self.df[feature], y_encoded)[0, 1]
            feature_importance.append({
                'Feature': feature,
                'Correlation': abs(correlation),
                'Correlation_Raw': correlation
            })
        
        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values('Correlation', ascending=True)
        
        # Bar plot feature importance
        fig = px.bar(
            importance_df,
            x='Correlation',
            y='Feature',
            orientation='h',
            title="Feature Importance (Correlation với target)",
            color='Correlation',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def generate_html_report(self):
        """Tạo báo cáo HTML hoàn chỉnh"""
        print("📄 Đang tạo báo cáo HTML...")
        
        # Tạo tất cả biểu đồ
        plots = {
            'Phân bố dữ liệu': self.create_distribution_plots(),
            'Ma trận tương quan': self.create_correlation_plots(),
            'Scatter Plot Matrix': self.create_scatter_matrix(),
            'Box Plot': self.create_box_plots(),
            'Violin Plot': self.create_violin_plots(),
            'Phân tích PCA': self.create_pca_plots(),
            '3D Scatter Plot': self.create_3d_scatter(),
            'Thống kê tóm tắt': self.create_summary_statistics(),
            'Phân tích Outliers': self.create_outlier_analysis(),
            'Feature Importance': self.create_feature_importance()
        }
        
        # Tạo HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo phân tích dữ liệu Iris - Tương tác</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ecf0f1;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .stat-card p {{
            margin: 0;
            font-size: 2em;
            font-weight: bold;
        }}
        .conclusion {{
            background-color: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #27ae60;
            margin: 30px 0;
        }}
        .conclusion h3 {{
            color: #27ae60;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌸 Báo cáo phân tích dữ liệu Iris - Tương tác</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Tổng số mẫu</h3>
                <p>{len(self.df)}</p>
            </div>
            <div class="stat-card">
                <h3>Số đặc trưng</h3>
                <p>{len(self.numeric_features)}</p>
            </div>
            <div class="stat-card">
                <h3>Số loài</h3>
                <p>{len(self.df['Species'].unique())}</p>
            </div>
            <div class="stat-card">
                <h3>Missing values</h3>
                <p>{self.df.isnull().sum().sum()}</p>
            </div>
        </div>
        
        <h2>📊 Phân bố dữ liệu</h2>
        <div class="plot-container">
            {plots['Phân bố dữ liệu'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>🔗 Phân tích tương quan</h2>
        <div class="plot-container">
            {plots['Ma trận tương quan'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <div class="plot-container">
            {plots['Scatter Plot Matrix'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>📦 Phân tích phân bố chi tiết</h2>
        <div class="plot-container">
            {plots['Box Plot'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <div class="plot-container">
            {plots['Violin Plot'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>📉 Phân tích thành phần chính (PCA)</h2>
        <div class="plot-container">
            {plots['Phân tích PCA'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>🌐 Biểu đồ 3D</h2>
        <div class="plot-container">
            {plots['3D Scatter Plot'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>📊 Thống kê tóm tắt</h2>
        <div class="plot-container">
            {plots['Thống kê tóm tắt'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>🔍 Phân tích Outliers</h2>
        <div class="plot-container">
            {plots['Phân tích Outliers'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <h2>🌟 Feature Importance</h2>
        <div class="plot-container">
            {plots['Feature Importance'].to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <div class="conclusion">
            <h3>✅ Kết luận</h3>
            <ul>
                <li><strong>Dữ liệu sạch:</strong> Không có missing values, chất lượng dữ liệu tốt</li>
                <li><strong>Phân bố cân bằng:</strong> Mỗi loài có 50 mẫu, không có bias</li>
                <li><strong>Tương quan mạnh:</strong> Các đặc trưng có tương quan cao với nhau</li>
                <li><strong>Phân tách tốt:</strong> Các loài có thể phân tách được bằng các đặc trưng</li>
                <li><strong>PCA hiệu quả:</strong> 2 thành phần đầu giải thích >95% phương sai</li>
                <li><strong>Phù hợp ML:</strong> Dữ liệu lý tưởng cho machine learning</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        # Lưu file HTML
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Đã tạo báo cáo HTML: {self.output_file}")
    
    def run(self):
        """Chạy toàn bộ quá trình tạo báo cáo"""
        if not self.load_data():
            return
        
        print("="*60)
        print("📄 TẠO BÁO CÁO HTML TƯƠNG TÁC")
        print("="*60)
        
        self.generate_html_report()
        
        print("\n" + "="*60)
        print("✅ HOÀN THÀNH TẠO BÁO CÁO!")
        print("="*60)
        print(f"📄 Báo cáo HTML: {self.output_file}")
        print("🌐 Mở file trong trình duyệt để xem báo cáo tương tác")

def main():
    """Hàm chính"""
    generator = InteractiveReportGenerator()
    generator.run()

if __name__ == "__main__":
    main() 