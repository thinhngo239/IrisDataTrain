
# BÁO CÁO PHÂN TÍCH DỮ LIỆU IRIS

## 📊 Thông tin tổng quan
- **Số mẫu**: 150
- **Số đặc trưng**: 4
- **Số loài**: 3

## 📈 Thống kê mô tả
       SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
count     150.000000    150.000000     150.000000    150.000000
mean        5.843333      3.054000       3.758667      1.198667
std         0.828066      0.433594       1.764420      0.763161
min         4.300000      2.000000       1.000000      0.100000
25%         5.100000      2.800000       1.600000      0.300000
50%         5.800000      3.000000       4.350000      1.300000
75%         6.400000      3.300000       5.100000      1.800000
max         7.900000      4.400000       6.900000      2.500000

## 🌸 Phân bố các loài
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50

## 🔗 Tương quan giữa các đặc trưng
               SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
SepalLengthCm          1.000        -0.109          0.872         0.818
SepalWidthCm          -0.109         1.000         -0.421        -0.357
PetalLengthCm          0.872        -0.421          1.000         0.963
PetalWidthCm           0.818        -0.357          0.963         1.000

## 📉 Phân tích PCA
- PC1 giải thích: 72.8% phương sai
- PC2 giải thích: 23.0% phương sai
- Tổng cộng 2 thành phần đầu giải thích: 95.8% phương sai

## 🔍 Phát hiện Outliers
- SepalLengthCm: 0 outliers (Z-score), 0 outliers (IQR)
- SepalWidthCm: 1 outliers (Z-score), 4 outliers (IQR)
- PetalLengthCm: 0 outliers (Z-score), 0 outliers (IQR)
- PetalWidthCm: 0 outliers (Z-score), 0 outliers (IQR)

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
