import pandas as pd
import re
from sklearn.impute import KNNImputer

def remove_units(column):
    # Xóa tất cả ký tự không phải số
    column_cleaned = column.astype(str).apply(lambda x: re.sub(r'[^0-9.]', '', x))
    # Thay thế giá trị rỗng ('') bằng NaN
    column_cleaned.replace('', float('nan'), inplace=True)
    # Chuyển sang kiểu float
    return column_cleaned.astype(float)

def extract_year(date_column):
    return pd.to_datetime(date_column, errors='coerce').dt.year  # Trích xuất năm

def process_l2_cache(value):
    match = re.match(r'(\d+)KB(?:\(x(\d+)\))?', value)
    if match:
        size = int(match.group(1))  # Extract the numeric part
        multiplier = int(match.group(2)) if match.group(2) else 1  # Extract multiplier (default is 1)
        return size * multiplier  # Return total size in KB
    return 0  # Default to 0 if parsing fails

def fill_missing_values(df):
    """
    Điền giá trị khuyết thiếu:
    - Với cột số: sử dụng KNN Imputer thay vì median
    - Với cột phân loại: thay bằng mode
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Điền giá trị cho cột phân loại bằng mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Áp dụng KNN Imputer cho cột số
    imputer = KNNImputer(n_neighbors=5)  # Chọn 5 hàng gần nhất để tính toán
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def process_missing_values(file_path, output_file):
    # Load dữ liệu
    df = pd.read_csv(file_path)
    
    # Tính phần trăm giá trị thiếu
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("Missing percentage per column:")
    print(missing_percentage.to_string())  # In đầy đủ danh sách

    # Chỉ giữ lại các cột có ít hơn 20% giá trị khuyết thiếu
    selected_columns = missing_percentage[missing_percentage < 20].index
    new_gpu_data = df[selected_columns]

    # Xử lý các cột số học
    columns_to_clean = ["Memory_Bandwidth", "Memory_Bus", "Memory_Speed", 
                         "Core_Speed", "Texture_Rate", "Pixel_Rate", "Process", "Max_Power"]
    
    for col in columns_to_clean:
        if col in new_gpu_data.columns:
            new_gpu_data[col] = remove_units(new_gpu_data[col])

    new_gpu_data["L2_Cache"] = new_gpu_data["L2_Cache"].astype(str).apply(process_l2_cache)
    
    # Trích xuất năm từ cột Release_Date
    if "Release_Date" in new_gpu_data.columns:
        new_gpu_data["Release_Year"] = extract_year(new_gpu_data["Release_Datae"])
    
    # Điền giá trị khuyết thiếu bằng KNN Imputer
    new_gpu_data = fill_missing_values(new_gpu_data)

    # Xuất dữ liệu ra file mới
    new_gpu_data.to_csv(output_file, index=False)
    
    # Hiển thị thông tin dataset sau khi xử lý
    print("\nStructure of cleaned dataset:")
    print(new_gpu_data.info())
    
    return new_gpu_data

# Đường dẫn file
input_file = "data/All_GPUs.csv"  
output_file = "data/Cleaned_GPUs_KNN.csv"  

# Chạy hàm xử lý dữ liệu
processed_data = process_missing_values(input_file, output_file)
