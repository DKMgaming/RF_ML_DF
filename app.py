import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.externals import joblib

# Tải mô hình đã huấn luyện
model = joblib.load('model_trained.pkl')

# Hàm tính tọa độ đích dựa trên thông tin azimuth và khoảng cách
def calculate_destination(lat_receiver, lon_receiver, azimuth, distance):
    R = 6371  # Bán kính Trái đất tính bằng km
    lat1 = np.radians(lat_receiver)
    lon1 = np.radians(lon_receiver)
    azimuth = np.radians(azimuth)

    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance / R) + np.cos(lat1) * np.sin(distance / R) * np.cos(azimuth))
    lon2 = lon1 + np.arctan2(np.sin(azimuth) * np.sin(distance / R) * np.cos(lat1), np.cos(distance / R) - np.sin(lat1) * np.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    return lat2, lon2

# Giao diện người dùng cho việc nhập thông tin và tải lên file
st.title("Dự đoán tọa độ nguồn phát xạ")
st.sidebar.header("Thông tin đầu vào")

# Nhập thông tin cho dự đoán đơn lẻ
lat_receiver = st.sidebar.number_input("Vĩ độ trạm thu", -90.0, 90.0, 14.0)
lon_receiver = st.sidebar.number_input("Kinh độ trạm thu", -180.0, 180.0, 108.0)
h_receiver = st.sidebar.number_input("Chiều cao anten trạm thu (m)", 0, 5000, 50)
frequency = st.sidebar.number_input("Tần số (Hz)", 1000000, 10000000000, 900000000)
azimuth = st.sidebar.number_input("Góc phương vị (Azimuth)", 0, 360, 90)

# Chức năng tải lên file Excel
file = st.sidebar.file_uploader("Tải lên file Excel chứa dữ liệu dự đoán", type=["xls", "xlsx"])

if file:
    # Đọc file Excel
    df = pd.read_excel(file)
    st.write("Dữ liệu đã tải lên:", df)

    if st.button("🔮 Dự đoán cho toàn bộ dữ liệu"):
        try:
            # Tiền xử lý dữ liệu cho toàn bộ file
            df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
            df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))
            
            # Chuẩn bị dữ liệu đầu vào cho mô hình
            input_data = df[['lat_receiver', 'lon_receiver', 'h_receiver', 'frequency', 'azimuth_sin', 'azimuth_cos']].values
            
            # Dự đoán khoảng cách cho toàn bộ dữ liệu
            distance_preds = model.predict(input_data)
            df['predicted_distance'] = distance_preds
            
            # Tính toán tọa độ nguồn phát xạ
            lat_tx = []
            lon_tx = []
            for i, row in df.iterrows():
                lat, lon = calculate_destination(row['lat_receiver'], row['lon_receiver'], row['azimuth'], row['predicted_distance'])
                lat_tx.append(lat)
                lon_tx.append(lon)
            
            df['predicted_lat_tx'] = lat_tx
            df['predicted_lon_tx'] = lon_tx

            # Hiển thị kết quả dự đoán
            st.write("Kết quả dự đoán cho toàn bộ dữ liệu:", df)
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

# Chức năng dự đoán cho một kết quả duy nhất
if st.button("🔮 Dự đoán"):
    try:
        # Tính toán giá trị sin và cos của azimuth
        azimuth_sin = np.sin(np.radians(azimuth))
        azimuth_cos = np.cos(np.radians(azimuth))

        # Đảm bảo cung cấp đủ 7 đặc trưng (bao gồm cả azimuth_sin và azimuth_cos)
        input_data = np.array([[lat_receiver, lon_receiver, h_receiver, frequency, azimuth_sin, azimuth_cos]])
        
        # Dự đoán khoảng cách bằng mô hình đã huấn luyện
        distance_pred = model.predict(input_data)[0]
        st.write(f"Dự đoán khoảng cách đến nguồn phát: {distance_pred:.2f} km")

        # Tính tọa độ nguồn phát xạ
        lat_tx, lon_tx = calculate_destination(lat_receiver, lon_receiver, azimuth, distance_pred)
        st.write(f"Tọa độ nguồn phát xạ dự đoán: (Lat: {lat_tx:.4f}, Lon: {lon_tx:.4f})")

        # Vẽ bản đồ
        m = folium.Map(location=[lat_receiver, lon_receiver], zoom_start=12)
        folium.Marker([lat_receiver, lon_receiver], popup="Trạm thu", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([lat_tx, lon_tx], popup="Nguồn phát xạ", icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700, height=500)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
