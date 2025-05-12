import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from io import BytesIO
from math import radians, degrees, sin, cos, atan2, sqrt
import folium
from streamlit_folium import st_folium

# --- Hàm phụ cho Triangulation ---
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    azimuth = (degrees(atan2(x, y)) + 360) % 360
    return azimuth

def calculate_distance(lat1, lon1, lat2, lon2):
    # Tính khoảng cách giữa 2 điểm (km)
    R = 6371.0  # Bán kính Trái Đất (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def triangulation(lat1, lon1, az1, lat2, lon2, az2):
    # Tính toán tọa độ nguồn phát từ 2 trạm thu và các góc phương vị
    # Chuyển đổi azimuth và tọa độ sang radian
    az1 = radians(az1)
    az2 = radians(az2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Tính khoảng cách giữa 2 trạm thu
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    
    # Công thức để tính toán tọa độ nguồn phát xạ
    d = distance  # Khoảng cách giữa 2 trạm thu
    a1 = az1  # Phương vị của trạm thu 1
    a2 = az2  # Phương vị của trạm thu 2

    x = (sin(a1) - sin(a2)) * d / (cos(a1) - cos(a2))
    y = (sin(a2) * cos(a1) - cos(a2) * sin(a1)) * d / (cos(a1) - cos(a2))

    # Tính toán vị trí nguồn phát
    lat3 = lat1 + y / 6371.0  # Độ vĩ độ của nguồn phát
    lon3 = lon1 + x / (6371.0 * cos(lat1))  # Độ kinh độ của nguồn phát

    # Chuyển tọa độ trở lại độ
    lat3 = degrees(lat3)
    lon3 = degrees(lon3)
    
    return lat3, lon3

# --- Tab 1: Dự đoán với mô hình học máy và Triangulation ---
with st.expander("📍 Dự đoán nguồn phát xạ từ mô hình và Triangulation"):
    st.subheader("🌍 Dự đoán tọa độ nguồn phát xạ từ mô hình và Triangulation")
    
    st.write("Nhập thông tin của các trạm thu và dự đoán tọa độ nguồn phát xạ.")

    # Nhập thông tin cho trạm thu 1
    st.write("📡 Trạm thu 1")
    lat1 = st.number_input("Vĩ độ trạm thu 1", value=16.0)
    lon1 = st.number_input("Kinh độ trạm thu 1", value=108.0)
    azimuth1 = st.number_input("Góc phương vị trạm thu 1 (độ)", value=45.0)

    # Nhập thông tin cho trạm thu 2
    st.write("📡 Trạm thu 2")
    lat2 = st.number_input("Vĩ độ trạm thu 2", value=16.1)
    lon2 = st.number_input("Kinh độ trạm thu 2", value=108.1)
    azimuth2 = st.number_input("Góc phương vị trạm thu 2 (độ)", value=135.0)

    # Tải mô hình đã huấn luyện (mô hình học máy để dự đoán khoảng cách)
    uploaded_model = st.file_uploader("📂 Tải mô hình học máy đã huấn luyện (.joblib)", type=["joblib"])
    if uploaded_model:
        model = joblib.load(uploaded_model)

        # Dự đoán khoảng cách từ các trạm thu tới nguồn phát xạ
        st.write("📝 Nhập các thông tin để mô hình dự đoán khoảng cách:")
        signal1 = st.number_input("Mức tín hiệu trạm thu 1 (dBµV/m)", value=-80.0)
        signal2 = st.number_input("Mức tín hiệu trạm thu 2 (dBµV/m)", value=-80.0)
        frequency = st.number_input("Tần số (MHz)", value=900.0)

        if st.button("🔍 Dự đoán tọa độ nguồn phát"):
            # Xử lý tín hiệu và các tham số
            az1_sin = np.sin(np.radians(azimuth1))
            az1_cos = np.cos(np.radians(azimuth1))
            az2_sin = np.sin(np.radians(azimuth2))
            az2_cos = np.cos(np.radians(azimuth2))

            X_input = np.array([[lat1, lon1, signal1, frequency, az1_sin, az1_cos]])
            predicted_distance1 = model.predict(X_input)[0]  # Dự đoán khoảng cách trạm thu 1

            X_input = np.array([[lat2, lon2, signal2, frequency, az2_sin, az2_cos]])
            predicted_distance2 = model.predict(X_input)[0]  # Dự đoán khoảng cách trạm thu 2

            # Tiến hành triangulation để định vị nguồn phát xạ
            lat3, lon3 = triangulation(lat1, lon1, azimuth1, lat2, lon2, azimuth2)
            
            # Hiển thị kết quả
            st.success(f"🎯 Tọa độ nguồn phát xạ dự đoán:")
            st.markdown(f"- **Vĩ độ**: `{lat3:.6f}`")
            st.markdown(f"- **Kinh độ**: `{lon3:.6f}`")
            
            # Hiển thị kết quả trên bản đồ
            m = folium.Map(location=[lat1, lon1], zoom_start=10)
            folium.Marker([lat1, lon1], tooltip="Trạm thu 1", icon=folium.Icon(color='blue')).add_to(m)
            folium.Marker([lat2, lon2], tooltip="Trạm thu 2", icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([lat3, lon3], tooltip="Nguồn phát xạ dự đoán", icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine(locations=[[lat1, lon1], [lat3, lon3]], color='orange').add_to(m)
            folium.PolyLine(locations=[[lat2, lon2], [lat3, lon3]], color='orange').add_to(m)
            
            st_folium(m, width=800, height=500)
