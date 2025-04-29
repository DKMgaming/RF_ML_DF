import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np

# Hàm tính toán tọa độ đích
def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    R = 6371.0  # Bán kính trái đất (km)
    brng = np.radians(azimuth_deg)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    
    lat2 = np.arcsin(np.sin(lat1)*np.cos(distance_km/R) + np.cos(lat1)*np.sin(distance_km/R)*np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(distance_km/R)*np.cos(lat1),
                             np.cos(distance_km/R) - np.sin(lat1)*np.sin(lat2))
    
    return np.degrees(lat2), np.degrees(lon2)

# Giao diện Streamlit
st.set_page_config(layout="wide")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

# 1. Tải mô hình đã huấn luyện
uploaded_model = st.file_uploader("📂 Tải mô hình (.joblib) đã huấn luyện", type=["joblib"], key="model_file")
if uploaded_model:
    model = joblib.load(uploaded_model)
    st.success("✅ Đã nạp mô hình.")

# 2. Khởi tạo biến lưu trong session_state
if "file_map" not in st.session_state:
    st.session_state.file_map = None

# 3. Dự đoán từ file Excel
uploaded_excel = st.file_uploader("Tải file Excel", type=["xlsx"], key="rx_excel")
if st.button("🔍 Chạy dự đoán file Excel") and uploaded_excel and uploaded_model:
    df_input = pd.read_excel(uploaded_excel)
    results = []
    map_center = [df_input['lat_receiver'].mean(), df_input['lon_receiver'].mean()]
    m = folium.Map(location=map_center, zoom_start=8)

    for _, row in df_input.iterrows():
        az_sin = np.sin(np.radians(row['azimuth']))
        az_cos = np.cos(np.radians(row['azimuth']))
        X_input = np.array([[row['lat_receiver'], row['lon_receiver'],
                             row['antenna_height'], row['signal_strength'],
                             row['frequency'], az_sin, az_cos]])
        pred_dist = max(model.predict(X_input)[0], 0.1)
        lat_pred, lon_pred = calculate_destination(
            row['lat_receiver'], row['lon_receiver'],
            row['azimuth'], pred_dist)

        folium.Marker([lat_pred, lon_pred],
                      tooltip=f"Nguồn phát dự đoán\nTần số: {row['frequency']} MHz\nMức tín hiệu: {row['signal_strength']} dBm",
                      icon=folium.Icon(color='red')).add_to(m)
        folium.Marker([row['lat_receiver'], row['lon_receiver']],
                      tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
        folium.PolyLine([[row['lat_receiver'], row['lon_receiver']],
                         [lat_pred, lon_pred]], color='green').add_to(m)

        results.append({
            "lat_receiver": row['lat_receiver'],
            "lon_receiver": row['lon_receiver'],
            "lat_pred": lat_pred,
            "lon_pred": lon_pred,
            "predicted_distance_km": pred_dist,
            "frequency": row['frequency'],
            "signal_strength": row['signal_strength']
        })

    # Lưu vào session_state
    st.session_state.file_results = pd.DataFrame(results)
    st.session_state.file_map = m
    st.success("✅ Hoàn tất dự đoán file Excel!")

# Hiển thị kết quả (nếu đã có)
if st.session_state.file_results is not None:
    st.dataframe(st.session_state.file_results)
if st.session_state.file_map is not None:
    st_folium(st.session_state.file_map, width=800, height=500)

