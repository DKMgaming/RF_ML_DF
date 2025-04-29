import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikeras[tensorflow]"])
# ---------- Hàm build NN dùng cho cả train & predict ----------
def build_model():
    """Trả về mô hình Keras 2 hidden‑layer; input_shape cố định = 7 feature."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------- Hàm phụ ----------
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1); lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    return (degrees(atan2(x, y)) + 360) % 360

def simulate_signal_strength(dist_km, h, freq_mhz):
    path_loss = 32.45 + 20*np.log10(dist_km + 0.1) + 20*np.log10(freq_mhz + 1)
    return -30 - path_loss + 10*np.log10(h + 1)

def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    R = 6371.0
    brng = radians(azimuth_deg)
    lat1 = radians(lat1); lon1 = radians(lon1)
    lat2 = np.arcsin(sin(lat1)*cos(distance_km/R) + cos(lat1)*sin(distance_km/R)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(distance_km/R)*cos(lat1),
                        cos(distance_km/R) - sin(lat1)*sin(lat2))
    return degrees(lat2), degrees(lon2)

# ---------- Giao diện ----------
st.set_page_config(layout="wide")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

tab2 = st.tabs(["2. Dự đoán tọa độ"])


# ---------- Tab 2 ----------
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")

    # ---- 1. Tải mô hình đã huấn luyện ----
    uploaded_model = st.file_uploader(
        "📂 Tải mô hình (.joblib) đã huấn luyện",
        type=["joblib"],
        key="model_file")
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.success("✅ Đã nạp mô hình.")

    # ---- 2. Khởi tạo biến lưu trong session_state (chỉ 1 lần) ----
    for key in ("file_results", "file_map", "single_result", "single_map"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ===============================================================
    # 📄  DỰ ĐOÁN TỪ FILE EXCEL TRẠM THU
    # ===============================================================
    st.markdown("### 📄 Dự đoán từ file Excel danh sách trạm thu")
    uploaded_excel = st.file_uploader(
        "Tải file Excel", type=["xlsx"], key="rx_excel")

    # Nút chạy dự đoán file
    if st.button("🔍 Chạy dự đoán file Excel") and uploaded_excel and uploaded_model:
        df_input = pd.read_excel(uploaded_excel)
        results = []
        map_center = [df_input['lat_receiver'].mean(),
                      df_input['lon_receiver'].mean()]
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
                          tooltip=(f"Nguồn phát dự đoán\nTần số: {row['frequency']} MHz"
                                   f"\nMức tín hiệu: {row['signal_strength']} dBm"),
                          icon=folium.Icon(color='red')).add_to(m)
            folium.Marker([row['lat_receiver'], row['lon_receiver']],
                          tooltip="Trạm thu",
                          icon=folium.Icon(color='blue')).add_to(m)
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

    # Hiển thị (nếu đã có)
    if st.session_state.file_results is not None:
        st.dataframe(st.session_state.file_results)
    if st.session_state.file_map is not None:
        st_folium(st.session_state.file_map, width=1300, height=500)

    # ===============================================================
    # ⌨️  DỰ ĐOÁN NHẬP TAY (FORM)
    # ===============================================================
    st.markdown("---\n### ⌨️ Dự đoán bằng cách nhập tay thông số")

    with st.form("input_form", clear_on_submit=False):
        lat_rx = st.number_input("Vĩ độ trạm thu", value=16.0)
        lon_rx = st.number_input("Kinh độ trạm thu", value=108.0)
        h_rx = st.number_input("Chiều cao anten (m)", value=30.0)
        signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80.0)
        freq = st.number_input("Tần số (MHz)", value=900.0)
        azimuth = st.number_input("Góc phương vị (độ)", value=45.0)
        submitted = st.form_submit_button("🔍 Dự đoán")

    if submitted and uploaded_model:
        az_sin = np.sin(np.radians(azimuth))
        az_cos = np.cos(np.radians(azimuth))
        X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])
        pred_dist = max(model.predict(X_input)[0], 0.1)
        lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, pred_dist)

        # Lưu kết quả & map
        st.session_state.single_result = (lat_pred, lon_pred, pred_dist)
        m2 = folium.Map(location=[lat_rx, lon_rx], zoom_start=10)
        folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu",
                      icon=folium.Icon(color='blue')).add_to(m2)
        folium.Marker([lat_pred, lon_pred],
                      tooltip=(f"Nguồn phát dự đoán\nTần số: {freq} MHz\nMức tín hiệu: {signal} dBm\nKhoảng cách: {pred_dist} km"),
                      icon=folium.Icon(color='red')).add_to(m2)
        folium.PolyLine([[lat_rx, lon_rx], [lat_pred, lon_pred]],
                        color='green').add_to(m2)
        st.session_state.single_map = m2
        st.success("✅ Đã tính xong toạ độ!")

    # Hiển thị kết quả nhập tay
    if st.session_state.single_result is not None:
        lat_pred, lon_pred, dist = st.session_state.single_result
        st.info(f"🎯 **Vĩ độ**: `{lat_pred:.6f}`  "
                f"**Kinh độ**: `{lon_pred:.6f}`  "
                f"**Khoảng cách**: `{dist:.2f} km`")
    if st.session_state.single_map is not None:
        st_folium(st.session_state.single_map, width=1300, height=500)

