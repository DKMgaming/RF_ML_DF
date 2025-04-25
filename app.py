import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
import os

st.set_page_config(page_title="Dự đoán nguồn phát", layout="wide")

st.title("📡 Ứng dụng dự đoán tọa độ nguồn phát xạ")

# ---------- Hàm ----------
def calculate_destination(lat, lon, azimuth_deg, distance_km):
    R = 6371.0
    bearing = np.radians(azimuth_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    d_div_R = distance_km / R

    lat2 = np.arcsin(np.sin(lat1)*np.cos(d_div_R) +
                     np.cos(lat1)*np.sin(d_div_R)*np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing)*np.sin(d_div_R)*np.cos(lat1),
                             np.cos(d_div_R)-np.sin(lat1)*np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔧 Huấn luyện mô hình", "🎯 Dự đoán tọa độ nguồn phát"])

with tab1:
    st.header("🔧 Huấn luyện mô hình")

    uploaded_file = st.file_uploader("Tải lên file Excel chứa dữ liệu huấn luyện (tọa độ trạm thu, góc azimuth, tín hiệu, tọa độ nguồn phát)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("📄 Xem trước dữ liệu:")
        st.dataframe(df.head())

        required_cols = ["lat_rx", "lon_rx", "height_rx", "signal", "azimuth", "lat_tx", "lon_tx"]
        if not all(col in df.columns for col in required_cols):
            st.error("❌ Thiếu cột bắt buộc trong dữ liệu.")
        else:
            def compute_distance(row):
                rx = (row["lat_rx"], row["lon_rx"])
                tx = (row["lat_tx"], row["lon_tx"])
                return geodesic(rx, tx).km

            df["distance_km"] = df.apply(compute_distance, axis=1)

            X = df[["lat_rx", "lon_rx", "height_rx", "signal", "azimuth"]]
            y = df["distance_km"]

            xgb = XGBRegressor()
            rf = RandomForestRegressor()
            nn = MLPRegressor(max_iter=1000)

            model = StackingRegressor(
                estimators=[('xgb', xgb), ('rf', rf), ('nn', nn)],
                final_estimator=Ridge()
            )

            model.fit(X, y)
            dump(model, "stacking_model.joblib")
            st.success("✅ Mô hình đã huấn luyện xong và lưu thành công!")

with tab2:
    st.header("🎯 Dự đoán tọa độ nguồn phát")

    col1, col2 = st.columns([1, 2])
    with col1:
        model_file = st.file_uploader("📁 Tải mô hình đã huấn luyện (.joblib)", type=["joblib"])

    if model_file:
        with open("model_loaded.joblib", "wb") as f:
            f.write(model_file.read())
        model = load("model_loaded.joblib")

        subtab1, subtab2 = st.tabs(["📄 Dự đoán từ file Excel", "📝 Nhập tay"])

        with subtab1:
            st.subheader("📄 Tải file Excel dự đoán")
            input_file = st.file_uploader("Tải file Excel đầu vào (lat_rx, lon_rx, height_rx, signal, azimuth, freq)", type=["xlsx"], key="predict_excel")

            if input_file:
                df_input = pd.read_excel(input_file)
                st.write("📑 Dữ liệu đầu vào:")
                st.dataframe(df_input.head())

                results = []
                m = folium.Map(location=[df_input["lat_rx"].mean(), df_input["lon_rx"].mean()], zoom_start=7)

                for _, row in df_input.iterrows():
                    lat_rx, lon_rx, h_rx = row["lat_rx"], row["lon_rx"], row["height_rx"]
                    signal, azimuth, freq = row["signal"], row["azimuth"], row["freq"]
                    X_input = np.array([[lat_rx, lon_rx, h_rx, signal, azimuth]])
                    pred_dist = max(model.predict(X_input)[0], 0.1)
                    lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, pred_dist)

                    result = {
                        "lat_receiver": lat_rx,
                        "lon_receiver": lon_rx,
                        "lat_pred": lat_pred,
                        "lon_pred": lon_pred,
                        "predicted_distance_km": pred_dist,
                        "frequency": freq,
                        "signal_strength": signal
                    }
                    results.append(result)

                    folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
                    folium.Marker([lat_pred, lon_pred], tooltip=f"Nguồn phát dự đoán\nTần số: {freq} MHz", icon=folium.Icon(color='red')).add_to(m)
                    folium.PolyLine([[lat_rx, lon_rx], [lat_pred, lon_pred]], color='green').add_to(m)

                st.success(f"✅ Đã dự đoán {len(results)} điểm.")
                st.dataframe(pd.DataFrame(results))
                st_folium(m, width=1000, height=500)

        with subtab2:
            st.subheader("📝 Nhập tay thông số trạm thu")
            lat_rx = st.number_input("Vĩ độ trạm thu", format="%.6f")
            lon_rx = st.number_input("Kinh độ trạm thu", format="%.6f")
            h_rx = st.number_input("Chiều cao anten trạm thu (m)", value=10)
            signal = st.number_input("Mức tín hiệu thu (dBm)", value=-90)
            azimuth = st.number_input("Hướng azimuth (độ)", min_value=0, max_value=360, value=0)
            freq = st.number_input("Tần số sử dụng (MHz)", value=900)

            if st.button("📍 Dự đoán tọa độ nguồn phát"):
                X_input = np.array([[lat_rx, lon_rx, h_rx, signal, azimuth]])
                pred_dist = max(model.predict(X_input)[0], 0.1)
                lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, pred_dist)

                st.write("### ✅ Kết quả dự đoán")
                result = {
                    "lat_receiver": lat_rx,
                    "lon_receiver": lon_rx,
                    "lat_pred": lat_pred,
                    "lon_pred": lon_pred,
                    "predicted_distance_km": pred_dist,
                    "frequency": freq,
                    "signal_strength": signal
                }
                st.json(result)

                m = folium.Map(location=[lat_rx, lon_rx], zoom_start=10)
                folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
                folium.Marker([lat_pred, lon_pred], tooltip="Nguồn phát dự đoán", icon=folium.Icon(color='red')).add_to(m)
                folium.PolyLine([[lat_rx, lon_rx], [lat_pred, lon_pred]], color='green').add_to(m)
                st_folium(m, width=800, height=500)
