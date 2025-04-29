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
from streamlit_folium import st_folium

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

tab1, tab2 = st.tabs(["1. Huấn luyện mô hình", "2. Dự đoán tọa độ"])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("📡 Huấn luyện mô hình với dữ liệu mô phỏng hoặc thực tế")

    option = st.radio("Chọn nguồn dữ liệu huấn luyện:",
                      ("Sinh dữ liệu mô phỏng", "Tải file Excel dữ liệu thực tế"))
    df = None

    # --- Sinh dữ liệu mô phỏng ---
    if option == "Sinh dữ liệu mô phỏng":
        if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
            st.info("Đang sinh dữ liệu mô phỏng...")
            np.random.seed(42)
            n_samples = 1000
            data = []
            for _ in range(n_samples):
                lat_tx = np.random.uniform(10.0, 21.0)
                lon_tx = np.random.uniform(105.0, 109.0)
                lat_rx = lat_tx + np.random.uniform(-0.05, 0.05)
                lon_rx = lon_tx + np.random.uniform(-0.05, 0.05)
                h_rx = np.random.uniform(5, 50)
                freq = np.random.uniform(400, 2600)

                az = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
                dist = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
                signal = simulate_signal_strength(dist, h_rx, freq)

                data.append({
                    "lat_receiver": lat_rx,
                    "lon_receiver": lon_rx,
                    "antenna_height": h_rx,
                    "azimuth": az,
                    "frequency": freq,
                    "signal_strength": signal,
                    "distance_km": dist
                })
            df = pd.DataFrame(data)
            st.success("Đã sinh dữ liệu mô phỏng!")
            st.dataframe(df.head())

            towrite = BytesIO()
            df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button("📥 Tải dữ liệu mô phỏng (.xlsx)",
                               data=towrite,
                               file_name="simulation_data.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # --- Tải file thực tế ---
    else:
        uploaded_data = st.file_uploader("📂 Tải file Excel dữ liệu thực tế", type=["xlsx"])
        if uploaded_data:
            df = pd.read_excel(uploaded_data)
            st.success("Đã tải dữ liệu.")
            st.dataframe(df.head())
        else:
            st.info("Vui lòng tải file dữ liệu để huấn luyện.")

    # ---------- Huấn luyện ----------
    if df is not None and st.button("🔧 Tiến hành huấn luyện mô hình"):
        try:
            st.info("Đang huấn luyện mô hình...")

            # Tiền xử lý
            df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
            df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))

            X = df[['lat_receiver', 'lon_receiver', 'antenna_height',
                    'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
            y = df[['distance_km']]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            nn_reg = KerasRegressor(model=build_model,
                                    epochs=50,
                                    batch_size=32,
                                    verbose=0)

            estimators = [
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('nn', nn_reg)
            ]

            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression()
            )

            stacking_model.fit(X_train, y_train)

            # Đánh giá
            y_pred = stacking_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success(f"MAE: {mae:.3f} km")
            st.success(f"RMSE: {rmse:.3f} km")
            st.success(f"R²: {r2:.3f}")

            buffer = BytesIO()
            joblib.dump(stacking_model, buffer)
            buffer.seek(0)
            st.download_button("📥 Tải mô hình (.joblib)",
                               data=buffer,
                               file_name="stacking_model.joblib",
                               mime="application/octet-stream")
        except Exception as e:
            st.error("Đã xảy ra lỗi khi huấn luyện.")
            st.exception(e)


# Tab 2
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
        st_folium(st.session_state.file_map, width=800, height=500)

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
                      tooltip=(f"Nguồn phát dự đoán\nTần số: {freq} MHz\nMức tín hiệu: {signal} dBm"),
                      icon=folium.Icon(color='red')).add_to(m2)
        folium.PolyLine([[lat_rx, lon_rx], [lat_pred, lon_pred]],
                        color='green').add_to(m2)
        st.session_state.single_map = m2
        st.success("✅ Đã tính xong toạ độ!")

    # Hiển thị kết quả nhập tay
    if st.session_state.single_result is not None:
        lat_pred, lon_pred, dist = st.session_state.single_result
        st.info(f"🎯 **Vĩ độ**: {lat_pred:.6f}  "
                f"**Kinh độ**: {lon_pred:.6f}  "
                f"**Khoảng cách**: {dist:.2f} km")
    if st.session_state.single_map is not None:
        st_folium(st.session_state.single_map, width=800, height=500)
