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
from sklearn.model_selection import GridSearchCV

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
@st.cache_data
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1); lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    return (degrees(atan2(x, y)) + 360) % 360

@st.cache_data
def simulate_signal_strength(dist_km, h, freq_mhz):
    path_loss = 32.45 + 20*np.log10(dist_km + 0.1) + 20*np.log10(freq_mhz + 1)
    return -30 - path_loss + 10*np.log10(h + 1)

@st.cache_data
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
    else:
        uploaded_data = st.file_uploader("📂 Tải file Excel dữ liệu thực tế", type=["xlsx"])
        if uploaded_data:
            df = pd.read_excel(uploaded_data)
            st.success("Đã tải dữ liệu.")
            st.dataframe(df.head())
        else:
            st.info("Vui lòng tải file dữ liệu để huấn luyện.")

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

            # Hyperparameter tuning - GridSearchCV
            param_grid = {'final_estimator__fit_intercept': [True, False]}
            grid_search = GridSearchCV(stacking_model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            # Đánh giá
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success(f"MAE: {mae:.3f} km")
            st.success(f"RMSE: {rmse:.3f} km")
            st.success(f"R²: {r2:.3f}")

            buffer = BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)
            st.download_button("📥 Tải mô hình (.joblib)",
                               data=buffer,
                               file_name="stacking_model.joblib",
                               mime="application/octet-stream")
        except Exception as e:
            st.error("Đã xảy ra lỗi khi huấn luyện.")
            st.exception(e)

# ---------- Tab 2 ----------
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")
    uploaded_model = st.file_uploader(
        "📂 Tải mô hình (.joblib) đã huấn luyện",
        type=["joblib"],
        key="model_file")
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.success("✅ Đã nạp mô hình thành công!")

        st.write("Nhập thông tin để dự đoán:")
        lat_receiver = st.number_input("Vĩ độ trạm thu:", min_value=-90.0, max_value=90.0, value=10.0)
        lon_receiver = st.number_input("Kinh độ trạm thu:", min_value=-180.0, max_value=180.0, value=105.0)
        h_receiver = st.number_input("Chiều cao anten trạm thu (m):", min_value=0.0, max_value=100.0, value=10.0)
        azimuth = st.number_input("Góc phương vị (°):", min_value=0.0, max_value=360.0, value=30.0)
        frequency = st.number_input("Tần số (MHz):", min_value=100.0, max_value=5000.0, value=1500.0)

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
