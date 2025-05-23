import streamlit as st
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# --- Chuyển đổi dBm sang dBµV/m ---
def dBm_to_dBµV_m(dBm):
    return dBm + 120  # Công thức chuyển đổi từ dBm sang dBµV/m

def simulate_signal_strength(dist_km, h, freq_mhz):
    # Kiểm tra giá trị đầu vào của các tham số để tránh lỗi
    if dist_km <= 0 or h <= 0 or freq_mhz <= 0:
        raise ValueError("Các tham số dist_km, h và freq_mhz phải lớn hơn 0.")
    
    # Công thức tính tín hiệu với đơn vị dBm (sau khi chuyển đổi, chúng ta sẽ sử dụng dBµV/m)
    path_loss = 32.45 + 20 * np.log10(dist_km) + 20 * np.log10(freq_mhz)
    signal_dBm = -30 - path_loss + 10 * np.log10(h)
    
    # Chuyển đổi tín hiệu từ dBm sang dBµV/m
    signal_dBµV_m = dBm_to_dBµV_m(signal_dBm)
    
    return signal_dBµV_m

def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    # Kiểm tra xem khoảng cách có hợp lệ không
    if distance_km <= 0:
        raise ValueError("Khoảng cách phải lớn hơn 0.")
    
    # Kiểm tra các giá trị lat1, lon1, azimuth_deg có hợp lệ không
    if not (-90 <= lat1 <= 90) or not (-180 <= lon1 <= 180):
        raise ValueError("Vị trí tọa độ không hợp lệ.")
    
    R = 6371.0  # Bán kính Trái Đất (km)
    brng = radians(azimuth_deg)  # Chuyển đổi góc phương vị sang radian
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    # Tính toán vị trí đích
    lat2 = np.arcsin(sin(lat1) * cos(distance_km / R) + cos(lat1) * sin(distance_km / R) * cos(brng))
    lon2 = lon1 + atan2(sin(brng) * sin(distance_km / R) * cos(lat1), cos(distance_km / R) - sin(lat1) * sin(lat2))

    # Chuyển tọa độ trở lại độ
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)
    
    return lat2, lon2  # Trả về tọa độ điểm đích (lat2, lon2)
def find_intersection(lat1, lon1, az1, lat2, lon2, az2):
    # Tính toán tọa độ giao điểm giữa 2 tia từ 2 trạm thu
    az1 = radians(az1)
    az2 = radians(az2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Tính khoảng cách giữa 2 trạm thu
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    
    # Công thức tính toán điểm giao nhau giữa 2 tia
    x = (sin(az1) - sin(az2)) * distance / (cos(az1) - cos(az2))
    y = (sin(az2) * cos(az1) - cos(az2) * sin(az1)) * distance / (cos(az1) - cos(az2))

    # Tính toán vị trí giao điểm
    lat3 = lat1 + y / 6371.0  # Độ vĩ độ của giao điểm
    lon3 = lon1 + x / (6371.0 * cos(lat1))  # Độ kinh độ của giao điểm

    # Chuyển tọa độ trở lại độ
    lat3 = degrees(lat3)
    lon3 = degrees(lon3)

    return lat3, lon3

def calculate_intersection(lat1, lon1, az1, lat2, lon2, az2):
    # Chuyển độ sang radian
    theta1 = np.deg2rad(az1)
    theta2 = np.deg2rad(az2)

    # Vector hướng
    dx1, dy1 = np.sin(theta1), np.cos(theta1)
    dx2, dy2 = np.sin(theta2), np.cos(theta2)

    # Dựng ma trận hệ phương trình
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    b = np.array([lat2 - lat1, lon2 - lon1])  # chú ý: lat là y, lon là x

    try:
        t = np.linalg.solve(A, b)
        intersection_lat = lat1 + t[0] * dy1
        intersection_lon = lon1 + t[0] * dx1
        return intersection_lat, intersection_lon
    except np.linalg.LinAlgError:
        return None  # 2 tia song song hoặc gần song song
def azimuth_to_vector(azimuth_deg):
    """
    Chuyển góc phương vị (độ) thành vector hướng (dx, dy) trong tọa độ phẳng
    0 độ là Bắc, 90 là Đông, 180 là Nam, 270 là Tây
    """
    azimuth_rad = np.deg2rad(azimuth_deg)
    dx = np.sin(azimuth_rad)
    dy = np.cos(azimuth_rad)
    return dx, dy

def compute_intersection_from_azimuths(lat1, lon1, az1, lat2, lon2, az2):
    """
    Tính điểm giao giữa 2 tia từ trạm thu 1 và 2, theo góc phương vị az1 và az2
    """
    x1, y1 = lon1, lat1
    dx1, dy1 = azimuth_to_vector(az1)

    x2, y2 = lon2, lat2
    dx2, dy2 = azimuth_to_vector(az2)

    A = np.array([[dx1, -dx2],
                  [dy1, -dy2]])
    b = np.array([x2 - x1, y2 - y1])

    try:
        t, s = np.linalg.solve(A, b)
        x_int = x1 + t * dx1
        y_int = y1 + t * dy1
        return y_int, x_int  # lat, lon
    except np.linalg.LinAlgError:
        return None  # Hai tia song song hoặc không xác định giao điểm

# --- Giao diện ---
st.set_page_config(layout="wide")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

tab1, tab2 = st.tabs(["1. Huấn luyện mô hình", "2. Dự đoán tọa độ"])

# --- Tab 1: Huấn luyện ---
with tab1:
    st.subheader("📡 Huấn luyện mô hình với dữ liệu mô phỏng hoặc thực tế")

    option = st.radio("Chọn nguồn dữ liệu huấn luyện:", ("Sinh dữ liệu mô phỏng", "Tải file Excel dữ liệu thực tế"))

    df = None  # Đặt mặc định tránh lỗi NameError

    if option == "Sinh dữ liệu mô phỏng":
        if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
            st.info("Đang sinh dữ liệu mô phỏng...")
            np.random.seed(42)
            n_samples = 1000  # Tạo 1000 mẫu dữ liệu mô phỏng
            data = []
            for _ in range(n_samples):
                lat_tx = np.random.uniform(10.0, 21.0)
                lon_tx = np.random.uniform(105.0, 109.0)
                lat_rx = lat_tx + np.random.uniform(-0.05, 0.05)
                lon_rx = lon_tx + np.random.uniform(-0.05, 0.05)
                h_rx = np.random.uniform(2, 100)
                freq = np.random.uniform(30, 2600)

                azimuth = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
                distance = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
                signal = simulate_signal_strength(distance, h_rx, freq)

                data.append({
                    "lat_receiver": lat_rx,
                    "lon_receiver": lon_rx,
                    "antenna_height": h_rx,
                    "azimuth": azimuth,
                    "frequency": freq,
                    "signal_strength": signal,  # Đơn vị dBµV/m
                    "distance_km": distance
                })

            df = pd.DataFrame(data)
            st.success("Dữ liệu mô phỏng đã được sinh thành công!")

            # Hiển thị 5 dòng đầu tiên của dữ liệu mô phỏng
            st.dataframe(df.head())

            # Tạo file Excel để tải xuống
            towrite = BytesIO()
            df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="📥 Tải dữ liệu mô phỏng (.xlsx)",
                data=towrite,
                file_name="simulation_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        uploaded_data = st.file_uploader("📂 Tải file Excel dữ liệu thực tế", type=["xlsx"])
        if uploaded_data:
            df = pd.read_excel(uploaded_data)
            st.success("Đã tải dữ liệu thực tế.")
            st.dataframe(df.head())  # Hiển thị dữ liệu thực tế tải lên
        else:
            st.info("Vui lòng tải file dữ liệu để huấn luyện.")

    if df is not None and st.button("🔧 Tiến hành huấn luyện mô hình"):
        try:
            st.info("Đang huấn luyện mô hình...")

            # Xử lý thêm dữ liệu
            df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
            df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))

            X = df[['lat_receiver', 'lon_receiver', 'antenna_height', 'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
            y = df[['distance_km']]

            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Tuning tham số với RandomizedSearchCV ---
            param_dist = {
                'n_estimators': [100, 200, 300],  # Giảm số lượng giá trị tham số để thử
                'max_depth': [3, 6, 9],  # Giảm số giá trị tham số
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }

            model = XGBRegressor(random_state=42)

            # Giảm số vòng lặp để tăng tốc
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)

            # Thêm thông báo cho người dùng khi quá trình huấn luyện bắt đầu
            st.info("Đang thực hiện RandomizedSearchCV để tìm tham số tối ưu...")

            random_search.fit(X_train, y_train.values.ravel())

            best_model = random_search.best_estimator_

            # Đánh giá mô hình
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success(f"Huấn luyện xong - MAE khoảng cách: {mae:.3f} km")
            st.success(f"RMSE: {rmse:.3f} km")
            st.success(f"R²: {r2:.3f}")

            # ✅ Lưu mô hình vào file gốc (local server)
            model_path = "distance_model_new.joblib"
            joblib.dump(best_model, model_path)

            buffer = BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)
            
      
            # Cung cấp nút tải mô hình đã huấn luyện
            st.download_button(
                label="📥 Tải mô hình huấn luyện (.joblib)",
                data=buffer,
                file_name="distance_model.joblib",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
            st.exception(e)

# --- Tab 2: Dự đoán ---
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")

    model = None
    model_path = "distance_model_2.joblib"

    # Kiểm tra file có tồn tại trên hệ thống không
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Đã tải mô hình từ file local.")
    else:
        uploaded_model = st.file_uploader("📂 Tải mô hình đã huấn luyện (.joblib)", type=["joblib"])
        if uploaded_model is not None:
            with open(model_path, "wb") as f:
                f.write(uploaded_model.read())
            model = joblib.load(model_path)
            st.success("✅ Đã tải mô hình từ file được tải lên.")
        else:
            st.warning("⚠️ Chưa có mô hình. Vui lòng tải lên file `.joblib`.")

    # Kiểm tra nếu model đã sẵn sàng thì tiếp tục
    if model is not None:
        # TODO: Thêm code để nhập dữ liệu và dự đoán
        st.write("Sẵn sàng dự đoán...")

        uploaded_excel = st.file_uploader("📄 Hoặc tải file Excel chứa thông tin các trạm thu", type=["xlsx"])

        if uploaded_excel:
            df_input = pd.read_excel(uploaded_excel)
            results = []
            st.session_state['intersection_points'] = []
            m = folium.Map(location=[df_input['lat_receiver'].mean(), df_input['lon_receiver'].mean()], zoom_start=8)

            for _, row in df_input.iterrows():
                az_sin = np.sin(np.radians(row['azimuth']))
                az_cos = np.cos(np.radians(row['azimuth']))
                X_input = np.array([[row['lat_receiver'], row['lon_receiver'], row['antenna_height'], row['signal_strength'], row['frequency'], az_sin, az_cos]])
                predicted_distance = model.predict(X_input)[0]
                predicted_distance = max(predicted_distance, 0.1)

                lat_pred, lon_pred = calculate_destination(row['lat_receiver'], row['lon_receiver'], row['azimuth'], predicted_distance)

                # Thêm thông tin về tần số và mức tín hiệu vào tooltip của "Nguồn phát dự đoán"
                folium.Marker(
                    [lat_pred, lon_pred],
                    tooltip=f"Nguồn phát dự đoán\nTần số: {row['frequency']} MHz\nMức tín hiệu: {row['signal_strength']} dBµV/m",
                    icon=folium.Icon(color='red')
                ).add_to(m)

                folium.Marker([row['lat_receiver'], row['lon_receiver']], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
                folium.PolyLine(locations=[[row['lat_receiver'], row['lon_receiver']], [lat_pred, lon_pred]], color='green').add_to(m)

                results.append({
                    "lat_receiver": row['lat_receiver'],
                    "lon_receiver": row['lon_receiver'],
                    "lat_pred": lat_pred,
                    "lon_pred": lon_pred,
                    "predicted_distance_km": predicted_distance,
                    "frequency": row['frequency'],
                    "signal_strength": row['signal_strength']
                })


                        

            st.dataframe(pd.DataFrame(results))
            
            # Tính toán điểm giao cắt nếu có tần số trùng
            if st.button("Tính điểm giao cắt nếu 1 tần số được thu từ 2 trạm"):
                    frequency_groups = df_input.groupby('frequency')
                    
                    # Kiểm tra xem có ít nhất 2 trạm thu cùng tần số
                    for freq, group in frequency_groups:
                        if len(group) >= 2:
                            #st.write(f"Đang tính điểm giao cắt cho tần số {freq} MHz...")
                            for i in range(len(group)):
                                for j in range(i + 1, len(group)):
                                    row1 = group.iloc[i]
                                    row2 = group.iloc[j]
                                    azimuth1 = row1['azimuth']
                                    azimuth2 = row2['azimuth']
                                    lat1, lon1 = row1['lat_receiver'], row1['lon_receiver']
                                    lat2, lon2 = row2['lat_receiver'], row2['lon_receiver']
                                    intersection_lat, intersection_lon = compute_intersection_from_azimuths(lat1, lon1, azimuth1, lat2, lon2, azimuth2)
                                    folium.Marker([intersection_lat, intersection_lon], tooltip=f"Tọa độ nguồn phát tần số {freq} MHz là {intersection_lat:.4f},{intersection_lon:.4f}", icon=folium.Icon(color='green')).add_to(m)
                                    st.write(f"Tọa độ điểm giao cắt cho tần số {freq} MHz là: {intersection_lat:.4f} / {intersection_lon:.4f}")
                                    # Lưu điểm giao cắt vào session_state
                                    st.session_state['intersection_points'].append((intersection_lat, intersection_lon))

                    # Hiển thị điểm giao cắt trên bản đồ
                    #for lat, lon in st.session_state['intersection_points']:
                        #folium.Marker([lat, lon], tooltip="Điểm giao cắt", icon=folium.Icon(color='green')).add_to(m)
                        #st.write(f"Tọa độ nguồn phát tần số {freq} MHz là {lat:.4f},{lon:.4f}...")
            with st.container():
                st_folium(m, width=1300, height=500, returned_objects=[])
            #st_folium(m, width=1300, height=500)
        else:
            with st.form("input_form"):
                lat_rx = st.number_input("Vĩ độ trạm thu", value=21.339)
                lon_rx = st.number_input("Kinh độ trạm thu", value=105.4056)
                h_rx = st.number_input("Chiều cao anten (m)", value=30.0)
                signal = st.number_input("Mức tín hiệu thu (dBµV/m)", value=50.0)  # Đơn vị dBµV/m
                freq = st.number_input("Tần số (MHz)", value=900.0)
                azimuth = st.number_input("Góc phương vị (độ)", value=45.0)
                submitted = st.form_submit_button("🔍 Dự đoán tọa độ nguồn phát")

            if submitted:
                az_sin = np.sin(np.radians(azimuth))
                az_cos = np.cos(np.radians(azimuth))
                X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])
                predicted_distance = model.predict(X_input)[0]
                predicted_distance = max(predicted_distance, 0.1)

                lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, predicted_distance)

                st.success("🎯 Tọa độ nguồn phát xạ dự đoán:")
                st.markdown(f"- **Vĩ độ**: `{lat_pred:.6f}`")
                st.markdown(f"- **Kinh độ**: `{lon_pred:.6f}`")
                st.markdown(f"- **Khoảng cách dự đoán**: `{predicted_distance:.2f} km`")

                m = folium.Map(location=[lat_rx, lon_rx], zoom_start=10)
                folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
                folium.Marker(
                    [lat_pred, lon_pred],
                    tooltip=f"Nguồn phát dự đoán\nTần số: {freq} MHz\nMức tín hiệu: {signal} dBµV/m",
                    icon=folium.Icon(color='red')
                ).add_to(m)
                folium.PolyLine(locations=[[lat_rx, lon_rx], [lat_pred, lon_pred]], color='green').add_to(m)

                with st.container():
                    st_folium(m, width=1300, height=500, returned_objects=[])
