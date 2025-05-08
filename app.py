import streamlit as st
import numpy as np
import pandas as pd

# Import các thư viện ML/DL. Bắt lỗi nếu thiếu.
try:
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    # from sklearn.linear_model import LinearRegression # Nếu cần dùng cho stacking
    import joblib
    print("Successfully imported ML/DL and joblib.")

    # Import thư viện cho Keras/TensorFlow VÀ Scikeras.
    # Các thư viện này cần thiết nếu mô hình joblib được tạo từ Scikeras.KerasRegressor.
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from scikeras.wrappers import KerasRegressor
        print("Successfully imported TensorFlow/Keras and Scikeras.")
        KERAS_AVAILABLE = True
    except ImportError:
        print("TensorFlow/Keras hoặc Scikeras không tìm thấy. build_model sẽ không hoạt động.")
        KERAS_AVAILABLE = False

except ImportError as e:
    st.error(f"Lỗi: Không tìm thấy thư viện cần thiết. Vui lòng cài đặt chúng. Chi tiết: {e}")
    st.stop() # Dừng ứng dụng nếu các thư viện cốt lõi bị thiếu

from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt
import folium
from streamlit_folium import st_folium

# ---------- Hằng số ----------
EARTH_RADIUS_KM = 6371.0

# ---------- Hàm build NN dùng cho cả train & predict ----------
# HÀM NÀY CẦN ĐƯỢC ĐỊNH NGHĨA NẾU MÔ HÌNH JOBLIB CỦA BẠN LÀ Scikeras.KerasRegressor
# JOBILIB SẼ CẦN HÀM NÀY KHI LOAD MÔ HÌNH ĐÓ
# Nếu mô hình của bạn là XGBoost, RF, StackingRegressor... được lưu bằng joblib,
# thì hàm này không cần thiết cho quá trình *load* mô hình đó, nhưng cần tồn tại
# nếu file joblib được tạo từ KerasRegressor.
# Cách tốt nhất để load Keras là dùng model.save()/.h5 và tf.keras.models.load_model().
# Tuy nhiên, để sửa lỗi hiện tại, ta cứ đảm bảo hàm này tồn tại VÀ có thể được gọi
# (tức là TF/Keras/Scikeras phải được cài đặt).

def build_model():
    """Trả về mô hình Keras 2 hidden‑layer; input_shape cố định = 7 feature."""
    if not KERAS_AVAILABLE:
        # Trường hợp này chỉ xảy ra nếu build_model được gọi MÀ KERAS_AVAILABLE là False.
        # Joblib gọi hàm này, nên lỗi ban đầu "Can't get attribute" xảy ra khi hàm
        # chưa được định nghĩa hoặc bị lỗi import.
        # Nếu import thành công, KERAS_AVAILABLE là True, code này sẽ không chạy.
        st.error("Lỗi nội bộ: Hàm build_model được gọi nhưng TensorFlow/Keras không sẵn sàng.")
        return None

    # Định nghĩa cấu trúc mô hình Keras
    model = Sequential([
        Dense(128, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(1) # Output là 1 giá trị: khoảng cách
    ])
    # Compile mô hình
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------- Hàm phụ tính toán địa lý/RF ----------
# (Giữ nguyên như phiên bản tối ưu)
def calculate_azimuth(lat1, lon1, lat2, lon2):
    """Tính góc phương vị từ điểm 1 đến điểm 2 (độ)."""
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1); lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    return (degrees(atan2(x, y)) + 360) % 360

def simulate_signal_strength(dist_km, h, freq_mhz):
    """Hàm mô phỏng cường độ tín hiệu (không dùng trong code dự đoán này, giữ lại nếu cần)."""
    path_loss = 32.45 + 20*np.log10(dist_km + 0.1) + 20*np.log10(freq_mhz + 1)
    return -30 - path_loss + 10*np.log10(h + 1)

def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    """Tính tọa độ điểm đích từ điểm gốc, phương vị và khoảng cách (km)."""
    brng = radians(azimuth_deg)
    lat1 = radians(lat1); lon1 = radians(lon1)

    lat2 = np.arcsin(sin(lat1)*cos(distance_km/EARTH_RADIUS_KM) +
                     cos(lat1)*sin(distance_km/EARTH_RADIUS_KM)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(distance_km/EARTH_RADIUS_KM)*cos(lat1),
                        cos(distance_km/EARTH_RADIUS_KM) - sin(lat1)*sin(lat2))

    return degrees(lat2), degrees(lon2)

# ---------- Hàm xử lý dự đoán chung ----------
def predict_location_from_inputs(model, lat_rx, lon_rx, h_rx, signal, freq, azimuth):
    """Nhận các thông số đầu vào và trả về tọa độ nguồn phát dự đoán và khoảng cách."""
    if model is None:
        # Lỗi này không xảy ra nếu nút predict bị disabled đúng cách
        st.error("Lỗi: Mô hình chưa được nạp.")
        return None, None, None

    az_sin = np.sin(np.radians(azimuth))
    az_cos = np.cos(np.radians(azimuth))
    X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])

    try:
        pred_dist_raw = model.predict(X_input)[0]
        pred_dist = max(pred_dist_raw, 0.01) # Khoảng cách tối thiểu 10m
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán khoảng cách bằng mô hình: {e}")
        # Thêm gợi ý nếu lỗi liên quan đến cấu trúc mô hình
        if "'build_model'" in str(e) or "keras" in str(e).lower() or "scikeras" in str(e).lower():
            st.warning("Gợi ý: Lỗi này có thể do mô hình Keras/Scikeras không tương thích hoặc thư viện chưa sẵn sàng.")
        return None, None, None

    try:
        lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, pred_dist)
    except Exception as e:
        st.error(f"Lỗi trong quá trình tính toán tọa độ đích: {e}")
        return None, None, None

    return lat_pred, lon_pred, pred_dist

# ---------- Giao diện Streamlit ----------
st.set_page_config(layout="wide", page_title="Dự đoán Vị trí Nguồn Phát Xạ")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

# ---------- 2. Khởi tạo biến lưu trong session_state (chỉ 1 lần) ----------
for key in ("model", "file_results", "file_map", "single_result", "single_map"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---- 1. Tải mô hình đã huấn luyện ----
st.sidebar.header("Tải mô hình")
st.sidebar.info("Mô hình (.joblib) phải được huấn luyện trên 7 đặc trưng theo thứ tự: Lat Receiver, Lon Receiver, Antenna Height, Signal Strength, Frequency, Azimuth Sin, Azimuth Cos.")
st.sidebar.warning("Nếu mô hình là KerasRegressor được lưu bằng joblib, bạn cần đảm bảo TensorFlow, Keras, Scikeras đã cài đặt và hàm `build_model` có trong script này.")

uploaded_model = st.sidebar.file_uploader(
    "📂 Tải file mô hình (.joblib)",
    type=["joblib"],
    key="model_file_uploader")

# Logic tải mô hình: Chỉ tải khi file mới được upload và khác với file đã tải trước đó
# Sử dụng object ID hoặc hash của uploaded_model để kiểm tra
if uploaded_model is not None:
    # Kiểm tra xem đã có model trong session state chưa hoặc file upload có khác file trước không
    # uploaded_model là một BytesIO-like object, có thể so sánh getvalue() hoặc sử dụng hash
    # Cách đơn giản là kiểm tra sự tồn tại của session state key cho file ID
    if "uploaded_model_id" not in st.session_state or st.session_state.uploaded_model_id != id(uploaded_model):
         # Lưu ID file mới để so sánh lần sau
        st.session_state.uploaded_model_id = id(uploaded_model)

        try:
            with st.spinner("Đang nạp mô hình..."):
                # Đối với KerasRegressor qua joblib, hàm build_model phải có sẵn tại đây
                st.session_state.model = joblib.load(uploaded_model)
            st.sidebar.success(f"✅ Đã nạp mô hình thành công: {type(st.session_state.model).__name__}.")
            # Reset kết quả cũ khi load mô hình mới
            st.session_state.file_results = None
            st.session_state.file_map = None
            st.session_state.single_result = None
            st.session_state.single_map = None

        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi nạp mô hình. Vui lòng kiểm tra file. Chi tiết: {e}")
            st.session_state.model = None # Đảm bảo session state model là None nếu load lỗi
            # Xóa ID file để cho phép tải lại cùng file sau khi sửa lỗi
            if "uploaded_model_id" in st.session_state:
                 del st.session_state.uploaded_model_id


# Hiển thị trạng thái mô hình
if st.session_state.model is not None:
    st.sidebar.info(f"🌟 Mô hình đã sẵn sàng: {type(st.session_state.model).__name__}")
else:
    st.sidebar.warning("⚠️ Chưa có mô hình được nạp hoặc nạp bị lỗi.")


st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")


# ===============================================================
# 📄  DỰ ĐOÁN TỪ FILE EXCEL TRẠM THU
# ===============================================================
st.markdown("### 📄 Dự đoán từ file Excel danh sách trạm thu")
st.info("Tải lên file Excel (.xlsx) với các cột bắt buộc: `lat_receiver`, `lon_receiver`, `antenna_height`, `signal_strength`, `frequency`, `azimuth`.")
uploaded_excel = st.file_uploader(
    "Tải file Excel", type=["xlsx"], key="rx_excel_uploader")

# Nút chạy dự đoán file
# Nút này chỉ được enable khi có cả file Excel và mô hình đã nạp thành công
predict_file_button = st.button(
    "🔍 Chạy dự đoán từ file Excel",
    disabled=(uploaded_excel is None or st.session_state.model is None)
)

if predict_file_button:
    # Kiểm tra lại một lần nữa dù nút đã bị disabled
    if st.session_state.model is None:
        st.warning("⚠️ Vui lòng tải mô hình trước khi chạy dự đoán file.")
    elif uploaded_excel is None:
        st.warning("⚠️ Vui lòng tải file Excel trước khi chạy dự đoán file.")
    else:
        try:
            with st.spinner("Đang xử lý file Excel và dự đoán..."):
                df_input = pd.read_excel(uploaded_excel)

                # Kiểm tra các cột bắt buộc
                required_cols = ['lat_receiver', 'lon_receiver', 'antenna_height',
                                 'signal_strength', 'frequency', 'azimuth']
                if not all(col in df_input.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df_input.columns]
                    st.error(f"❌ File Excel thiếu các cột bắt buộc: {', '.join(missing)}")
                    st.session_state.file_results = None # Reset kết quả cũ
                    st.session_state.file_map = None # Reset bản đồ cũ
                else:
                    results = []
                    # Tính toán trung tâm bản đồ từ các điểm trạm thu
                    if not df_input.empty:
                        map_center = [df_input['lat_receiver'].mean(),
                                      df_input['lon_receiver'].mean()]
                    else:
                        map_center = [0, 0] # Default center if dataframe is empty

                    # Khởi tạo bản đồ MỚI cho mỗi lần chạy file
                    st.session_state.file_map = folium.Map(location=map_center, zoom_start=8)

                    # Sử dụng iterrows() là chuẩn nhưng kém hiệu quả với DF lớn.
                    # Apply hoặc list comprehension có thể nhanh hơn với DF lớn.
                    # Tuy nhiên, với việc vẽ marker/line trên map, iterrows() dễ quản lý.
                    # Nếu dataset rất lớn, cần cân nhắc tối ưu vòng lặp này và cách vẽ map.
                    processed_count = 0
                    for index, row in df_input.iterrows():
                        # Kiểm tra dữ liệu cơ bản trong dòng
                        if pd.isna(row[['lat_receiver', 'lon_receiver', 'antenna_height',
                                         'signal_strength', 'frequency', 'azimuth']]).any():
                            st.warning(f"⚠️ Bỏ qua dòng {index + 1} trong file Excel do thiếu dữ liệu.")
                            continue # Bỏ qua dòng này

                        # Sử dụng hàm predict_location_from_inputs chung
                        lat_pred, lon_pred, pred_dist = predict_location_from_inputs(
                            st.session_state.model,
                            row['lat_receiver'], row['lon_receiver'],
                            row['antenna_height'], row['signal_strength'],
                            row['frequency'], row['azimuth']
                        )

                        if lat_pred is not None and lon_pred is not None:
                            processed_count += 1
                            # Thêm marker và line vào bản đồ hiện tại của file
                            folium.Marker([lat_pred, lon_pred],
                                          tooltip=(f"Nguồn phát dự đoán ({index + 1})\n"
                                                   f"Tần số: {row['frequency']} MHz\n"
                                                   f"Mức tín hiệu: {row['signal_strength']} dBm\n"
                                                   f"Khoảng cách: {pred_dist:.2f} km"),
                                          icon=folium.Icon(color='red', icon='info-sign')).add_to(st.session_state.file_map)
                            folium.Marker([row['lat_receiver'], row['lon_receiver']],
                                          tooltip=f"Trạm thu ({index + 1})\n"
                                                  f"Azimuth: {row['azimuth']}°",
                                          icon=folium.Icon(color='blue', icon='signal')).add_to(st.session_state.file_map)
                            folium.PolyLine([[row['lat_receiver'], row['lon_receiver']],
                                             [lat_pred, lon_pred]], color='green', weight=2.5, opacity=0.7).add_to(st.session_state.file_map)

                            results.append({
                                "ID Trạm Thu": index + 1, # Thêm ID để dễ theo dõi
                                "Vĩ độ Trạm thu": row['lat_receiver'],
                                "Kinh độ Trạm thu": row['lon_receiver'],
                                "Góc phương vị (°))": row['azimuth'],
                                "Tần số (MHz)": row['frequency'],
                                "Mức tín hiệu (dBm)": row['signal_strength'],
                                "Vĩ độ Nguồn (Dự đoán)": lat_pred,
                                "Kinh độ Nguồn (Dự đoán)": lon_pred,
                                "Khoảng cách (Dự đoán) (km)": pred_dist,
                            })
                        # Nếu predict_location_from_inputs trả về None do lỗi, lỗi đã được báo trong hàm đó.

                    if processed_count > 0:
                        st.session_state.file_results = pd.DataFrame(results)
                        st.success(f"✅ Hoàn tất dự đoán từ file Excel! Đã xử lý {processed_count} dòng.")
                    else:
                         st.warning("⚠️ Không có dòng nào trong file Excel được xử lý thành công.")
                         st.session_state.file_results = None
                         st.session_state.file_map = None


        except FileNotFoundError:
            st.error("❌ Lỗi: Không tìm thấy file Excel.")
        except pd.errors.EmptyDataError:
            st.error("❌ Lỗi: File Excel trống.")
            st.session_state.file_results = None
            st.session_state.file_map = None
        except pd.errors.ParserError:
             st.error("❌ Lỗi: Không thể đọc file Excel. Vui lòng kiểm tra định dạng file.")
             st.session_state.file_results = None
             st.session_state.file_map = None
        except Exception as e:
            st.error(f"❌ Lỗi không mong muốn khi xử lý file Excel: {e}")
            st.session_state.file_results = None
            st.session_state.file_map = None


# Hiển thị kết quả file (nếu đã có)
if st.session_state.file_results is not None:
    st.markdown("#### Kết quả dự đoán từ file")
    st.dataframe(st.session_state.file_results)

if st.session_state.file_map is not None:
    st.markdown("#### Bản đồ kết quả từ file")
    # Render bản đồ đã lưu trong session state
    st_folium(st.session_state.file_map, width=1300, height=500, key="file_map_display")


# ===============================================================
# ⌨️  DỰ ĐOÁN NHẬP TAY (FORM)
# ===============================================================
st.markdown("---\n### ⌨️ Dự đoán bằng cách nhập tay thông số")
st.info("Nhập thông tin của một trạm thu duy nhất để dự đoán vị trí nguồn phát.")

# Nút dự đoán nhập tay chỉ được enable khi có mô hình
manual_predict_disabled = st.session_state.model is None

with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        lat_rx = st.number_input("Vĩ độ trạm thu (°)", value=16.0, format="%.6f", help="Ví dụ: 16.0479")
        lon_rx = st.number_input("Kinh độ trạm thu (°)", value=108.0, format="%.6f", help="Ví dụ: 108.2209")
        azimuth = st.number_input("Góc phương vị (độ)", value=45.0, min_value=0.0, max_value=360.0, format="%.2f", help="Góc từ trạm thu đến nguồn phát, tính từ hướng Bắc theo chiều kim đồng hồ (0-360°)")
    with col2:
        h_rx = st.number_input("Chiều cao anten (m)", value=30.0, min_value=0.0, format="%.2f", help="Chiều cao anten so với mặt đất")
        signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80.0, format="%.2f", help="Cường độ tín hiệu đo được tại trạm thu")
        freq = st.number_input("Tần số (MHz)", value=900.0, min_value=1.0, format="%.2f", help="Tần số hoạt động của nguồn phát")

    submitted = st.form_submit_button(
        "🔍 Dự đoán từ thông số nhập tay",
        disabled=manual_predict_disabled
    )

if submitted:
    # Kiểm tra lại một lần nữa dù nút đã bị disabled
    if st.session_state.model is None:
         st.warning("⚠️ Vui lòng tải mô hình trước khi chạy dự đoán nhập tay.")
    else:
        # Thực hiện validation cơ bản cho nhập tay
        if not (-90 <= lat_rx <= 90) or not (-180 <= lon_rx <= 180):
            st.error("❌ Lỗi: Vĩ độ phải nằm trong khoảng [-90, 90] và Kinh độ phải nằm trong khoảng [-180, 180].")
            st.session_state.single_result = None
            st.session_state.single_map = None
        elif h_rx < 0 or signal > 0 or freq <= 0 or not (0 <= azimuth <= 360):
             st.error("❌ Lỗi: Vui lòng kiểm tra lại các giá trị nhập (chiều cao >= 0, tín hiệu <= 0, tần số > 0, phương vị 0-360).")
             st.session_state.single_result = None
             st.session_state.single_map = None
        else:
            with st.spinner("Đang dự đoán..."):
                # Sử dụng hàm predict_location_from_inputs chung
                lat_pred, lon_pred, pred_dist = predict_location_from_inputs(
                    st.session_state.model,
                    lat_rx, lon_rx, h_rx, signal, freq, azimuth
                )

            if lat_pred is not None and lon_pred is not None:
                # Lưu kết quả
                st.session_state.single_result = (lat_pred, lon_pred, pred_dist)

                # Tạo bản đồ MỚI cho kết quả nhập tay
                m2 = folium.Map(location=[lat_rx, lon_rx], zoom_start=10)
                folium.Marker([lat_rx, lon_rx],
                              tooltip=f"Trạm thu\nLat: {lat_rx:.4f}, Lon: {lon_rx:.4f}",
                              icon=folium.Icon(color='blue', icon='signal')).add_to(m2)
                folium.Marker([lat_pred, lon_pred],
                              tooltip=(f"Nguồn phát dự đoán\n"
                                       f"Tần số: {freq} MHz\n"
                                       f"Mức tín hiệu: {signal} dBm\n"
                                       f"Khoảng cách: {pred_dist:.2f} km"),
                              icon=folium.Icon(color='red', icon='info-sign')).add_to(m2)
                folium.PolyLine([[lat_rx, lon_rx], [lat_pred, lon_pred]],
                                color='green', weight=2.5, opacity=0.7).add_to(m2)
                st.session_state.single_map = m2

                st.success("✅ Đã tính xong toạ độ dự đoán!")
            else:
                # Nếu dự đoán lỗi, đảm bảo kết quả cũ bị xóa
                st.session_state.single_result = None
                st.session_state.single_map = None


# Hiển thị kết quả nhập tay (nếu đã có)
if st.session_state.single_result is not None:
    lat_pred, lon_pred, dist = st.session_state.single_result
    st.markdown("#### Kết quả dự đoán nhập tay")
    st.info(f"🎯 **Vĩ độ Nguồn**: {lat_pred:.6f}°  |  "
            f"**Kinh độ Nguồn**: {lon_pred:.6f}°  |  "
            f"**Khoảng cách (Dự đoán)**: {dist:.2f} km")

if st.session_state.single_map is not None:
    st.markdown("#### Bản đồ kết quả nhập tay")
    # Render bản đồ đã lưu trong session state
    st_folium(st.session_state.single_map, width=1300, height=500, key="single_map_display")

st.markdown("---")
st.write("Ứng dụng dự đoán vị trí nguồn phát xạ v1.2 (Debug)")
