import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import tempfile
import folium
from streamlit_folium import folium_static

# Thiết lập tiêu đề
st.set_page_config(page_title="Dự đoán toạ độ nguồn phát xạ", layout="wide")
st.title("🔍 Dự đoán Tọa độ Nguồn Phát Xạ")

# Upload mô hình dạng .keras
uploaded_model = st.file_uploader("📁 Tải mô hình (.keras)", type=["keras"])

model = None
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(uploaded_model.read())
        tmp_path = tmp.name
    model = load_model(tmp_path)
    st.success("✅ Mô hình đã được nạp thành công!")

# Tải dữ liệu
uploaded_data = st.file_uploader("📄 Tải dữ liệu Excel", type=["xls", "xlsx"])

if model and uploaded_data:
    df = pd.read_excel(uploaded_data)

    required_columns = ['longitude_rx', 'latitude_rx', 'antenna_height', 'rssi', 'frequency', 'azimuth']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Dữ liệu thiếu các cột: {', '.join(missing_columns)}")
    else:
        df['elevation'] = 0  # Thêm độ cao địa hình mặc định = 0

        # Dự đoán tọa độ nguồn phát
        features = df[['longitude_rx', 'latitude_rx', 'antenna_height', 'rssi', 'elevation', 'frequency', 'azimuth']]
        predictions = model.predict(features)

        df['predicted_longitude_tx'] = predictions[:, 0]
        df['predicted_latitude_tx'] = predictions[:, 1]

        # Hiển thị bảng kết quả
        st.subheader("📊 Kết quả Dự đoán")
        st.dataframe(df)

        # Bản đồ
        st.subheader("🗺️ Bản đồ Vị trí Trạm Thu và Nguồn Phát")

        m = folium.Map(location=[df['latitude_rx'].mean(), df['longitude_rx'].mean()], zoom_start=6)

        for i, row in df.iterrows():
            # Marker Trạm thu
            folium.Marker(
                location=[row['latitude_rx'], row['longitude_rx']],
                popup=f"Trạm Thu #{i+1}",
                icon=folium.Icon(color="blue", icon="tower")
            ).add_to(m)

            # Marker Nguồn phát dự đoán
            folium.Marker(
                location=[row['predicted_latitude_tx'], row['predicted_longitude_tx']],
                popup=f"Dự đoán Phát #{i+1}",
                icon=folium.Icon(color="red", icon="send")
            ).add_to(m)

            # Tia định hướng
            folium.PolyLine(
                locations=[
                    [row['latitude_rx'], row['longitude_rx']],
                    [row['predicted_latitude_tx'], row['predicted_longitude_tx']]
                ],
                color="green",
                weight=2,
                dash_array='5,10'
            ).add_to(m)

        folium_static(m)
