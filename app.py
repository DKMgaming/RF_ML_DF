import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import math

def calculate_destination(lat, lon, azimuth, distance_km):
    R = 6371.0  # Earth radius in km
    bearing = math.radians(azimuth)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d_by_r = distance_km / R

    lat2 = math.asin(math.sin(lat1) * math.cos(d_by_r) +
                     math.cos(lat1) * math.sin(d_by_r) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_by_r) * math.cos(lat1),
                             math.cos(d_by_r) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

st.set_page_config(layout="wide")
st.title("Dự đoán tọa độ nguồn phát xạ")

uploaded_file = st.file_uploader("Tải file Excel dự đoán (gồm tọa độ trạm thu, độ cao anten, mức tín hiệu, azimuth)...", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dữ liệu đầu vào:", df)

    # Giả lập tọa độ dự đoán (cần thay bằng mô hình ML thực tế)
    df['lat_pred'] = df['lat_receiver'] + 0.05
    df['lon_pred'] = df['lon_receiver'] + 0.05

    # Bản đồ hiển thị
    m = folium.Map(location=[df['lat_receiver'].mean(), df['lon_receiver'].mean()], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        folium.Marker(location=[row['lat_receiver'], row['lon_receiver']],
                      popup=f"Trạm thu\nLat: {row['lat_receiver']}\nLon: {row['lon_receiver']}",
                      icon=folium.Icon(color='blue', icon='tower', prefix='fa')).add_to(marker_cluster)

        folium.Marker(location=[row['lat_pred'], row['lon_pred']],
                      popup=f"Dự đoán nguồn phát\nLat: {row['lat_pred']}\nLon: {row['lon_pred']}",
                      icon=folium.Icon(color='red', icon='signal', prefix='fa')).add_to(marker_cluster)

        folium.PolyLine(locations=[[row['lat_receiver'], row['lon_receiver']],
                                   [row['lat_pred'], row['lon_pred']]],
                        color='green', weight=2, tooltip="Tia dự đoán").add_to(m)

        # Vẽ tia định hướng 50 km theo góc phương vị từ trạm thu
        lat_end, lon_end = calculate_destination(row['lat_receiver'], row['lon_receiver'], row['azimuth'], 50)
        folium.PolyLine(locations=[[row['lat_receiver'], row['lon_receiver']], [lat_end, lon_end]],
                        color='orange', weight=2, dash_array='5, 5', tooltip="Tia định hướng").add_to(m)

    st_data = st_folium(m, width=1000, height=600)
else:
    st.subheader("Hoặc nhập thông tin trực tiếp để dự đoán")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Vĩ độ trạm thu", value=21.0)
        lon = st.number_input("Kinh độ trạm thu", value=105.0)
        h = st.number_input("Độ cao anten (m)", value=10)
    with col2:
        signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80)
        azimuth = st.number_input("Góc phương vị (độ)", value=45)

    if st.button("Dự đoán"):
        # Giả lập tọa độ dự đoán (thực tế là mô hình máy học)
        lat_pred = lat + 0.05
        lon_pred = lon + 0.05

        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon], popup="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker([lat_pred, lon_pred], popup="Nguồn phát (dự đoán)", icon=folium.Icon(color='red')).add_to(m)
        folium.PolyLine(locations=[[lat, lon], [lat_pred, lon_pred]], color='green', weight=2, tooltip="Tia dự đoán").add_to(m)

        # Tính và vẽ tia định hướng theo azimuth dài 50km
        lat_end, lon_end = calculate_destination(lat, lon, azimuth, 50)
        folium.PolyLine(locations=[[lat, lon], [lat_end, lon_end]],
                        color='orange', weight=2, dash_array='5, 5', tooltip="Tia định hướng").add_to(m)

        st_folium(m, width=1000, height=600)
