import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pvlib
from zoneinfo import ZoneInfo
import requests
from FlightRadar24 import FlightRadar24API

# PEP8準拠
# Page config
st.set_page_config(
    page_title="Flight Glare Checker - 飛行機の窓際まぶしさ判定",
    page_icon="✈️",
    layout="wide"
)

def fetch_flight_data(flight_number: str) -> tuple:
    """
    FlightRadarAPIおよび公開検索APIから便名を検索し、
    直近の完了済みフライトの詳細データを取得して(DataFrame, 情報辞書)として返す。
    """
    flight_list_url = f"https://api.flightradar24.com/common/v1/flight/list.json?query={flight_number}&fetchBy=flight&page=1&limit=25"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }
    
    try:
        res = requests.get(flight_list_url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        st.error(f"FlightRadar24の検索APIへの通信に失敗しました: {e}")
        return None, None
        
    try:
        flights = data.get('result', {}).get('response', {}).get('data', [])
        target_flight_id = None
        
        # arrival(到着済み)のステータスを持つ直近のフライトを探す
        for f in flights:
            status = f.get('status', {}).get('generic', {}).get('status', {}).get('type')
            if status == 'arrival':
                if f.get('identification', {}).get('id'):
                    target_flight_id = f['identification']['id']
                    break
                    
        if not target_flight_id:
            st.error(f"便名 '{flight_number}' の直近の到着済みフライト履歴が見つかりませんでした。")
            return None, None

        # FR24 APIを使用して詳細（trail等）を取得
        fr_api = FlightRadar24API()
        
        # IDだけで検索するためのダミークラス
        class DummyFlight:
            def __init__(self, obj_id):
                self.id = obj_id
        
        details = fr_api.get_flight_details(DummyFlight(target_flight_id))
        
        # フライト付加情報の取得
        origin = details.get('airport', {}).get('origin', {}).get('name', '不明')
        destination = details.get('airport', {}).get('destination', {}).get('name', '不明')
        aircraft = '不明'
        if 'aircraft' in details and details['aircraft']:
            aircraft = details['aircraft'].get('model', {}).get('text', '不明')
            
        flight_info = {
            'origin': origin,
            'destination': destination,
            'aircraft': aircraft,
            'flight_number': flight_number
        }
        
        trail = details.get('trail', [])
        if not trail:
            st.error("このフライトのトラッキングデータ（軌跡）が存在しません。")
            return None, None
            
        records = []
        for point in trail:
            lat = point.get('lat')
            lon = point.get('lng')
            alt = point.get('alt')
            hd = point.get('hd')
            ts = point.get('ts')
            
            # データが欠損している地点はスキップ
            if None in (lat, lon, alt, hd, ts):
                continue
                
            utc_time = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            records.append({
                'Position': f"{lat},{lon}",
                'Altitude': alt,
                'Direction': hd,
                'UTC': utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            })
            
        df = pd.DataFrame(records)
        return df, flight_info
            
    except Exception as e:
        st.error(f"フライトデータの解析中にエラーが発生しました: {e}")
        return None, None


def analyze_flight_data(df: pd.DataFrame) -> tuple:
    """
    FlightRadar24等のDataFrameを読み込み、ベクトル化処理を用いて一括で眩しさ判定を行う。
    戻り値: (df, timeline, summary_dict)
    """
    
    # 1. 前処理
    df[['lat', 'lon']] = df['Position'].str.split(',', expand=True).astype(float)
    df['alt_m'] = df['Altitude'] * 0.3048
    
    # 10,000 ft 以上のデータにフィルタリング
    df = df[df['Altitude'] >= 10000].copy()
    if df.empty:
        st.warning("10,000フィート（巡航高度）以上のデータが存在しません。")
        return None, None, None
        
    # 時刻処理
    df['dt_utc'] = pd.to_datetime(df['UTC'])
    df['datetime_jst'] = df['dt_utc'].dt.tz_convert('Asia/Tokyo')
    
    # 昇順ソート
    df = df.sort_values('dt_utc').reset_index(drop=True)
    
    # 2. pvlib等を用いた一括計算
    times = pd.DatetimeIndex(df['dt_utc'])
    lats = df['lat'].values
    lons = df['lon'].values
    alts = df['alt_m'].values
    headings = df['Direction'].values
    
    solpos = pvlib.solarposition.get_solarposition(times, lats, lons, altitude=alts)
    sun_az = solpos['azimuth'].values
    sun_alt = solpos['apparent_elevation'].values
    
    rel_angle = (sun_az - headings) % 360.0
    
    # 窓の物理的な配置に基づく判定
    conditions = [
        sun_alt < 0,
        (rel_angle >= 45) & (rel_angle < 135),
        (rel_angle >= 135) & (rel_angle < 225),
        (rel_angle >= 225) & (rel_angle < 315),
        (rel_angle >= 315) | (rel_angle < 45)
    ]
    choices = [
        "夜間（太陽が出ていない）",
        "右側(K席側)から眩しい",
        "後方（機体に遮られるため眩しくない）",
        "左側(A席側)から眩しい",
        "正面（景色が見やすい）"
    ]
    pos_str = np.select(conditions, choices, default="不明")
    
    is_glaring = np.where(
        sun_alt < 0, False,
        np.where(((rel_angle >= 45) & (rel_angle < 135)) | ((rel_angle >= 225) & (rel_angle < 315)), True, False)
    )
    
    df['sun_azimuth_deg'] = np.round(sun_az, 2)
    df['sun_altitude_deg'] = np.round(sun_alt, 2)
    df['relative_angle_deg'] = np.where(sun_alt < 0, np.nan, np.round(rel_angle, 2))
    df['is_glaring'] = is_glaring
    df['position_detail'] = pos_str
    
    # 3. タイムラインの生成
    df_copy = df.copy()
    df_copy['block'] = (df_copy['position_detail'] != df_copy['position_detail'].shift()).cumsum()
    timeline = df_copy.groupby('block').agg(
        start_time=('datetime_jst', 'min'),
        end_time=('datetime_jst', 'max'),
        position_detail=('position_detail', 'first'),
        is_glaring=('is_glaring', 'first')
    ).reset_index(drop=True)
    
    # 表示用の整形
    timeline['時間帯'] = timeline.apply(
        lambda r: f"{r['start_time'].strftime('%H:%M:%S')} 〜 {r['end_time'].strftime('%H:%M:%S')}", axis=1)
    
    # 4. サマリー集計 (秒単位での微分積算)
    df['duration_sec'] = (-df['dt_utc'].diff(-1).dt.total_seconds()).fillna(0)
    
    a_glaring_sec = df.loc[df['position_detail'] == "左側(A席側)から眩しい", 'duration_sec'].sum()
    k_glaring_sec = df.loc[df['position_detail'] == "右側(K席側)から眩しい", 'duration_sec'].sum()
    front_sec = df.loc[df['position_detail'] == "正面（景色が見やすい）", 'duration_sec'].sum()
    back_sec = df.loc[df['position_detail'] == "後方（機体に遮られるため眩しくない）", 'duration_sec'].sum()
    night_sec = df.loc[df['position_detail'] == "夜間（太陽が出ていない）", 'duration_sec'].sum()
    
    total_sec = df['duration_sec'].sum()
    
    summary = {
        "total_min": int(total_sec // 60),
        "a_glaring_min": int(a_glaring_sec // 60),
        "k_glaring_min": int(k_glaring_sec // 60),
        "front_min": int(front_sec // 60),
        "back_min": int(back_sec // 60),
        "night_min": int(night_sec // 60),
        "not_glaring_min": int((front_sec + back_sec + night_sec) // 60)
    }
    
    return df, timeline, summary

def main():
    st.title("☀️ Flight Glare Checker - 飛行機の窓際まぶしさ判定")
    st.markdown("""
        このアプリは、Flightradar24のデータを元に、フライト中のどの時間帯にどちらの窓から太陽光が入るかをシミュレーションします。
        便名を入力して「自動判定スタート」を押すだけで直近のフライト履歴を自動解析します。
        座席選びの参考にしてください（高度10,000フィート以上の巡航中データのみを対象としています）。
    """)
    
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        flight_number_input = st.text_input("便名を入力してください（例: JL901, NH15）", value="JL901")
    with col_btn:
        st.write("") # アライメント用
        st.write("") 
        is_submitted = st.button("自動判定スタート", type="primary", use_container_width=True)
    
    if is_submitted and flight_number_input:
        # 入力文字列の正規化（大文字化・空白削除）
        flight_number = flight_number_input.upper().strip().replace(" ", "")
        
        # ユーザーフレンドリー対応：一般的な3レターコード(ICAO)を2レターコード(IATA)に自動変換
        airline_map = {
            # 日本のエアライン
            "JAL": "JL", "ANA": "NH", "SKY": "BC", "ADO": "HD", "SNJ": "6J", 
            "SFJ": "7G", "FDA": "JH", "APJ": "MM", "JJP": "GK", "JTA": "NU", 
            "RAC": "RC", "JAC": "JC", "AHX": "MZ", "NCA": "KZ", "IBX": "FW", 
            "ORC": "OC", "SJO": "IJ", "TZP": "ZG",
            # Oneworld
            "AAL": "AA", "BAW": "BA", "CPA": "CX", "FIN": "AY", "IBE": "IB",
            "MAS": "MH", "QFA": "QF", "QTR": "QR", "RAM": "AT", "RJA": "RJ",
            "ALK": "UL", "ASQ": "AS",
            # Star Alliance
            "ACA": "AC", "CCA": "CA", "AIC": "AI", "ANZ": "NZ", "AAR": "OZ",
            "AUA": "OS", "AVA": "AV", "BEL": "SN", "CMP": "CM", "EVA": "BR",
            "DLH": "LH", "LOT": "LO", "SIA": "SQ", "SAA": "SA", "SWR": "LX",
            "TAP": "TP", "THA": "TG", "THY": "TK", "UAL": "UA", "CSZ": "ZH",
            "CRO": "OU", "EGY": "MS", "ETH": "ET", "AEE": "A3",
            # SkyTeam
            "ARG": "AR", "AMX": "AM", "AEA": "UX", "AFR": "AF", "DAL": "DL",
            "CAL": "CI", "CES": "MU", "CSA": "OK", "GIA": "GA", "KQA": "KQ",
            "KLM": "KL", "KOR": "KE", "MEA": "ME", "SAS": "SK", "SVA": "SV",
            "TAR": "RO", "VNM": "VN", "VIR": "VS",
            # その他主要・リクエスト特定（中国南方航空、スターラックス等）
            "CSN": "CZ", "SJX": "JX", "UAE": "EK", "ETD": "EY", "JST": "JQ",
            "SWA": "WN", "RYR": "FR", "HKE": "UO"
        }
        for icao, iata in airline_map.items():
            if flight_number.startswith(icao):
                flight_number = flight_number.replace(icao, iata, 1)
                break

        with st.spinner(f"FlightRadar24から '{flight_number}' のフライトデータを取得し、解析を行っています..."):
            df_flight, flight_info = fetch_flight_data(flight_number)
            
            if df_flight is not None:
                df, timeline, summary = analyze_flight_data(df_flight)
                
                if df is not None:
                    st.success("データの取得と解析が完了しました！")
                    
                    # --- フライト情報の表示 ---
                    st.header(f"✈️ フライト情報: {flight_info['flight_number']}")
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric(label="出発地 (Origin)", value=flight_info['origin'])
                    with info_col2:
                        st.metric(label="到着地 (Destination)", value=flight_info['destination'])
                    with info_col3:
                        st.metric(label="機材 (Aircraft)", value=flight_info['aircraft'])
                        
                    st.markdown("---")
            
                    # --- 最終結論 ---
                    st.header("🎯 最終結論（おすすめの座席）")
                    
                    total_m = summary["total_min"]
                    a_m = summary["a_glaring_min"]
                    k_m = summary["k_glaring_min"]
                    diff_m = abs(a_m - k_m)
                    
                    if (a_m < total_m * 0.05 and a_m < 10) and (k_m < total_m * 0.05 and k_m < 10):
                        st.success("✨ どちらの席も眩しい時間はごくわずかで快適なフライトです！お好みの景色が見える席を選んでOKです。")
                    elif diff_m < total_m * 0.10 and diff_m <= 15:
                        st.info("⚖️ A席・K席で眩しさに大きな差はありません。時間帯によってどちらも同じくらいの日差しを受ける可能性があります。")
                    elif a_m > k_m:
                        st.info("💡 **右側（K席側）** を予約した方が、眩しい時間が短く快適に過ごせます。")
                    elif k_m > a_m:
                        st.info("💡 **左側（A席側）** を予約した方が、眩しい時間が短く快適に過ごせます。")
                    else:
                        st.warning("⚖️ A席・K席での明確な有利不利はありません。")
                        
                    st.markdown("---")
            
            # --- サマリー (KPI) ---
            st.header("📊 フライト眩しさ サマリー")
            st.caption(f"フライト総時間 (10,000ft以上) : 約 {summary['total_min']} 分")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="⬅️ A席側(左側)が眩しい時間", value=f"{summary['a_glaring_min']} 分")
            with col2:
                st.metric(label="➡️ K席側(右側)が眩しい時間", value=f"{summary['k_glaring_min']} 分")
            with col3:
                st.metric(label="✨ どちらでも景色を楽しめる時間", value=f"{summary['not_glaring_min']} 分", 
                          help=f"内訳 - 正面: {summary['front_min']}分, 後方: {summary['back_min']}分, 夜間: {summary['night_min']}分")
            
            st.markdown("---")
            
            # --- タイムライン ---
            st.header("⏱️ 時間帯ごとの詳細（タイムライン）")
            # タイムラインの表示用整形
            display_tl = timeline[['時間帯', 'position_detail']].rename(columns={'position_detail': '眩しさの状況'})
            
            # 行ごとに色を変える等の装飾も可能だが、今回はシンプルにテーブル表示
            st.table(display_tl)
            
            st.markdown("---")
            
            # --- 詳細データ(生データ+解析結果) ---
            with st.expander("詳細データを確認する"):
                show_cols = [
                    'datetime_jst', 'Altitude', 'Direction', 
                    'sun_azimuth_deg', 'sun_altitude_deg', 'relative_angle_deg', 
                    'is_glaring', 'position_detail'
                ]
                # 存在しない列は除外
                exist_cols = [c for c in show_cols if c in df.columns]
                st.dataframe(df[exist_cols], use_container_width=True)

if __name__ == "__main__":
    main()
