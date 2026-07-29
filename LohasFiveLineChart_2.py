import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# 1. 系統環境配置
# ==============================================================================

st.set_page_config(page_title="長線股價五線譜", layout="wide")

if "is_dark" not in st.session_state:
    st.session_state.is_dark = False

# ==============================================================================
# 2. 視覺設計 Tokens（依 Design Handoff「Classical」設計系統規格）
# ==============================================================================

LIGHT_TOKENS = {
    "bg": "#f3f2f2",
    "surface_alt": "#f8f7f6",
    "text": "#201f1d",
    "text_muted": "rgba(32,31,29,0.55)",
    "divider": "rgba(32,31,29,0.16)",
    "accent": "#b68235",
    "grid_line": "rgba(32,31,29,0.08)",
    "shadow": "0 3px 10px rgba(45,43,43,0.14)",
    "lines": {
        "close": "#1a3a6b", "extreme_bull": "#8b1e1e", "bull": "#e2726b",
        "trend": "#d4a017", "bear": "#7fb88f", "extreme_bear": "#1f5c3d",
    },
}
DARK_TOKENS = {
    "bg": "#17140f",
    "surface_alt": "#1c1912",
    "text": "#f3ede2",
    "text_muted": "rgba(243,237,226,0.6)",
    "divider": "rgba(243,237,226,0.16)",
    "accent": "#c99a4e",
    "grid_line": "rgba(243,237,226,0.12)",
    "shadow": "0 12px 32px rgba(0,0,0,0.5)",
    "lines": {
        "close": "#5b9bf0", "extreme_bull": "#d0342c", "bull": "#f0918a",
        "trend": "#e6b325", "bear": "#8fd6a3", "extreme_bear": "#2f9d68",
    },
}

is_dark = st.session_state.is_dark
tok = DARK_TOKENS if is_dark else LIGHT_TOKENS
chart_template = "plotly_dark" if is_dark else "plotly_white"

# ==============================================================================
# 3. 全域樣式（字型／配色／按鈕／輸入框／側邊欄版面）
# ==============================================================================

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Lora:wght@400;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Lora', serif; }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
    }}

    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {tok['bg']} !important;
        color: {tok['text']} !important;
    }}
    [data-testid="stHeader"] {{ background-color: transparent !important; }}

    /* 側邊欄版面 */
    [data-testid="stSidebar"] {{
        background-color: {tok['bg']} !important;
        border-right: 1px solid {tok['divider']};
        min-width: 300px !important;
        max-width: 300px !important;
    }}
    [data-testid="stSidebarUserContent"] {{ padding: 40px 28px !important; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 18px; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 12px !important;
        color: {tok['text_muted']} !important;
        font-family: 'Lora', serif !important;
    }}

    /* 輸入框：邊框只畫在最外層容器，內層元素一律去邊框/去底色，避免雙層邊框與底色不一致 */
    [data-testid="stSidebar"] div[data-baseweb="input"] {{
        border: 1px solid {tok['divider']} !important;
        border-radius: 4px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="input"] * {{
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="input"] input {{
        color: {tok['text']} !important;
    }}

    /* 瀏覽器自動填入(autofill)會強制套用自己的底色，一般CSS蓋不掉，需用此專門技巧解除 */
    [data-testid="stSidebar"] input:-webkit-autofill,
    [data-testid="stSidebar"] input:-webkit-autofill:hover,
    [data-testid="stSidebar"] input:-webkit-autofill:focus,
    [data-testid="stSidebar"] input:-webkit-autofill:active {{
        -webkit-box-shadow: 0 0 0 1000px transparent inset !important;
        -webkit-text-fill-color: {tok['text']} !important;
        caret-color: {tok['text']};
        transition: background-color 9999s ease-in-out 0s;
    }}

    /* 按鈕：設計系統一律採 outline 樣式，不使用實心填色 */
    div.stButton > button {{
        width: 100%;
        background-color: transparent !important;
        border-radius: 4px !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: none !important;
    }}
    div.stButton > button[kind="secondary"] {{
        color: {tok['text']} !important;
        border: 1px solid {tok['divider']} !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        border-color: {tok['accent']} !important;
        color: {tok['accent']} !important;
    }}
    div.stButton > button[kind="primary"] {{
        color: {tok['accent']} !important;
        border: 1px solid {tok['accent']} !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: color-mix(in srgb, {tok['accent']} 12%, transparent) !important;
    }}

    /* 圖表卡片容器（st.container(border=True)） */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {tok['divider']} !important;
        border-radius: 7px !important;
        background-color: {tok['surface_alt']} !important;
        box-shadow: {tok['shadow']};
    }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. 側邊欄：使用者參數輸入區
# ==============================================================================

st.sidebar.markdown(
    f"<div style='font-size:12px;letter-spacing:.08em;text-transform:uppercase;"
    f"color:{tok['text_muted']};margin-bottom:-6px;'>查詢設定</div>",
    unsafe_allow_html=True,
)

# 股票代號輸入：預設為 2330.TW (台積電)
stock_id = st.sidebar.text_input("股票代號(如2330或AAPL)", "2330")
# 日期範圍選擇：設定資料擷取的起始與結束時間
start_date = st.sidebar.date_input("起始日期(YYYY/MM/DD)", datetime(2015, 8, 1))
end_date = st.sidebar.date_input("結束日期(YYYY/MM/DD)", datetime.now())

st.sidebar.markdown(
    f"<hr style='border:none;border-top:1px solid {tok['divider']};margin:4px 0;'>",
    unsafe_allow_html=True,
)

# 視覺主題切換：以按鈕切換整頁亮/深色（對應網頁背景）
theme_label = "切換為深色" if not is_dark else "切換為亮色"
theme_icon = ":material/dark_mode:" if not is_dark else ":material/light_mode:"
theme_clicked = st.sidebar.button(theme_label, icon=theme_icon, type="secondary", use_container_width=True)

# 開始計算觸發按鈕
calculate_btn = st.sidebar.button("開始計算", type="primary", use_container_width=True)

if theme_clicked:
    st.session_state.is_dark = not st.session_state.is_dark
    st.rerun()

# ==============================================================================
# 5. 主程式執行邏輯
# ==============================================================================

st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;">
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="{tok['accent']}" stroke-width="1.6">
        <path d="M3 17l5-6 4 3 6-9"/><path d="M14 5h5v5"/>
      </svg>
      <h1 style="margin:0;font-size:36px;">長線股價五線譜</h1>
    </div>
    """, unsafe_allow_html=True)

# 判斷邏輯：如果按鈕「還沒被按下」
if not calculate_btn:
    st.markdown(
        f"<div style='color:{tok['text_muted']};font-size:15px;margin-top:12px;'>"
        f"請點開左上角選單 [ >> ] 在左側面板設定參數後，按「開始計算」即可產出圖表</div>",
        unsafe_allow_html=True,
    )
# 判斷邏輯：按下按鈕後才執行抓取資料的動作：
else:
    # 若使用者已輸入含 . 的代號（如 2330.TW），直接使用；否則依序嘗試原始、.TW、.TWO
    if '.' in stock_id:
        candidates = [stock_id]
    else:
        candidates = [stock_id, f"{stock_id}.TW", f"{stock_id}.TWO"]

    data = pd.DataFrame()
    search_id = candidates[0]
    for candidate in candidates:
        _dl = yf.download(candidate, start=start_date, end=end_date, auto_adjust=True)
        if not _dl.empty:
            data = _dl
            search_id = candidate
            break

    # 顯示最終使用的股票代碼
    st.markdown(
        f"<div style='margin-top:4px;font-family:\"Cormorant Garamond\",serif;"
        f"font-size:19px;color:{tok['text_muted']};'>{search_id}</div>",
        unsafe_allow_html=True,
    )

    if not data.empty:
        # 相容新版 yfinance MultiIndex 欄位結構：直接萃取 Close 欄位，避免 reset_index 後欄名不穩定
        close_series = data['Close']
        # 若為 DataFrame（新版 yfinance 多層索引會回傳 DataFrame），取第一欄
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        # 建立乾淨的工作用 DataFrame，明確指定欄位名稱
        df = pd.DataFrame({
            'Date': close_series.index,
            'Close_1D': close_series.values.flatten()
        }).reset_index(drop=True)

        # 移除時區資訊（yfinance 新版可能回傳 tz-aware 日期，Plotly 繪圖需純日期）
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # 排除無交易資料的日期 (NaN)，確保計算精確度
        df = df.dropna(subset=['Close_1D'])
        # 建立時間索引 (0, 1, 2...) 作為線性回歸的自變數 X
        df['Time_Idx'] = np.arange(len(df))

        # --- B. 核心原理：線性回歸與標準差軌道計算 ---
        if len(df) > 1:
            # 1. 線性回歸：找出股價的中長線趨勢中心
            # 公式為：$y = ax + b$ (a=斜率, b=截距)
            z = np.polyfit(df['Time_Idx'], df['Close_1D'], 1)
            p = np.poly1d(z)
            df['Trend_Line'] = p(df['Time_Idx'])  # 產出回歸趨勢線數據

            # 2. 計算離差標準差 (Standard Deviation)：衡量股價對趨勢線的波動幅度
            # 標準差愈大，代表股價波動愈劇烈，軌道間距也會隨之拉大
            std_dev = (df['Close_1D'] - df['Trend_Line']).std()

            # 3. 建立五線譜軌道
            # 高點軌道：趨勢中線 + 1倍或2倍標準差
            # 低點軌道：趨勢中線 - 1倍或2倍標準差
            df['Upper_2'] = df['Trend_Line'] + 2 * std_dev  # 極端樂觀線
            df['Upper_1'] = df['Trend_Line'] + 1 * std_dev  # 樂觀線
            df['Lower_1'] = df['Trend_Line'] - 1 * std_dev  # 悲觀線
            df['Lower_2'] = df['Trend_Line'] - 2 * std_dev  # 極端悲觀線

            # --- C. 視覺化繪圖 (Plotly 互動式圖表) ---
            fig = go.Figure()
            lines = tok['lines']

            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close_1D'], name='收盤價',
                                     line=dict(color=lines['close'], width=2)))

            # 定義各軌道線的顏色與名稱（依 Design Handoff 配色 tokens）
            band_specs = [
                ('Upper_2', '極端樂觀', lines['extreme_bull']),
                ('Upper_1', '樂觀', lines['bull']),
                ('Trend_Line', '趨勢中線', lines['trend']),
                ('Lower_1', '悲觀', lines['bear']),
                ('Lower_2', '極端悲觀', lines['extreme_bear']),
            ]

            for band, name, color in band_specs:
                # 趨勢中線使用實線，其餘 SD 軌道使用虛線以利區分
                is_trend = (band == 'Trend_Line')
                fig.add_trace(go.Scatter(x=df['Date'], y=df[band], name=name,
                                         line=dict(dash='solid' if is_trend else 'dash',
                                                   color=color, width=2 if is_trend else 1.6)))

            # 圖表版面優化
            fig.update_layout(
                height=600,
                template=chart_template,
                hovermode='x unified',  # 統一滑鼠懸停資訊
                paper_bgcolor=tok['surface_alt'],
                plot_bgcolor=tok['surface_alt'],
                font=dict(color=tok['text'], family='Lora, serif'),
                xaxis=dict(color=tok['text'], tickfont=dict(color=tok['text']), gridcolor=tok['grid_line'],
                           zeroline=True, zerolinecolor=tok['grid_line'], zerolinewidth=1),
                yaxis=dict(color=tok['text'], tickfont=dict(color=tok['text']), gridcolor=tok['grid_line'],
                           zeroline=True, zerolinecolor=tok['grid_line'], zerolinewidth=1),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                            font=dict(color=tok['text_muted'], family='Lora, serif'))
            )

            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

            # --- D. 數據摘要與投資評語 ---
            st.markdown("<h4 style='margin:28px 0 16px;'>數據摘要</h4>", unsafe_allow_html=True)

            last_price = df['Close_1D'].iloc[-1]  # 最後一筆交易價格
            trend_val = df['Trend_Line'].iloc[-1]  # 對應日期的趨勢線值
            # 計算目前股價落在幾倍標準差位置 ($\sigma$ 值)
            sigma = (last_price - trend_val) / std_dev

            def _stat_card(label, value):
                return f"""
                <div style="border:1px solid {tok['divider']};border-radius:4px;padding:20px 22px;">
                  <div style="font-size:12px;color:{tok['text_muted']};margin-bottom:8px;">{label}</div>
                  <div style="font-family:'Cormorant Garamond',serif;font-size:28px;font-variant-numeric:tabular-nums;">{value}</div>
                </div>
                """

            # 儀表板數值呈現
            col1, col2, col3 = st.columns(3)
            col1.markdown(_stat_card("最後收盤價", f"{last_price:.2f}"), unsafe_allow_html=True)
            col2.markdown(_stat_card("回歸中線", f"{trend_val:.2f}"), unsafe_allow_html=True)
            col3.markdown(_stat_card("目前區間", f"{sigma:.2f} σ"), unsafe_allow_html=True)

            # 根據 $\sigma$ (Sigma) 值給予自動化投資評註
            if sigma > 2:
                banner_text = f"目前價格極度高估（高於中線 {sigma:.2f} 個標準差）"
            elif sigma < -2:
                banner_text = f"目前價格極度低估（低於中線 {abs(sigma):.2f} 個標準差）"
            else:
                banner_text = "目前價格處於合理回歸區間內"

            st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-top:20px;
                            border:1px solid color-mix(in srgb, {tok['accent']} 40%, transparent);
                            background:color-mix(in srgb, {tok['accent']} 10%, transparent);
                            border-radius:4px;padding:16px 20px;">
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="{tok['accent']}"
                       stroke-width="1.8" style="flex-shrink:0">
                    <path d="M9 18h6M10 21h4M12 3a6 6 0 00-3.6 10.8c.5.4.8 1 .8 1.7v.5h5.6v-.5c0-.7.3-1.3.8-1.7A6 6 0 0012 3z"/>
                  </svg>
                  <span style="font-size:14px;">{banner_text}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("資料量不足以計算回歸線。")
    else:
        st.error(f"找不到股票資料（已嘗試：{', '.join(candidates)}），請檢查代號或日期。")
