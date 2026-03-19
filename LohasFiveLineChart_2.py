import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 網頁配置
st.set_page_config(page_title="長線股價五線譜", layout="wide")

# --- 側邊欄：查詢設定 ---
st.sidebar.header("查詢設定")
stock_id = st.sidebar.text_input("股票代號(如2330.TW或AAPL)", "2330.TW")
start_date = st.sidebar.date_input("起始日期(YYYY/MM/DD)", datetime(2015, 8, 1))
end_date = st.sidebar.date_input("結束日期(YYYY/MM/DD)", datetime.now())

theme_choice = st.sidebar.radio("圖表主題(對應網頁背景)", ["亮色(白色背景)", "深色(深色背景)"])

# --- 強制背景色切換邏輯 (CSS) ---
if theme_choice == "深色(深色背景)":
    chart_template = "plotly_dark"
    font_color = "white"
    bg_color = "#0E1117"
    st.markdown("""
        <style>
        /* 強制側邊欄、主背景、文字顏色為深色 */
        [data-testid="stSidebar"], .stApp, header { background-color: #0E1117 !important; color: white !important; }
        .stMarkdown, p, h1, h2, h3, span { color: white !important; }
        /* 調整輸入框文字顏色 */
        input { color: white !important; background-color: #262730 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    font_color = "black"
    bg_color = "#FFFFFF"
    st.markdown("""
        <style>
        /* 1. 強制背景與文字顏色 */
        [data-testid="stSidebar"], .stApp, header { 
            background-color: #FFFFFF !important; 
            color: black !important; 
        }
        .stMarkdown, p, h1, h2, h3, span { color: black !important; }
        
        /* 2. 徹底消除輸入框右側的陰影與淡淡格線 */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input {
            background-color: white !important;
            border-color: #dcdcdc !important; /* 設定一個淺灰色的統一邊框 */
            box-shadow: none !important;      /* 移除所有陰影 */
        }
        
        /* 針對日期選取器內部的特殊容器進行修正 */
        div[role="combobox"] {
            background-color: white !important;
            border: none !important;
        }

        /* 3. 強制按鈕內部的所有文字元素變白 */
        div.stButton > button {
            background-color: #000000 !important;
            border: 1px solid #000000 !important;
            font-weight: bold !important;
        }
        div.stButton > button * {
            color: #FFFFFF !important;
        }
        
        div.stButton > button:hover {
            background-color: #333333 !important;
        }

        /* 4. 側邊欄與輸入框整體調整 */
        [data-testid="stSidebar"] { border-right: 1px solid #f0f2f6; }
        input { 
            color: black !important; 
            background-color: white !important; 
        }
        </style>
        """, unsafe_allow_html=True)

# --- 1. 先定義按鈕 (放在側邊欄設定的最後面) ---
calculate_btn = st.sidebar.button("開始計算")

# --- 2. 處理搜尋 台股代號搜尋ID ---
search_id = f"{stock_id}.TW" if stock_id.isdigit() else stock_id

st.title("📈 長線股價五線譜")

# --- 3. 判斷邏輯：如果按鈕「還沒被按下」 ---
if not calculate_btn:
    st.info("💡 請點開左上角選單 [ > ] 設定參數後按「開始計算」。")
else:
    # --- 4. 按下按鈕後才執行的動作：抓取資料 ---
    data = yf.download(search_id, start=start_date, end=end_date, auto_adjust=True)
    
    # 關鍵修正：過濾掉空值，避免 nan 導致計算失敗
    if not data.empty:
        df = data.copy().reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 移除任何包含 NaN 的列
        df = df.dropna(subset=['Close'])
        
        df['Close_1D'] = df['Close'].values.flatten()
        df['Time_Idx'] = np.arange(len(df)) 
        
        # --- 計算股價五線譜 (線性回歸) ---
        if len(df) > 1:
            z = np.polyfit(df['Time_Idx'], df['Close_1D'], 1)
            p = np.poly1d(z)
            df['Trend_Line'] = p(df['Time_Idx'])
            
            std_dev = (df['Close_1D'] - df['Trend_Line']).std()
            df['Upper_2'] = df['Trend_Line'] + 2 * std_dev
            df['Upper_1'] = df['Trend_Line'] + 1 * std_dev
            df['Lower_1'] = df['Trend_Line'] - 1 * std_dev
            df['Lower_2'] = df['Trend_Line'] - 2 * std_dev

            # --- 繪製圖表 ---
            fig = go.Figure()
            
            # 收盤價
            line_color = "white" if theme_choice == "深色(深色背景)" else "black"
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close_1D'], name='收盤價', 
                                     line=dict(color=line_color, width=1.5)))
            
            # 五線譜線段
            colors = ['#FF4136', '#FF851B', '#0074D9', '#2ECC40', '#3D9970']
            names = ['極端樂觀', '樂觀', '趨勢中線', '悲觀', '極端悲觀']
            bands = ['Upper_2', 'Upper_1', 'Trend_Line', 'Lower_1', 'Lower_2']
            
            for idx, band in enumerate(bands):
                fig.add_trace(go.Scatter(x=df['Date'], y=df[band], name=names[idx], 
                                         line=dict(dash='dash' if 'Trend' not in band else 'solid', 
                                                   color=colors[idx], width=1)))

            fig.update_layout(
                height=600, 
                template=chart_template, 
                hovermode='x unified',
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=font_color),
                xaxis=dict(color=font_color, tickfont=dict(color=font_color)),
                yaxis=dict(color=font_color, tickfont=dict(color=font_color)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color=font_color))
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 數據摘要 ---
            st.header("📊 數據摘要")
            last_price = df['Close_1D'].iloc[-1]
            trend_val = df['Trend_Line'].iloc[-1]
            sigma = (last_price - trend_val) / std_dev
            
            col1, col2, col3 = st.columns(3)
            col1.metric("最後收盤價", f"{last_price:.2f}")
            col2.metric("回歸中線", f"{trend_val:.2f}")
            col3.metric("目前區間", f"{sigma:.2f} σ")
            
            if sigma > 2:
                st.error(f"⚠️ 目前價格極度高估（高於中線 {sigma:.2f} 個標準差）")
            elif sigma < -2:
                st.success(f"✅ 目前價格極度低估（低於中線 {abs(sigma):.2f} 個標準差）")
            else:
                st.info(f"💡 目前價格處於合理回歸區間內")
        else:
            st.warning("資料量不足以計算回歸線。")
    else:
        st.error("找不到資料，請檢查代號是否正確。")