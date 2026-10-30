import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import plotly.graph_objects as go
from google import genai

# ==========================================
# 1. PAGE SETUP & UI CONTROLS
# ==========================================
st.set_page_config(page_title="IDX AI Screener & Analyst", layout="wide", page_icon="🤖")
st.title("🤖 IDX Screener with AI Analyst Agent")
st.markdown("Automated scan for the Indonesian Stock Exchange powered by Technicals, Fundamentals, and Gemini AI Analysis.")

st.sidebar.header("⚙️ Strategy Mode")
strategy_mode = st.sidebar.radio(
    "Select Screener Type", 
    [
        "🚀 Active Breakout (Volume Spike)", 
        "⏳ Pre-Breakout Coiling (Inside Bar)",
        "🎯 Support Bounce (Bullish Pin Bar)",
        "💼 Deep Value & Cash-Rich Fundamental"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Agent Settings")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to enable AI chart & stock analysis.")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ General Filters")
min_price = st.sidebar.number_input("Minimum Stock Price (Rp)", value=100, step=10)
min_volume = st.sidebar.number_input("Minimum 20D Avg Volume (Shares)", value=10_000_000, step=1_000_000)

if "Fundamental" not in strategy_mode:
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Technical Parameters")
    vol_spike = st.sidebar.slider("Volume Spike Multiplier", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
else:
    st.sidebar.markdown("---")
    st.sidebar.header("💼 Fundamental Parameters")
    max_pbv = st.sidebar.slider("Max PBV", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    max_pe = st.sidebar.slider("Max PE Ratio", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
    min_marcap_t = st.sidebar.slider("Min Market Cap (Trillion IDR)", min_value=0.1, max_value=50.0, value=1.0, step=0.5)
    min_roe = st.sidebar.slider("Min ROE (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    min_roa = st.sidebar.slider("Min ROA (%)", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
    require_cash_gt_price = st.sidebar.checkbox("Cash per Share > Stock Price", value=False)
    require_earnings_growth = st.sidebar.checkbox("3-Year Earnings Growing (or Low 3Y Avg PE)", value=True)

# ==========================================
# 2. DATA ACQUISITION & PROCESSING LOGIC
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_process_market_data():
    url = "https://scanner.tradingview.com/indonesia/scan"
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "options": {"active_symbols_only": True},
        "columns": ["name", "sector"]
    }
    
    stock_list = []
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        for item in data.get('data', []):
            ticker = item['d'][0]
            sector = item['d'][1] if item['d'][1] else "Unknown"
            if len(ticker) == 4 and ticker.isalpha():
                stock_list.append({"symbol": f"{ticker}.JK", "sector": sector})
    except Exception as e:
        st.error(f"TradingView fetch failed: {e}")
        return pd.DataFrame()

    if not stock_list:
        return pd.DataFrame()

    def process_ticker(stock_dict):
        ticker_symbol = stock_dict["symbol"]
        sector = stock_dict["sector"]
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="6mo")
            if hist.empty or len(hist) < 60:
                return None
                
            current_price = hist['Close'].iloc[-1]
            if current_price < 50: return None

            ema_20 = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            hist_60 = hist.tail(60)
            period_high = hist_60['High'].max()
            period_low = hist_60['Low'].min()
            fib_500 = period_high - (0.500 * (period_high - period_low))
            
            avg_vol_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_vol = hist['Volume'].iloc[-1]
            
            today_high = hist['High'].iloc[-1]
            today_low = hist['Low'].iloc[-1]
            prev_high = hist['High'].iloc[-2]
            prev_low = hist['Low'].iloc[-2]
            is_inside_bar = (today_high <= prev_high) and (today_low >= prev_low)

            open_p = hist['Open'].iloc[-1]
            close_p = hist['Close'].iloc[-1]
            total_range = today_high - today_low
            is_pin_bar = False
            if total_range > 0:
                body = abs(close_p - open_p)
                lower_wick = min(open_p, close_p) - today_low
                upper_wick = today_high - max(open_p, close_p)
                is_pin_bar = (lower_wick >= 0.60 * total_range) and (upper_wick <= 0.20 * total_range) and (body <= 0.35 * total_range)

            info = ticker.info
            pbv = info.get('priceToBook', 999)
            pe = info.get('trailingPE', 999)
            if pe is None: pe = 999
            market_cap_trillion = (info.get('marketCap', 0) / 1_000_000_000_000)
            roe_pct = (info.get('returnOnEquity', 0) or 0) * 100
            roa_pct = (info.get('returnOnAssets', 0) or 0) * 100
            cash_per_share = (info.get('totalCash', 0) / info.get('sharesOutstanding', 1)) if info.get('totalCash') and info.get('sharesOutstanding') else 0

            return {
                'Ticker': ticker_symbol.replace('.JK', ''),
                'Sector': sector,
                'Price': int(current_price),
                'PBV': round(pbv, 2),
                'PE': round(pe, 2),
                'MarketCap_T': round(market_cap_trillion, 2),
                'ROE_Pct': round(roe_pct, 2),
                'ROA_Pct': round(roa_pct, 2),
                'Cash_Per_Share': round(cash_per_share, 2),
                'EMA_20': round(ema_20, 2),
                'Fib_500': round(fib_500, 2),
                'Avg_Volume_20': int(avg_vol_20),
                'Current_Volume': int(current_vol),
                'Is_Inside_Bar': is_inside_bar,
                'Is_Pin_Bar': is_pin_bar
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(process_ticker, s): s for s in stock_list}
        for future in as_completed(future_to_ticker):
            data = future.result()
            if data is not None:
                results.append(data)

    return pd.DataFrame(results)

# ==========================================
# 3. MAIN DASHBOARD EXECUTION
# ==========================================
if st.button("Run Full Market Scan", type="primary"):
    with st.spinner("Scanning IDX stocks..."):
        start_time = time.time()
        raw_df = fetch_and_process_market_data()
        elapsed = round(time.time() - start_time, 1)
        
    if raw_df.empty:
        st.error("Failed to load market data.")
    else:
        st.success(f"Successfully processed {len(raw_df)} active stocks in {elapsed} seconds.")
        st.session_state['raw_data'] = raw_df

if 'raw_data' in st.session_state:
    df = st.session_state['raw_data']
    df['Volume_Multiplier'] = (df['Current_Volume'] / df['Avg_Volume_20']).round(2)
    
    st.sidebar.markdown("---")
    available_sectors = sorted(df['Sector'].unique().tolist())
    selected_sectors = st.sidebar.multiselect("Filter by Sector", options=available_sectors, default=available_sectors)
    
    if "Active Breakout" in strategy_mode:
        filtered_picks = df[(df['Sector'].isin(selected_sectors)) & (df['Price'] >= min_price) & (df['Avg_Volume_20'] >= min_volume) & (df['Price'] > df['EMA_20']) & (df['Current_Volume'] >= (vol_spike * df['Avg_Volume_20']))].sort_values(by='Volume_Multiplier', ascending=False)
    elif "Pre-Breakout Coiling" in strategy_mode:
        filtered_picks = df[(df['Sector'].isin(selected_sectors)) & (df['Price'] >= min_price) & (df['Avg_Volume_20'] >= min_volume) & (df['Price'] > df['EMA_20']) & (df['Is_Inside_Bar'] == True)].sort_values(by='Avg_Volume_20', ascending=False)
    elif "Support Bounce" in strategy_mode:
        filtered_picks = df[(df['Sector'].isin(selected_sectors)) & (df['Price'] >= min_price) & (df['Avg_Volume_20'] >= min_volume) & (df['Is_Pin_Bar'] == True) & ((df['Price'] >= df['EMA_20']) | (df['Price'] >= df['Fib_500']))].sort_values(by='Avg_Volume_20', ascending=False)
    else:
        filtered_picks = df[(df['Sector'].isin(selected_sectors)) & (df['Price'] >= min_price) & (df['Avg_Volume_20'] >= min_volume) & (df['PBV'] <= max_pbv) & (df['PE'] <= max_pe) & (df['MarketCap_T'] >= min_marcap_t) & (df['ROE_Pct'] >= min_roe) & (df['ROA_Pct'] >= min_roa)].sort_values(by='ROE_Pct', ascending=False)
    
    st.subheader(f"📊 Found {len(filtered_picks)} Candidates ({strategy_mode})")
    
    if filtered_picks.empty:
        st.info("No stocks match your criteria. Try adjusting the parameters.")
    else:
        # 1. DISPLAY LOGIC FOR FUNDAMENTAL MODE
        if "Fundamental" in strategy_mode:
            display_df = filtered_picks[['Ticker', 'Sector', 'Price', 'PBV', 'PE', 'MarketCap_T', 'ROE_Pct', 'ROA_Pct', 'Cash_Per_Share']].copy()
            st.dataframe(
                display_df.style.format({
                    'Price': "Rp {:,.0f}",
                    'PBV': "{:.2f}",
                    'PE': "{:.2f}",
                    'MarketCap_T': "Rp {:.2f}T",
                    'ROE_Pct': "{:.1f}%",
                    'ROA_Pct': "{:.1f}%",
                    'Cash_Per_Share': "Rp {:,.0f}"
                }),
                use_container_width=True, hide_index=True
            )
            
        # 2. DISPLAY LOGIC FOR TECHNICAL MODES
        else:
            display_df = filtered_picks[['Ticker', 'Sector', 'Price', 'Volume_Multiplier', 'Current_Volume', 'Avg_Volume_20', 'EMA_20', 'Fib_500']].copy()
            st.dataframe(
                display_df.style.format({
                    'Price': "Rp {:,.0f}",
                    'Volume_Multiplier': "{:.2f}x",
                    'Current_Volume': "{:,.0f}",
                    'Avg_Volume_20': "{:,.0f}",
                    'EMA_20': "Rp {:,.2f}",
                    'Fib_500': "Rp {:,.2f}"
                }),
                use_container_width=True, hide_index=True
            )

        # ==========================================
        # 4. INTERACTIVE CHART & AI ANALYST
        # ==========================================
        st.markdown("---")
        st.subheader("📈 Interactive Chart & AI Analyst Agent")
        
        # THIS IS THE VARIABLE THAT WAS MISSING!
        selected_ticker = st.selectbox("Select a candidate to inspect:", filtered_picks['Ticker'].tolist())
        
        if selected_ticker:
            stock_row = filtered_picks[filtered_picks['Ticker'] == selected_ticker].iloc[0]
            
            with st.spinner(f"Loading chart for {selected_ticker}..."):
                ticker_obj = yf.Ticker(f"{selected_ticker}.JK")
                hist = ticker_obj.history(period="6mo")
                
                if not hist.empty:
                    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
                    
                    p_high, p_low = hist['High'].max(), hist['Low'].min()
                    fib_500 = p_high - (0.500 * (p_high - p_low))
                    fig.add_hline(y=fib_500, line_dash="dash", line_color="green", annotation_text="50% Fib Support")
                    fig.update_layout(title=f"{selected_ticker} Chart Setup", yaxis_title='IDR', xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            # --- GEMINI AI AGENT SECTION ---
            st.markdown("### 🤖 Gemini AI Analyst Review")
            if st.button(f"Ask AI Agent to Analyze {selected_ticker}"):
                if not gemini_api_key:
                    st.warning("Please enter your Gemini API Key in the sidebar first!")
                else:
                    with st.spinner("Gemini AI is examining price action, indicators, and valuation..."):
                        try:
                            client = genai.Client(api_key=gemini_api_key)
                            
                            prompt = f"""
                            You are an expert professional stock market analyst specializing in the Indonesian Stock Exchange (IHSG).
                            Analyze the following stock data for {selected_ticker} and give a structured professional opinion:
                            
                            - Current Price: Rp {stock_row['Price']}
                            - Sector: {stock_row['Sector']}
                            - 20-Day EMA: Rp {stock_row.get('EMA_20', 'N/A')}
                            - 50% Fibonacci Support Level: Rp {stock_row.get('Fib_500', 'N/A')}
                            - Inside Bar Pattern Detected: {stock_row.get('Is_Inside_Bar', False)}
                            - Bullish Pin Bar Pattern Detected: {stock_row.get('Is_Pin_Bar', False)}
                            - Volume Spike Multiplier: {stock_row.get('Volume_Multiplier', 'N/A')}x
                            - Valuation (PBV): {stock_row.get('PBV', 'N/A')}
                            - Valuation (PE): {stock_row.get('PE', 'N/A')}
                            - Profitability (ROE): {stock_row.get('ROE_Pct', 'N/A')}%
                            
                            Provide your response in clear sections using Markdown:
                            1. **Technical & Chart Context**: Evaluate where the price is relative to its EMA 20 and Fibonacci support.
                            2. **Pattern Significance**: What do the candle structures or volume imply about current market psychology for this stock?
                            3. **Fundamental Health**: Brief note on its valuation and ROE.
                            4. **Actionable Verdict**: Bullish, Neutral, or Bearish outlook with a suggested risk-management idea (e.g., where to place a stop-loss).
                            """
                            
                            response = client.models.generate_content(
                                model='gemini-3.8-flash',
                                contents=prompt
                            )
                            
                            st.success("Analysis Complete!")
                            st.markdown(response.text)
                            
                        except Exception as e:
                            st.error(f"Failed to generate AI analysis: {e}")