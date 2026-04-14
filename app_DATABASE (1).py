import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

st.set_page_config(page_title='玻璃性質預測系統 (動態全組分版)', layout='wide')

@st.cache_resource
def load_and_train(file_source):
    # 1. 讀取資料並自動偵測所有組分 (以 compounds_ 開頭)
    if file_source.endswith('.parquet'):
        df_full = pd.read_parquet(file_source)
    else:
        df_full = pd.read_csv(file_source, encoding='latin1', low_memory=False)
    
    all_cols = df_full.columns.tolist()
    # 自動抓取資料庫中所有的組分欄位
    feature_cols = [c for c in all_cols if c.startswith('compounds_')]
    target_map = {
        'Dk': 'property_Permittivity', 
        'Df': 'property_TangentOfLossAngle', 
        'E': 'property_YoungModulus', 
        'CTE': 'property_CTEbelowTg', 
        'log3_viscosity': 'property_T3'
    }
    
    # 僅處理數值化
    usecols = feature_cols + [v for v in target_map.values() if v in all_cols]
    df_raw = df_full[usecols].copy()
    for c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
    
    # CTE 單位修正與資料篩選
    t_cte = 'property_CTEbelowTg'
    if t_cte in df_raw.columns:
        mask_small = df_raw[t_cte] < 0.1
        df_raw.loc[mask_small, t_cte] *= 1e6
        df_low = df_raw[df_raw[t_cte] < 3.5].dropna(subset=[t_cte])
    else:
        df_low = df_raw.copy()

    # 訓練模型
    models = {}
    for name, col in target_map.items():
        if col in df_low.columns:
            data = df_low[feature_cols + [col]].dropna(subset=[col])
            if len(data) >= 50:
                X, y = data[feature_cols], data[col]
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                    ('model', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ])
                model.fit(X, y)
                models[name] = model
    return models, df_low, feature_cols

# --- 檔案偵測與載入 ---
DB_FILES = ['database.parquet', 'database.csv']
data_source = None
for f in DB_FILES:
    if os.path.exists(f):
        data_source = f
        break

st.sidebar.title('📁 數據中心')
if data_source:
    st.sidebar.success(f'✅ 已自動載入內建資料庫')
else:
    uploaded_file = st.sidebar.file_uploader('請上傳玻璃數據庫', type=['csv', 'parquet'])
    data_source = uploaded_file

if data_source:
    trained_models, df_ref, feature_cols = load_and_train(data_source)
    st.title('🔬 玻璃性質預測與開發系統')
    st.markdown(f'**目前資料庫包含 {len(feature_cols)} 種化學成分，已全部開放輸入。**')

    tab1, tab2 = st.tabs(['🎯 正向物性預測', '⚙️ 逆向配方優化'])
    display_labels = [k.replace('compounds_', '') for k in feature_cols]
    
    with tab1:
        st.header('1. 設定化學成分 (wt%)')
        st.info('請在下方輸入各組分含量。系統將自動進行歸一化處理。')
        user_input = {}
        # 動態建立所有組分的輸入框，每列顯示 6 個
        cols = st.columns(6)
        for i, (col_name, label) in enumerate(zip(feature_cols, display_labels)):
            with cols[i % 6]:
                user_input[col_name] = st.number_input(label, value=0.0, step=0.1, key=f'fwd_{label}')
        
        if st.button('開始預測', use_container_width=True):
            total = sum(user_input.values())
            if total > 0:
                # 歸一化模型輸入
                X_new = pd.DataFrame([{k: v/total for k, v in user_input.items()}])[feature_cols]
                st.subheader('預測結果摘要')
                res_cols = st.columns(len(trained_models))
                for i, (prop, model) in enumerate(trained_models.items()):
                    val = model.predict(X_new)[0]
                    res_cols[i].metric(prop, f'{val:.4f}')
            else:
                st.warning('請至少輸入一些成分含量')

    with tab2:
        st.header('2. 設定目標物性')
        c1, c2 = st.columns(2)
        t_cte = c1.number_input('目標 CTE (ppm/K)', value=3.0, step=0.1)
        t_e = c2.number_input('目標 楊氏模數 E (GPa)', value=85.0, step=1.0)
        
        if st.button('啟動配方快速搜尋', use_container_width=True):
            if 'CTE' not in trained_models or 'E' not in trained_models:
                st.error('必要的預測模型未就緒')
            else:
                with st.spinner('正在分析最佳配方組合...'):
                    # 搜尋範圍定義
                    bounds = [(df_ref[c].min()*100, df_ref[c].max()*100) for c in feature_cols]
                    
                    def objective(weights):
                        total = sum(weights)
                        if total == 0: return 1e10
                        X = pd.DataFrame([weights/total], columns=feature_cols)
                        p_cte = trained_models['CTE'].predict(X)[0]
                        p_e = trained_models['E'].predict(X)[0]
                        return ((p_cte - t_cte)/t_cte)**2 + ((p_e - t_e)/t_e)**2

                    res = differential_evolution(objective, bounds, tol=0.1, popsize=5, maxiter=50, seed=42)
                    
                    if res.success or res.fun < 1.0:
                        st.success('✅ 已找到建議配方！')
                        final_comp_vals = res.x / sum(res.x) * 100
                        final_comp = dict(zip(display_labels, final_comp_vals))
                        
                        st.write('### 建議成分配比 (wt%)')
                        df_res = pd.DataFrame([final_comp]).T.rename(columns={0: '含量 (wt%)'})
                        # 僅顯示含量顯著的組分 (>0.01%)
                        st.dataframe(df_res[df_res['含量 (wt%)'] > 0.01].style.format('{:.2f}'), use_container_width=True)
                        
                        st.write('### 預期達成性質')
                        X_final = pd.DataFrame([res.x / sum(res.x)], columns=feature_cols)
                        m1, m2 = st.columns(2)
                        m1.metric('預計 CTE', f"{trained_models['CTE'].predict(X_final)[0]:.2f} ppm/K")
                        m2.metric('預計 E', f"{trained_models['E'].predict(X_final)[0]:.2f} GPa")
                    else:
                        st.error('搜尋失敗，請嘗試放寬目標範圍。')
else:
    st.warning('請上傳 CSV/Parquet 資料庫檔案。')
