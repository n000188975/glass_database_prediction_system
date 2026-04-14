import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

st.set_page_config(page_title='玻璃性質預測系統 (平行運算優化版)', layout='wide')

@st.cache_resource
def load_and_train(file_source):
    feature_cols = ['compounds_SiO2', 'compounds_Al2O3', 'compounds_B2O3', 'compounds_MgO', 'compounds_CaO', 'compounds_Na2O', 'compounds_K2O', 'compounds_F', 'compounds_TiO2', 'compounds_Fe2O3']
    target_map = {'Dk': 'property_Permittivity', 'Df': 'property_TangentOfLossAngle', 'E': 'property_YoungModulus', 'CTE': 'property_CTEbelowTg', 'log3_viscosity': 'property_T3'}
    
    if file_source.endswith('.parquet'):
        df_raw = pd.read_parquet(file_source)
    else:
        df_raw = pd.read_csv(file_source, usecols=feature_cols + list(target_map.values()), encoding='latin1', low_memory=False)
        
    for c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
    
    t_cte = 'property_CTEbelowTg'
    mask_small = df_raw[t_cte] < 0.1
    df_raw.loc[mask_small, t_cte] *= 1e6
    df_low = df_raw[df_raw[t_cte] < 3.5].dropna(subset=[t_cte])
    
    models = {}
    for name, col in target_map.items():
        data = df_low[feature_cols + [col]].dropna(subset=[col])
        if len(data) >= 50:
            X, y = data[feature_cols], data[col]
            # 啟用 n_jobs=-1 以使用所有核心進行平行運算
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                ('model', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            model.fit(X, y)
            models[name] = model
    return models, df_low, feature_cols

# --- 偵測資料來源 ---
DB_FILES = ['database.parquet', 'database.csv']
data_source = None
for f in DB_FILES:
    if os.path.exists(f):
        data_source = f
        break

st.sidebar.title('📁 數據中心')
if data_source:
    st.sidebar.success(f'✅ 已偵測到內建資料庫: {data_source}')
else:
    uploaded_file = st.sidebar.file_uploader('請上傳玻璃數據庫', type=['csv', 'parquet'])
    data_source = uploaded_file

if data_source:
    trained_models, df_ref, feature_cols = load_and_train(data_source)
    st.title('🔬 玻璃性質預測與逆向開發系統')
    
    tab1, tab2 = st.tabs(['🎯 正向物性預測', '⚙️ 逆向配方優化'])
    feat_keys = [k.replace('compounds_', '') for k in feature_cols]
    
    with tab1:
        st.header('1. 設定化學成分 (wt%)')
        user_input = {}
        cols = st.columns(5)
        for i, k in enumerate(feat_keys):
            with cols[i % 5]:
                user_input[k] = st.number_input(k, value=0.0, step=0.1, key=f'fwd_{k}')
        
        if st.button('開始預測', use_container_width=True):
            total = sum(user_input.values())
            if total > 0:
                X_new = pd.DataFrame([{f'compounds_{k}': v/total for k, v in user_input.items()}])[feature_cols]
                st.subheader('預測結果')
                res_cols = st.columns(len(trained_models))
                for i, (prop, model) in enumerate(trained_models.items()):
                    val = model.predict(X_new)[0]
                    res_cols[i].metric(prop, f'{val:.4f}')

    with tab2:
        st.header('2. 設定目標物性')
        c1, c2 = st.columns(2)
        t_cte = c1.number_input('目標 CTE (ppm/K)', value=3.0, step=0.1)
        t_e = c2.number_input('目標 楊氏模數 E (GPa)', value=85.0, step=1.0)
        
        if st.button('啟動配方快速搜尋', use_container_width=True):
            if 'CTE' not in trained_models or 'E' not in trained_models:
                st.error('模型未就緒，無法優化')
            else:
                with st.spinner('正在使用多核心進行優化計算...'):
                    bounds = [(df_ref[f'compounds_{k}'].min()*100, df_ref[f'compounds_{k}'].max()*100) for k in feat_keys]
                    
                    def objective(weights):
                        total = sum(weights)
                        if total == 0: return 1e10
                        X = pd.DataFrame([{f'compounds_{k}': v/total for k, v in zip(feat_keys, weights)}])[feature_cols]
                        p_cte = trained_models['CTE'].predict(X)[0]
                        p_e = trained_models['E'].predict(X)[0]
                        return ((p_cte - t_cte)/t_cte)**2 + ((p_e - t_e)/t_e)**2

                    # 優化搜尋設定
                    res = differential_evolution(objective, bounds, tol=0.1, popsize=5, maxiter=50, seed=42)
                    
                    if res.success or res.fun < 1.0:
                        st.success('✅ 已找到建議配方')
                        final_comp = {k: v/sum(res.x)*100 for k, v in zip(feat_keys, res.x)}
                        st.table(pd.DataFrame([final_comp]).T.rename(columns={0: '建議含量 (wt%)'}))
                        
                        X_final = pd.DataFrame([{f'compounds_{k}': v/100 for k, v in final_comp.items()}])[feature_cols]
                        st.metric('預計 CTE', f"{trained_models['CTE'].predict(X_final)[0]:.2f}")
                        st.metric('預計 E', f"{trained_models['E'].predict(X_final)[0]:.2f}")
                    else:
                        st.error('搜尋失敗，請嘗試放寬目標值。')
else:
    st.warning('請上傳或放置資料庫檔案。')
