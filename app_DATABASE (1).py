import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

st.set_page_config(page_title='玻璃開發逆向優化系統', layout='wide')

# --- 1. 核心邏輯：載入與訓練 ---
@st.cache_resource
def load_and_train(file_obj):
    feature_cols = ['compounds_SiO2', 'compounds_Al2O3', 'compounds_B2O3', 'compounds_MgO', 'compounds_CaO', 'compounds_Na2O', 'compounds_K2O', 'compounds_F', 'compounds_TiO2', 'compounds_Fe2O3']
    target_map = {'Dk': 'property_Permittivity', 'Df': 'property_TangentOfLossAngle', 'E': 'property_YoungModulus', 'CTE': 'property_CTEbelowTg', 'log3_viscosity': 'property_T3'}

    df_raw = pd.read_csv(file_obj, usecols=feature_cols + list(target_map.values()), encoding='latin1', low_memory=False)
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
            model = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('model', ExtraTreesRegressor(n_estimators=300, random_state=42))])
            model.fit(X, y)
            models[name] = model
    return models, df_low, feature_cols

# --- 2. UI 介面 ---
st.sidebar.title('📁 數據中心')
uploaded_file = st.sidebar.file_uploader('請上傳玻璃數據庫 (CSV)', type='csv')

if uploaded_file:
    trained_models, df_ref, feature_cols = load_and_train(uploaded_file)
    st.title('LCTE玻璃性質預測與開發系統')

    tab1, tab2 = st.tabs(['🎯 正向物性預測', '⚙️ 逆向配方優化'])

    with tab1:
        st.header('1. 設定化學成分 (wt%)')
        feat_keys = [k.replace('compounds_', '') for k in feature_cols]
        user_input = {}
        cols = st.columns(5)
        for i, k in enumerate(feat_keys):
            with cols[i % 5]:
                user_input[k] = st.number_input(k, value=0.0, step=0.1, format='%.2f', key=f'fwd_{k}')

        if st.button('開始預測', use_container_width=True):
            total = sum(user_input.values())
            if total > 0:
                X_new = pd.DataFrame([{f'compounds_{k}': v/total for k, v in user_input.items()}])[feature_cols]
                st.subheader('預測結果')
                # 過濾掉不想顯示的 Dk 和 Df
                display_props = {k: v for k, v in trained_models.items() if k not in ['Dk', 'Df']}
                res_cols = st.columns(len(display_props))

                for i, (prop, model) in enumerate(display_props.items()):
                    val = model.predict(X_new)[0]
                    # 邏輯變更：黏度數值扣除 150
                    if prop == 'log3_viscosity':
                        display_val = val - 150
                        unit = ' °C'
                    else:
                        display_val = val
                        unit = ' GPa' if prop == 'E' else ' ppm/K' if prop == 'CTE' else ''

                    res_cols[i].metric(prop, f'{display_val:.4f}{unit}')
            else:
                st.warning('請輸入成分含量')

    with tab2:
        st.header('2. 設定目標物性')
        c1, c2 = st.columns(2)
        target_cte = c1.number_input('目標 CTE (ppm/K)', value=3.0, min_value=1.0, max_value=4.0)
        target_e = c2.number_input('目標 楊氏模數 E (GPa)', value=85.0, min_value=50.0)

        if st.button('啟動配方搜尋 (高效模式)', use_container_width=True):
            with st.spinner('正在進行模擬尋找最佳配方，請稍候...'):
                bounds = [(df_ref[f'compounds_{k}'].min()*100, df_ref[f'compounds_{k}'].max()*100) for k in feat_keys]

                def objective(weights):
                    total = sum(weights)
                    if total == 0: return 1e10
                    X = pd.DataFrame([{f'compounds_{k}': v/total for k, v in zip(feat_keys, weights)}])[feature_cols]
                    p_cte = trained_models['CTE'].predict(X)[0]
                    p_e = trained_models['E'].predict(X)[0]
                    return ((p_cte - target_cte)/target_cte)**2 + ((p_e - target_e)/target_e)**2

                res = differential_evolution(objective, bounds, tol=0.05, popsize=5, maxiter=500, seed=42)

                if res.success:
                    st.success('✅ 已找到建議配方！')
                    final_comp = {k: v/sum(res.x)*100 for k, v in zip(feat_keys, res.x)}

                    col_res1, col_res2 = st.columns([1, 1])
                    with col_res1:
                        st.write('### 建議成分配比 (wt%)')
                        df_display = pd.DataFrame([final_comp]).T.rename(columns={0: '含量 (wt%)'})
                        st.dataframe(df_display.style.format('{:.2f}'))
                    with col_res2:
                        st.write('### 預期達成性質')
                        X_final = pd.DataFrame([{f'compounds_{k}': v/100 for k, v in final_comp.items()}])[feature_cols]
                        st.metric('預測 CTE', f"{trained_models['CTE'].predict(X_final)[0]:.4f} ppm/K")
                        st.metric('預測 E', f"{trained_models['E'].predict(X_final)[0]:.4f} GPa")
                else:
                    st.error('優化未能在限定時間內收斂，請嘗試調整目標值。')

else:
    st.info('請先在左側上傳 CSV 資料庫以啟用網頁功能。')
