import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

st.set_page_config(page_title='玻璃性質預測與優化系統 (進階選單版)', layout='wide')

@st.cache_resource
def load_and_train(file_source):
    if file_source.endswith('.parquet'):
        df_full = pd.read_parquet(file_source)
    else:
        df_full = pd.read_csv(file_source, encoding='latin1', low_memory=False)
    
    all_cols = df_full.columns.tolist()
    feature_cols = [c for c in all_cols if c.startswith('compounds_')]
    target_map = {'Dk': 'property_Permittivity', 'Df': 'property_TangentOfLossAngle', 'E': 'property_YoungModulus', 'CTE': 'property_CTEbelowTg', 'log3_viscosity': 'property_T3'}
    
    usecols = feature_cols + [v for v in target_map.values() if v in all_cols]
    df_raw = df_full[usecols].copy()
    for c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
    
    t_cte = 'property_CTEbelowTg'
    if t_cte in df_raw.columns:
        mask_small = df_raw[t_cte] < 0.1
        df_raw.loc[mask_small, t_cte] *= 1e6
        df_low = df_raw[df_raw[t_cte] < 3.5].dropna(subset=[t_cte])
    else:
        df_low = df_raw.copy()

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

    tab1, tab2 = st.tabs(['🎯 正向物性預測', '⚙️ 逆向配方優化'])
    
    # 準備排序過的組分清單
    display_labels_map = {k.replace('compounds_', ''): k for k in feature_cols}
    sorted_component_names = sorted(list(display_labels_map.keys()))
    
    with tab1:
        st.header('1. 正向組分設定')
        selected_labels_fwd = st.multiselect('請選擇要輸入的組分 (已按 A-Z 排序):', options=sorted_component_names, default=[n for n in ['SiO2', 'Al2O3', 'B2O3', 'MgO', 'CaO'] if n in sorted_component_names], key='fwd_select')
        
        user_input_vals = {}
        if selected_labels_fwd:
            st.write('---')
            cols = st.columns(4)
            for i, label in enumerate(selected_labels_fwd):
                with cols[i % 4]:
                    user_input_vals[display_labels_map[label]] = st.number_input(f'{label} (wt%)', value=0.0, step=0.1, format='%.2f', key=f'input_{label}')
        
        if st.button('開始預測', use_container_width=True):
            final_input = {c: 0.0 for c in feature_cols}
            for k, v in user_input_vals.items(): final_input[k] = v
            total = sum(final_input.values())
            if total > 0:
                X_new = pd.DataFrame([{k: v/total for k, v in final_input.items()}])[feature_cols]
                st.subheader('預測結果摘要')
                res_cols = st.columns(len(trained_models))
                for i, (prop, model) in enumerate(trained_models.items()):
                    val = model.predict(X_new)[0]
                    res_cols[i].metric(prop, f'{val:.4f}')
            else:
                st.warning('請輸入成分含量')

    with tab2:
        st.header('2. 逆向目標優化')
        # 逆向優化也加入組分選擇
        st.info('系統將只在您選擇的組分範圍內搜尋最佳配方。')
        selected_labels_inv = st.multiselect('請選擇允許搜尋的組分範圍:', options=sorted_component_names, default=[n for n in ['SiO2', 'Al2O3', 'B2O3', 'MgO', 'CaO'] if n in sorted_component_names], key='inv_select')
        
        c1, c2 = st.columns(2)
        t_cte = c1.number_input('目標 CTE (ppm/K)', value=3.0, step=0.1)
        t_e = c2.number_input('目標 楊氏模數 E (GPa)', value=85.0, step=1.0)
        
        if st.button('啟動配方搜尋', use_container_width=True):
            if not selected_labels_inv:
                st.error('請至少選擇一個搜尋組分')
            elif 'CTE' not in trained_models or 'E' not in trained_models:
                st.error('必要的模型未就緒')
            else:
                with st.spinner('正在分析最佳配方組合...'):
                    # 定義邊界：只針對選中的組分，其餘強制為 0
                    active_features = [display_labels_map[l] for l in selected_labels_inv]
                    bounds = []
                    for c in feature_cols:
                        if c in active_features:
                            bounds.append((df_ref[c].min()*100, df_ref[c].max()*100))
                        else:
                            bounds.append((0.0, 0.0))

                    def objective(weights):
                        total = sum(weights)
                        if total == 0: return 1e10
                        X = pd.DataFrame([weights/total], columns=feature_cols)
                        p_cte = trained_models['CTE'].predict(X)[0]
                        p_e = trained_models['E'].predict(X)[0]
                        return ((p_cte - t_cte)/t_cte)**2 + ((p_e - t_e)/t_e)**2

                    res = differential_evolution(objective, bounds, tol=0.1, popsize=5, maxiter=50, seed=42)
                    if res.success:
                        st.success('✅ 已找到建議配方！')
                        final_comp_vals = res.x / sum(res.x) * 100
                        st.write('### 建議成分配比 (wt%)')
                        df_res = pd.DataFrame({'組分': [n.replace('compounds_', '') for n in feature_cols], '含量 (wt%)': final_comp_vals})
                        df_res = df_res[df_res['含量 (wt%)'] > 0.01].sort_values('組分')
                        st.dataframe(df_res.style.format({'含量 (wt%)': '{:.2f}'}), use_container_width=True)
                        
                        X_final = pd.DataFrame([res.x / sum(res.x)], columns=feature_cols)
                        m1, m2 = st.columns(2)
                        m1.metric('預計 CTE', f"{trained_models['CTE'].predict(X_final)[0]:.2f} ppm/K")
                        m2.metric('預計 E', f"{trained_models['E'].predict(X_final)[0]:.2f} GPa")
else:
    st.warning('請上傳 CSV/Parquet 資料庫檔案。')
