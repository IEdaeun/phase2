# -*- coding: utf-8 -*-
from __future__ import annotations
import math, json, sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# --- 2단계 계획용 상수 ---
DATA_DIR = Path("data")
# --- 💡 주요 변경점 START ---
# '안전 재고 일수'. 평균 30일치 수요를 안전 재고로 확보하도록 설정.
# 이 값을 늘리면 Fill-Rate가 상승하지만, 재고 비용이 증가합니다.
SAFETY_DAYS = 30
# --- 💡 주요 변경점 END ---


# --- 유틸 함수 ---
def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l1 - l2)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def norm_city(name): return str(name).replace(' ', '_')
def ceil_pos(x): return 0 if x<=0 else int(math.ceil(x))
def truck_days_from_distance(dkm):
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8

def main_stage2_planner():
    print("--- 2단계 전술 계획 수립 시작 (재고 커버리지 강화 버전) ---")

    # =================== LOAD DATA ========================
    sites     = pd.read_csv(f'{DATA_DIR}/site_candidates.csv'); sites['city']=sites['city'].map(norm_city)
    mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv')
    
    try:
        con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
        d_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con, parse_dates=['date'])
        con.close()
        d_hist['city']=d_hist['city'].map(norm_city)
    except Exception:
        d_hist = pd.DataFrame()
    
    d_fut = pd.read_csv(f'{DATA_DIR}/forecast_submission_template.csv', parse_dates=['date'])
    d_fut['city']=d_fut['city'].map(norm_city)
    d_fut.rename(columns={'mean':'demand'}, inplace=True)
    all_dem = pd.concat([d_hist, d_fut], ignore_index=True).drop_duplicates(subset=['date','sku','city'], keep='last')

    fac_df = sites[sites['site_type']=='factory'].copy()
    wh_df  = sites[sites['site_type']=='warehouse'].copy()
    I = fac_df['site_id'].tolist()
    K = wh_df['site_id'].tolist()
    
    site_country = dict(zip(sites['site_id'], sites['country']))
    site_lat     = dict(zip(sites['site_id'], sites['lat']))
    site_lon     = dict(zip(sites['site_id'], sites['lon']))
    
    modes = mode_meta['mode'].tolist()
    alpha_lead = dict(zip(mode_meta['mode'], mode_meta['leadtime_factor']))
    dist_ik = {(i,k): haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k]) for i in I for k in K}
    L_ik = {m:{} for m in modes}
    for (i,k), d in dist_ik.items():
        base_days = truck_days_from_distance(d)
        for m in modes: L_ik[m][(i,k)] = ceil_pos(alpha_lead[m] * base_days)
    
    try:
        selected_factories = pd.read_csv("selected_factories.csv")['factory'].tolist()
        selected_warehouses = pd.read_csv("selected_warehouses.csv")['warehouse'].tolist()
    except FileNotFoundError:
        print("오류: 1단계 결과 파일('selected_factories.csv', 'selected_warehouses.csv')이 필요합니다.")
        return

    # =================== 소싱 및 재고 정책 수립 ========================
    wh_sourcing_options = defaultdict(list)
    for wh_id in selected_warehouses:
        options = []
        for fac_id in selected_factories:
            is_international = (site_country[wh_id] != site_country[fac_id])
            mode = 'SHIP' if is_international else 'TRUCK'
            lead_time = L_ik[mode][(fac_id, wh_id)]
            cost = dist_ik[(fac_id, wh_id)]
            options.append({'factory': fac_id, 'cost': cost, 'lead_time': int(lead_time)})
        wh_sourcing_options[wh_id] = sorted(options, key=lambda x: x['cost'])

    final_sourcing_policy = {}
    for wh_id in selected_warehouses:
        if not wh_sourcing_options[wh_id]: continue
        primary_option = wh_sourcing_options[wh_id][0]
        secondary_option = wh_sourcing_options[wh_id][1] if len(wh_sourcing_options[wh_id]) > 1 else None
        final_sourcing_policy[wh_id] = {"primary_source": primary_option, "secondary_source": secondary_option}
        
    inventory_policy = defaultdict(dict)
    city_coords = sites.groupby('city')[['lat', 'lon']].mean()
    unique_cities_in_demand = all_dem['city'].unique()
    city_to_wh_map = {city_name: min(selected_warehouses, key=lambda wh_id: haversine_km(city_coords.loc[city_name,'lat'], city_coords.loc[city_name,'lon'], site_lat[wh_id], site_lon[wh_id])) for city_name in unique_cities_in_demand if city_name in city_coords.index}
    all_dem['warehouse'] = all_dem['city'].map(city_to_wh_map)
    demand_stats = all_dem.groupby(['warehouse', 'sku'])['demand'].agg(['mean', 'std']).fillna(0)
    
    for (wh_id, sku_id), stats in demand_stats.iterrows():
        if wh_id not in final_sourcing_policy: continue
        
        lead_time = final_sourcing_policy[wh_id]['primary_source']['lead_time']
        
        # --- 💡 주요 변경점 START ---
        # 안전 재고를 '평균 N일치 수요'로 설정
        safety_stock = stats['mean'] * SAFETY_DAYS
        reorder_point = (lead_time * stats['mean']) + safety_stock
        # --- 💡 주요 변경점 END ---
        
        inventory_policy[wh_id][sku_id] = {
            "safety_stock": int(np.ceil(safety_stock)),
            "reorder_point": int(np.ceil(reorder_point))
        }

    final_plan = {"sourcing_policy": final_sourcing_policy, "inventory_policy": inventory_policy}
    output_path = "tactical_plan_advanced.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_plan, f, indent=4)
        
    print(f"\n--- 2단계 전술 계획 수립 완료 ---")
    print(f"안전 재고 일수: {SAFETY_DAYS}일")
    print(f"결과가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main_stage2_planner()
