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
# SKU 등급(Class)별 차등 안전 재고 일수 (단위: 일)
# AX: 잘 팔리고 안정적 -> 재고 최소화
# CZ: 안 팔리고 불안정 -> 재고 최대화 (품절 방지)
SAFETY_DAYS_MATRIX = {
    # A Class (가장 중요): 50일을 기준으로 보수적으로 운영
    'AX': 40, 'AY': 50, 'AZ': 65,
    # B Class (중간): 50일을 기준으로 운영
    'BX': 35, 'BY': 45, 'BZ': 55,
    # C Class (비용 최적화 대상): 재고 최소화
    'CX': 30, 'CY': 40, 'CZ': 50,
    'DEFAULT': 50 # 기준값
}
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
    print("--- 2단계 전술 계획 수립 시작 (차등 재고 정책 버전) ---")

    # =================== LOAD DATA ========================
    sites     = pd.read_csv(f'{DATA_DIR}/site_candidates.csv'); sites['city']=sites['city'].map(norm_city)
    mode_meta = pd.read_csv(f'{DATA_DIR}/transport_mode_meta.csv')
    
    try:
        con = sqlite3.connect(f'{DATA_DIR}/demand_train.db')
        d_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con, parse_dates=['date'])
        con.close()
    except Exception:
        d_hist = pd.DataFrame()
    
    d_fut = pd.read_csv(f'{DATA_DIR}/forecast_submission_template.csv', parse_dates=['date'])
    d_fut.rename(columns={'mean':'demand'}, inplace=True)
    all_dem = pd.concat([d_hist, d_fut], ignore_index=True).drop_duplicates(subset=['date','sku','city'], keep='last')
    all_dem['city'] = all_dem['city'].map(norm_city)

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
    
    # --- 💡 주요 변경점 START ---
    # SKU별 수요 통계 계산
    sku_demand_stats = all_dem.groupby('sku')['demand'].agg(['sum', 'mean', 'std']).fillna(0)
    
    # ABC 분석 (수요량 기준)
    sku_demand_stats['cum_sum'] = sku_demand_stats['sum'].sort_values(ascending=False).cumsum()
    total_sum = sku_demand_stats['sum'].sum()
    sku_demand_stats['cum_perc'] = sku_demand_stats['cum_sum'] / total_sum
    
    def abc_classify(perc):
        if perc <= 0.7: return 'A'
        if perc <= 0.9: return 'B'
        return 'C'
    sku_demand_stats['abc_class'] = sku_demand_stats['cum_perc'].apply(abc_classify)

    # XYZ 분석 (수요 변동성 기준 - CV)
    sku_demand_stats['cv'] = (sku_demand_stats['std'] / sku_demand_stats['mean']).fillna(0)
    
    def xyz_classify(cv):
        if cv <= 0.5: return 'X' # 안정
        if cv <= 1.0: return 'Y' # 보통
        return 'Z' # 불안정
    sku_demand_stats['xyz_class'] = sku_demand_stats['cv'].apply(xyz_classify)

    # 최종 등급 매핑
    sku_demand_stats['class'] = sku_demand_stats['abc_class'] + sku_demand_stats['xyz_class']
    sku_class_map = sku_demand_stats['class'].to_dict()
    
    # 창고-SKU별 재고 정책 수립
    inventory_policy = defaultdict(dict)
    city_coords = sites.groupby('city')[['lat', 'lon']].mean()
    unique_cities_in_demand = all_dem['city'].unique()
    city_to_wh_map = {city_name: min(selected_warehouses, key=lambda wh_id: haversine_km(city_coords.loc[city_name,'lat'], city_coords.loc[city_name,'lon'], site_lat[wh_id], site_lon[wh_id])) for city_name in unique_cities_in_demand if city_name in city_coords.index}
    all_dem['warehouse'] = all_dem['city'].map(city_to_wh_map)
    demand_stats_wh = all_dem.groupby(['warehouse', 'sku'])['demand'].agg(['mean']).fillna(0)

    for (wh_id, sku_id), stats in demand_stats_wh.iterrows():
        if wh_id not in final_sourcing_policy: continue
        
        lead_time = final_sourcing_policy[wh_id]['primary_source']['lead_time']
        
        # SKU 등급에 맞는 안전재고일수 가져오기
        sku_class = sku_class_map.get(sku_id, 'DEFAULT')
        safety_days = SAFETY_DAYS_MATRIX.get(sku_class, SAFETY_DAYS_MATRIX['DEFAULT'])
        
        # 새로운 재고 정책 계산
        safety_stock = stats['mean'] * safety_days
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
    print(f"SKU별 차등 재고 정책 적용 완료.")
    print(f"결과가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main_stage2_planner()
