# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sqlite3, math, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# ===================== CONFIG =====================
DATA_DIR   = Path("data")
DATE_START = datetime(2018,1,1)
DATE_END   = datetime(2024,12,31)

MAX_FAC = 5
MAX_WH  = 20

TRUCK_BASE_COST_PER_KM = 12.0
BORDER_COST = 4000.0
MAX_CROSS_BORDER_KM = 1000

# ==================================================

def norm_city(x: str) -> str:
    return str(x).replace(' ', '_')

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1)/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin((l1 - l2)/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def main():
    # ---------- load
    sites     = pd.read_csv(DATA_DIR/'site_candidates.csv')
    sites['city'] = sites['city'].map(norm_city)
    init_cost = pd.read_csv(DATA_DIR/'site_init_cost.csv')
    mode_meta = pd.read_csv(DATA_DIR/'transport_mode_meta.csv')
    fac_cap   = pd.read_csv(DATA_DIR/'factory_capacity.csv') # 생산 능력 데이터 로드

    # 수요 데이터 로드
    try:
        con = sqlite3.connect(DATA_DIR/'demand_train.db')
        dem_hist = pd.read_sql_query("SELECT date, sku, city, demand FROM demand_train", con)
        con.close()
        dem_hist['date'] = pd.to_datetime(dem_hist['date'])
        dem_hist['city'] = dem_hist['city'].map(norm_city)
    except Exception:
        dem_hist = pd.DataFrame(columns=['date','sku','city','demand'])

    dem_fc = pd.read_csv(DATA_DIR/'forecast_submission_template.csv')
    dem_fc['date'] = pd.to_datetime(dem_fc['date'])
    dem_fc['city'] = dem_fc['city'].map(norm_city)
    dem_fc = dem_fc.rename(columns={'mean':'demand'})

    dem = pd.concat([dem_hist, dem_fc], ignore_index=True)
    dem = dem[(dem['date']>=DATE_START)&(dem['date']<=DATE_END)]
    D_j = dem.groupby('city', as_index=False)['demand'].sum().rename(columns={'demand':'Q_city'})
    cities = sorted(D_j['city'].unique().tolist())
    
    total_demand = D_j['Q_city'].sum() # 전체 기간 총 수요량 계산

    # ---------- sets
    fac_df = sites[sites['site_type']=='factory'].copy()
    wh_df  = sites[sites['site_type']=='warehouse'].copy()

    I = fac_df['site_id'].tolist()
    K = wh_df['site_id'].tolist()
    J = cities

    site_lat = dict(zip(sites['site_id'], sites['lat']))
    site_lon = dict(zip(sites['site_id'], sites['lon']))
    site_country = dict(zip(sites['site_id'], sites['country']))
    site_city    = dict(zip(sites['site_id'], sites['city']))

    city_ll = sites.groupby('city')[['lat','lon']].mean().reset_index().set_index('city')
    city_lat = city_ll['lat'].to_dict()
    city_lon = city_ll['lon'].to_dict()

    city_country = {city: country for country, group in sites.groupby('country') for city in group['city'].unique()}

    # ---------- distances
    dist_ik = {(i,k): haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k]) for i in I for k in K}
    dist_kj = {(k,j): haversine_km(site_lat[k], site_lon[k], city_lat[j], city_lon[j]) for k in K for j in J}

    # ---------- transport per km cost
    beta = dict(zip(mode_meta['mode'], mode_meta['cost_per_km_factor']))
    costpkm_truck = TRUCK_BASE_COST_PER_KM * beta['TRUCK']
    costpkm_ship  = TRUCK_BASE_COST_PER_KM * beta['SHIP']

    def is_cross_border_u2v(cu: str, cv: str) -> bool:
        return cu != cv

    def border_cost_val(cu: str, cv: str) -> float:
        return 0.0 if not is_cross_border_u2v(cu, cv) else BORDER_COST

    def per_km_cost(from_country, to_country):
        return costpkm_ship if is_cross_border_u2v(from_country, to_country) else costpkm_truck

    # --- 💡 주요 변경점 START ---
    # 공장별 총 생산 능력 계산
    total_capacity_per_factory = fac_cap.groupby('factory')[['reg_capacity', 'ot_capacity']].sum().sum(axis=1)
    C_i = total_capacity_per_factory.to_dict()
    # --- 💡 주요 변경점 END ---
    
    # ---------- model
    m = Model("SITE_LOCATION_WITH_CAPACITY")
    m.Params.OutputFlag = 1

    # vars
    x_fac = m.addVars(I, vtype=GRB.BINARY, name="x_fac")
    x_wh  = m.addVars(K, vtype=GRB.BINARY, name="x_wh")
    a_kj = m.addVars(K, J, lb=0.0, name="assign_kj")
    f_ik = m.addVars(I, K, lb=0.0, name="flow_ik")

    # constraints
    m.addConstr(x_fac.sum() <= MAX_FAC, "fac_count")
    m.addConstr(x_wh.sum()  <= MAX_WH,  "wh_count")

    for city, g in fac_df.groupby('city'):
        m.addConstr(quicksum(x_fac[i] for i in g['site_id']) <= 1, f"one_fac_per_city_{city}")
    for city, g in wh_df.groupby('city'):
        m.addConstr(quicksum(x_wh[k] for k in g['site_id']) <= 1, f"one_wh_per_city_{city}")

    Q_by_city = dict(zip(D_j['city'], D_j['Q_city']))
    for j in J:
        m.addConstr(quicksum(a_kj[k,j] for k in K) == Q_by_city[j], f"city_assign_{j}")

    for k in K:
        for j in J:
            m.addConstr(a_kj[k,j] <= Q_by_city[j] * x_wh[k], f"use_open_wh_{k}_{j}")

    wh_demand = {k: quicksum(a_kj[k,j] for j in J) for k in K}
    for k in K:
        m.addConstr(quicksum(f_ik[i,k] for i in I) >= wh_demand[k], f"cover_wh_{k}")

    BIGM = total_demand
    for i in I:
        for k in K:
            m.addConstr(f_ik[i,k] <= BIGM * x_fac[i], f"use_open_fac_{i}_{k}")

    # --- 💡 주요 변경점 START ---
    # 총 생산능력이 총 수요량을 충족해야 한다는 제약 조건 추가
    m.addConstr(quicksum(C_i.get(i, 0) * x_fac[i] for i in I) >= total_demand, "capacity_covers_demand")
    # --- 💡 주요 변경점 END ---

    # objective
    init_cost_map = dict(zip(init_cost['site_id'], init_cost['init_cost_usd']))
    
    cost_ik = quicksum(f_ik[i,k] * (dist_ik[i,k] * per_km_cost(site_country[i], site_country[k]) + border_cost_val(site_country[i], site_country[k])) for i in I for k in K)
    cost_kj = quicksum(a_kj[k,j] * (dist_kj[k,j] * per_km_cost(site_country[k], city_country[j]) + border_cost_val(site_country[k], city_country[j])) for k in K for j in J)
    
    build_cost = quicksum(init_cost_map.get(i,0.0)*x_fac[i] for i in I) + quicksum(init_cost_map.get(k,0.0)*x_wh[k]  for k in K)

    m.setObjective(build_cost + cost_ik + cost_kj, GRB.MINIMIZE)
    m.optimize()

    # export results
    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0:
        print("No solution found.")
        return

    opened_fac = [i for i in I if x_fac[i].X > 0.5]
    opened_wh  = [k for k in K if x_wh[k].X  > 0.5]
    
    pd.DataFrame({"factory": opened_fac}).to_csv("selected_factories.csv", index=False)
    pd.DataFrame({"warehouse": opened_wh}).to_csv("selected_warehouses.csv", index=False)
    
    print("\n--- 1단계 입지 선정 완료 ---")
    print(f"선택된 공장 ({len(opened_fac)}개): {opened_fac}")
    print(f"선택된 창고 ({len(opened_wh)}개): {opened_wh}")
    print("결과가 'selected_factories.csv', 'selected_warehouses.csv' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
