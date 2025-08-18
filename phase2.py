# ============================================================================
#  Hybrid SCM Solver V6 (MIP for Strategy, Rolling LP for Operations)
#
#  - VERSION 6 BUGFIX:
#    Corrected a critical bug in `solve_operational_lp_chunk` where shipment
#    records from warehouses to cities (Y variables) were not being appended
#    to the results list. Also fixed a minor bug in the final inventory
#    carryover calculation. This should be the final fix to get a populated output.
# ============================================================================
from __future__ import annotations
import math, gc, os, sqlite3, psutil, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Set

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# ======================= CONFIG =======================
DATA_DIR = Path("data")
DB_NAME = "plan_submission_template.db"
TABLE_NAME = "plan_submission_template"
DATE_START = datetime(2018, 1, 1)
DATE_END = datetime(2024, 12, 31)
CHUNK_DAYS = 31 
RANDOM_SEED = 42
# ======================================================

PROC = psutil.Process(os.getpid())
def log(tag: str): print(f"[{tag:<12}] RSS = {PROC.memory_info().rss / 1024 ** 2:,.1f} MB")
def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2.0) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))
def week_of(dt: datetime): return (dt.date() - DATE_START.date()).days // 7

def load_and_prep_data():
    log("Load Data")
    sites = pd.read_csv(DATA_DIR / 'site_candidates.csv')
    sku_meta = pd.read_csv(DATA_DIR / 'sku_meta.csv')
    init_cost_df = pd.read_csv(DATA_DIR / 'site_init_cost.csv')
    prod_cost_df = pd.read_csv(DATA_DIR / 'prod_cost_excl_labour.csv')
    inv_cost_df = pd.read_csv(DATA_DIR / 'inv_cost.csv')
    short_cost_df = pd.read_csv(DATA_DIR / 'short_cost.csv')
    capacity_df = pd.read_csv(DATA_DIR / 'factory_capacity.csv')
    try:
        con = sqlite3.connect(DATA_DIR / 'demand_train.db'); demand_train = pd.read_sql_query("SELECT * FROM demand_train", con); con.close()
    except: demand_train = pd.DataFrame()
    forecast_df = pd.read_csv(DATA_DIR / 'forecast_submission_template.csv').rename(columns={'mean': 'demand'})
    all_dem = pd.concat([demand_train, forecast_df], ignore_index=True); all_dem['date'] = pd.to_datetime(all_dem['date'])
    I = sorted(sites[sites['site_type'] == 'factory']['site_id'].tolist()); K = sorted(sites[sites['site_type'] == 'warehouse']['site_id'].tolist())
    J = sorted(sites['city'].unique().tolist()); SKUS = sorted(sku_meta['sku'].unique().tolist())
    site_info = sites.set_index('site_id'); site_city = site_info['city'].to_dict()
    init_cost = init_cost_df.set_index('site_id')['init_cost_usd'].to_dict()
    D = {(r.city, r.sku, r.date): r.demand for _, r in all_dem.iterrows() if r.demand > 0}
    city_coords = sites.drop_duplicates(subset=['city']).set_index('city'); city_lat = city_coords['lat'].to_dict(); city_lon = city_coords['lon'].to_dict()
    site_lat = site_info['lat'].to_dict(); site_lon = site_info['lon'].to_dict()
    dist_ik = {(i, k): haversine_km(site_lat[i], site_lon[i], site_lat[k], site_lon[k]) for i in I for k in K}
    dist_kj = {(k, j): haversine_km(site_lat[k], site_lon[k], city_lat.get(j, 0), city_lon.get(j, 0)) for k in K for j in J}
    C_prod = {(r.sku, r.factory): r.base_cost_usd for _, r in prod_cost_df.iterrows()}; C_inv = dict(zip(inv_cost_df['sku'], inv_cost_df['inv_cost_per_day']))
    C_short = dict(zip(short_cost_df['sku'], short_cost_df['short_cost_per_unit'])); C_trans = 0.01
    mondays = pd.to_datetime(capacity_df['week']); capacity_df['week_idx'] = mondays.apply(week_of)
    Cap_reg = {(r.factory, r.week_idx): r.reg_capacity for _, r in capacity_df.iterrows()}
    L_ik = {(i, k): 2 for i, k in dist_ik.keys()}; L_kj = {(k, j): 2 for k, j in dist_kj.keys()}
    log("Data Ready")
    return I, K, J, SKUS, site_city, init_cost, dist_ik, dist_kj, D, C_prod, C_inv, C_short, C_trans, Cap_reg, L_ik, L_kj

def solve_strategic_mip(I, K, J, SKUS, dates_all, site_city, init_cost, dist_ik, dist_kj, D):
    log("MIP Start")
    m = Model("SCM_Strategic_Placement_V3")
    m.setParam('Seed', RANDOM_SEED); m.setParam('TimeLimit', 420); m.setParam('OutputFlag', 0)
    x_fac = m.addVars(I, vtype=GRB.BINARY, name="x_fac"); x_wh = m.addVars(K, vtype=GRB.BINARY, name="x_wh")
    avg_demand_j = {j: np.mean([D.get((j, s, t), 0) for s in SKUS for t in dates_all]) for j in J}
    flow_ik=m.addVars(I,K,lb=0.); flow_kj=m.addVars(K,J,lb=0.); unmet_j=m.addVars(J,lb=0.)
    m.addConstr(x_fac.sum()<=5); m.addConstr(x_wh.sum()<=20)
    for c in {c for c in site_city.values()}: m.addConstr(quicksum(x_fac[i] for i in I if site_city[i]==c)<=1); m.addConstr(quicksum(x_wh[k] for k in K if site_city[k]==c)<=1)
    for j in J: m.addConstr(quicksum(flow_kj[k, j] for k in K) + unmet_j[j] >= avg_demand_j.get(j, 0))
    for k in K: m.addConstr(quicksum(flow_kj[k, j] for j in J) <= quicksum(flow_ik[i, k] for i in I))
    cap_proxy=sum(avg_demand_j.values()); [m.addConstr(quicksum(flow_ik[i,k] for k in K)<=cap_proxy*x_fac[i]) for i in I]; [m.addConstr(quicksum(flow_kj[k,j] for j in J)<=cap_proxy*x_wh[k]) for k in K]
    cost_build=quicksum(init_cost.get(i,0)*x_fac[i] for i in I)+quicksum(init_cost.get(k,0)*x_wh[k] for k in K)
    cost_trans=quicksum(dist_ik.get((i,k),1e4)*flow_ik[i,k] for i,k in flow_ik)+quicksum(dist_kj.get((k,j),1e4)*flow_kj[k,j] for k,j in flow_kj)
    cost_unmet=quicksum(unmet_j[j]*1e6 for j in J); m.setObjective(cost_build+cost_trans+cost_unmet, GRB.MINIMIZE)
    m.optimize()
    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0: return set(), set()
    opened_factories = {i for i in I if x_fac[i].X > 0.5}; opened_warehouses = {k for k in K if x_wh[k].X > 0.5}
    log("MIP Done"); print(f"  -> Factories: {len(opened_factories)}, Warehouses: {len(opened_warehouses)}")
    return opened_factories, opened_warehouses

def solve_operational_lp_chunk(
    chunk_dates: List[datetime], opened_factories: Set[str], opened_warehouses: Set[str],
    initial_inventory: Dict, in_transit_carryover: Dict, all_data: Dict
) -> Tuple[List[Dict], Dict, Dict]:
    """Solves a detailed LP for a single chunk, now with in-transit inventory."""
    SKUS, J, D = all_data['SKUS'], all_data['J'], all_data['D']
    C_prod,C_inv,C_short,C_trans = all_data['C_prod'],all_data['C_inv'],all_data['C_short'],all_data['C_trans']
    Cap_reg, L_ik, L_kj = all_data['Cap_reg'], all_data['L_ik'], all_data['L_kj']
    dist_ik, dist_kj = all_data['dist_ik'], all_data['dist_kj']
    
    m = Model(f"OpLP_{chunk_dates[0].date()}"); m.setParam('OutputFlag', 0)

    P=m.addVars(opened_factories,SKUS,chunk_dates,lb=0.); Inv=m.addVars(opened_warehouses,SKUS,chunk_dates,lb=0.)
    X=m.addVars(opened_factories,opened_warehouses,SKUS,chunk_dates,lb=0.)
    Y=m.addVars(opened_warehouses,J,SKUS,chunk_dates,lb=0.); Short=m.addVars(J,SKUS,chunk_dates,lb=0.)
    
    weeks_in_chunk=sorted({week_of(t) for t in chunk_dates})
    for i in opened_factories:
        for w in weeks_in_chunk:
            days_in_week=[t for t in chunk_dates if week_of(t)==w]
            m.addConstr(quicksum(P[i,s,t] for s in SKUS for t in days_in_week) <= Cap_reg.get((i, w),0))

    for i in opened_factories:
        for s in SKUS:
            for t in chunk_dates: m.addConstr(P[i,s,t]==quicksum(X[i,k,s,t] for k in opened_warehouses))

    for k in opened_warehouses:
        for s in SKUS:
            for idx, t in enumerate(chunk_dates):
                prev_inv = initial_inventory.get((k,s),0) if idx==0 else Inv[k,s,chunk_dates[idx-1]]
                arrivals_from_ik = quicksum(X[i,k,s,t-timedelta(days=L_ik.get((i,k),2))] for i in opened_factories if t-timedelta(days=L_ik.get((i,k),2)) in chunk_dates)
                carryover_ik = in_transit_carryover.get(('ik', t, k, s), 0)
                departures_to_kj = quicksum(Y[k,j,s,t] for j in J)
                m.addConstr(Inv[k,s,t] == prev_inv + arrivals_from_ik + carryover_ik - departures_to_kj)

    for j in J:
        for s in SKUS:
            for t in chunk_dates:
                demand = D.get((j,s,t),0)
                arrivals_from_kj = quicksum(Y[k,j,s,t-timedelta(days=L_kj.get((k,j),2))] for k in opened_warehouses if t-timedelta(days=L_kj.get((k,j),2)) in chunk_dates)
                carryover_kj = in_transit_carryover.get(('kj', t, j, s), 0)
                m.addConstr(arrivals_from_kj + carryover_kj >= demand - Short[j,s,t])

    cost_prod=quicksum(C_prod.get((s,i),100)*P[i,s,t] for i,s,t in P)
    cost_inv=quicksum(C_inv.get(s,0.1)*Inv[k,s,t] for k,s,t in Inv)
    cost_short=quicksum(C_short.get(s,10)*Short[j,s,t] for j,s,t in Short)
    cost_trans=quicksum(C_trans*dist_ik[i,k]*X[i,k,s,t] for i,k,s,t in X) + quicksum(C_trans*dist_kj[k,j]*Y[k,j,s,t] for k,j,s,t in Y)
    m.setObjective(cost_prod+cost_inv+cost_short+cost_trans, GRB.MINIMIZE)
    m.optimize()

    results=[]; new_in_transit={}
    if m.Status == GRB.OPTIMAL:
        for i,s,t in P.keys():
            if P[i,s,t].X > 0.1: results.append({"date":t.strftime('%Y-%m-%d'),"factory/warehouse":i,"sku":s,"production_qty":round(P[i,s,t].X),"ot_qty":0,"ship_qty":0,"from":i,"to":None,"mode":None})
        for i,k,s,t in X.keys():
            if X[i,k,s,t].X > 0.1:
                qty=round(X[i,k,s,t].X)
                results.append({"date":t.strftime('%Y-%m-%d'),"factory/warehouse":i,"sku":s,"production_qty":0,"ot_qty":0,"ship_qty":qty,"from":i,"to":k,"mode":'TRUCK'})
                arrival_date = t + timedelta(days=L_ik.get((i,k),2))
                if arrival_date not in chunk_dates: new_in_transit[('ik',arrival_date,k,s)] = new_in_transit.get(('ik',arrival_date,k,s),0)+qty
        for k,j,s,t in Y.keys():
            if Y[k,j,s,t].X > 0.1:
                qty = round(Y[k, j, s, t].X)
                # --- THIS IS THE CORRECTED PART ---
                results.append({"date":t.strftime('%Y-%m-%d'),"factory/warehouse":k,"sku":s,"production_qty":0,"ot_qty":0,"ship_qty":qty,"from":k,"to":j,"mode":'TRUCK'})
                arrival_date = t + timedelta(days=L_kj.get((k,j),2))
                if arrival_date not in chunk_dates: new_in_transit[('kj',arrival_date,j,s)] = new_in_transit.get(('kj',arrival_date,j,s),0)+qty
    
    # Corrected final inventory calculation
    final_inventory = {(k,s): Inv[k,s,chunk_dates[-1]].X for k in opened_warehouses for s in SKUS} if m.SolCount > 0 else initial_inventory
    return results, final_inventory, new_in_transit

def run_pipeline():
    (I,K,J,SKUS,site_city,init_cost,dist_ik,dist_kj,D,C_prod,C_inv,C_short,C_trans,Cap_reg,L_ik,L_kj) = load_and_prep_data()
    dates_all = sorted(list({d for c,s,d in D.keys()})) or list(pd.date_range(DATE_START, DATE_END, freq='D'))
    all_data = {'SKUS':SKUS,'J':J,'D':D,'C_prod':C_prod,'C_inv':C_inv,'C_short':C_short,'C_trans':C_trans,'Cap_reg':Cap_reg,'L_ik':L_ik,'L_kj':L_kj,'dist_ik':dist_ik,'dist_kj':dist_kj}
    opened_factories, opened_warehouses = solve_strategic_mip(I,K,J,SKUS,dates_all,site_city,init_cost,dist_ik,dist_kj,D)

    if not opened_factories or not opened_warehouses: print("MIP did not select a valid network. Exiting."); return

    all_results = []
    inventory_carryover = {(k,s): 2000 for k in opened_warehouses for s in SKUS}
    in_transit_carryover = {}
    current_date = dates_all[0]
    while current_date <= dates_all[-1]:
        chunk_end_date = min(current_date + timedelta(days=CHUNK_DAYS - 1), dates_all[-1])
        chunk_dates = [d for d in dates_all if current_date <= d <= chunk_end_date]
        if not chunk_dates: break
        
        print(f"--- Solving LP for {chunk_dates[0].date()} to {chunk_dates[-1].date()} ---")
        chunk_results, final_inv, new_in_transit = solve_operational_lp_chunk(chunk_dates, opened_factories, opened_warehouses, inventory_carryover, in_transit_carryover, all_data)
        
        all_results.extend(chunk_results)
        inventory_carryover = final_inv
        in_transit_carryover = new_in_transit
        current_date = chunk_end_date + timedelta(days=1)
        
    log("Write DB")
    df_final = pd.DataFrame(all_results)
    all_cols=["date","factory/warehouse","sku","production_qty","ot_qty","ship_qty","from","to","mode"]
    for col in all_cols:
        if col not in df_final.columns: df_final[col] = 0 if 'qty' in col else None
    df_final = df_final[all_cols].fillna({'production_qty':0,'ot_qty':0,'ship_qty':0})
    for col in ["production_qty","ot_qty","ship_qty"]: df_final[col] = df_final[col].astype(int)
    if os.path.exists(DB_NAME): os.remove(DB_NAME)
    with sqlite3.connect(DB_NAME) as conn: df_final.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    print(f"✅  결과 DB 작성 완료  ->  {DB_NAME}"); print(f"    Total rows: {len(df_final)}")

if __name__ == "__main__":
    run_pipeline()