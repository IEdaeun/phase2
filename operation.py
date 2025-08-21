# -*- coding: utf-8 -*-
# ============================================================================
# SKY 산공학술교류제 공급망 최적화 프로젝트 - 3단계: 일일 운영 시뮬레이션
#
# [목표]
# - 2단계에서 수립된 전술 계획(소싱/재고 정책)이 실제 운영 환경에서 어떤 성과를 보이는지 검증합니다.
# - 2018년부터 2024년까지의 기간에 대해, 모든 동적 제약조건(수요 변동, 기계 고장, 날씨 등) 하에서 일일 시뮬레이션을 수행하여 최종 운영 계획(DB)과 성과 지표(총비용, Fill-Rate)를 도출합니다.
#
# [핵심 로직]
# 1.  데이터 및 계획 로드: 2단계 결과(JSON), 마스터 데이터, 시계열 데이터(수요, 날씨, 유가 등)를 모두 로드합니다.
# 2.  상태 변수 초기화: 재고, 파이프라인, 누적 비용 등 시뮬레이션의 상태를 추적할 변수들을 초기화합니다.
# 3.  일일 시뮬레이션 루프 (2018-01-01 ~ 2024-12-31):
#     A. 입고 처리: 오늘 도착 예정인 운송 중인 재고를 창고 재고에 반영합니다.
#     B. 수요 처리: 당일 발생한 수요에 대해 창고 재고로 출고를 시도하고, 부족 시 품절 비용을 계산합니다.
#     C. 재고 보충: 각 창고-SKU의 재고 포지션(현재고+운송중재고)을 점검하여 재주문점(ROP) 이하이면 주 공급 공장으로 생산 주문을 생성합니다.
#     D. 생산 및 출고: 공장별 생산 주문 대기열을 처리합니다. 주간 생산 능력, 공휴일, 기계 고장을 고려하여 생산량을 결정하고, 생산된 제품은 즉시 창고로 발송 처리합니다.
#     E. 비용 계산: 생산비, 운송비, 재고유지비 등 일일 발생 비용을 모두 계산하여 누적합니다.
#     F. 로그 기록: 당일의 모든 활동(생산, 운송)을 DB 제출 형식에 맞게 기록합니다.
# 4.  최종 집계 및 저장: 시뮬레이션 종료 후, 누적된 운영 비용에 초기 건설비와 환경세를 더해 최종 총비용을 계산하고, 전체 수요 충족률(Fill-Rate)을 산출하여 결과 파일들을 생성합니다.
# ============================================================================

import pandas as pd
import numpy as np
import json, sqlite3, os, math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

# ---------------- Consts & IO ----------------
DATA_DIR = Path("data")
DATE_START = datetime(2018, 1, 1)
DATE_END = datetime(2024, 12, 31)
DB_NAME = "plan_submission_template.db"
TABLE_NAME = "plan_submission_template"

# --- PDF 기반 상수 ---
CONTAINER_CAPACITY = 4000.0  # 컨테이너당 적재량 (unit)
BORDER_FEE_PER_SHIPMENT = 4000.0  # 국경 통과 '건당' 비용
EU_ZONE = {'DEU', 'FRA'}
CO2_TAX_PER_TON = 200.0 # 톤당 환경 부담금 (USD)

WEATHER_MULT_BAD = 3.0  # 악천후 할증
OIL_JUMP_PCT = 0.05
OIL_MULT_JUMP = 2.0  # 유가 급등 주간 할증
MODE_CHANGE_FEE_PCT = 0.05 # 운송 모드 변경 수수료 (이전 4주 운송비의 %)

# ---------------- Helpers ----------------
def norm_city(x: str) -> str:
    return str(x).replace(' ', '_')

def truck_days_from_distance(dkm: float) -> int:
    d = max(0.0, float(dkm))
    if d <= 500.0:   return 2
    elif d <= 1000.: return 3
    elif d <= 2000.: return 5
    else:            return 8

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1) / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin((l1 - l2) / 2.0)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def is_cross_border(cu, cv):
    if cu == cv: return False
    if (cu in EU_ZONE) and (cv in EU_ZONE): return False
    return True

def compute_lead_days(site_info, mode_meta, fac_id, wh_id, default_mode="SHIP"):
    lat_f, lon_f = site_info.loc[fac_id, ['lat', 'lon']]
    lat_w, lon_w = site_info.loc[wh_id, ['lat', 'lon']]
    c_f = site_info.loc[fac_id, 'country']
    c_w = site_info.loc[wh_id, 'country']
    dist_km = haversine_km(lat_f, lon_f, lat_w, lon_w)
    truck_d = truck_days_from_distance(dist_km)
    
    if c_f == c_w:
        mode = "TRUCK" if default_mode != "AIR" else "AIR"
    else:
        mode = default_mode
        
    factor = float(mode_meta.loc[mode, 'leadtime_factor'])
    lead = int(np.ceil(truck_d * factor))
    return max(0, lead), mode, dist_km, c_f, c_w

def is_bad_weather(weather_by_cc, country, dt):
    row = weather_by_cc.get((country, dt.date()))
    if row is None: return False
    rain, snow, wind, cloud = row
    return (rain >= 45.7) or (snow >= 3.85) or (wind >= 13.46) or (cloud >= 100.0)

def oil_jump_week(oil_jumps_set, dt):
    return dt.date() in oil_jumps_set

def monday_of(dt): return (dt - timedelta(days=dt.weekday())).date()

# ---------------- Main ----------------
def main_stage3_simulator():
    print("--- 3단계 운영 시뮬레이션 시작 (최종 수정 버전) ---")

    # ---------- (1) 계획/데이터 로드 ----------
    print("\n[3-1] 데이터 및 전술 계획 로드 중...")
    try:
        with open("tactical_plan_advanced.json", "r", encoding='utf-8') as f:
            plan = json.load(f)
        sourcing_policy = plan['sourcing_policy']
        inventory_policy = plan['inventory_policy']
    except FileNotFoundError:
        print("오류: 2단계 결과 파일('tactical_plan_advanced.json')이 필요합니다.")
        return

    # 마스터 데이터
    sites_df = pd.read_csv(DATA_DIR / 'site_candidates.csv')
    sites_df['city'] = sites_df['city'].map(norm_city)
    site_info = sites_df.set_index('site_id')
    mode_meta = pd.read_csv(DATA_DIR / 'transport_mode_meta.csv').set_index('mode')
    init_cost_df = pd.read_csv(DATA_DIR / 'site_init_cost.csv')

    # 생산/노동 관련
    fac_cap_df = pd.read_csv(DATA_DIR / 'factory_capacity.csv', parse_dates=['week'])
    fac_cap_df['week'] = fac_cap_df['week'].dt.date
    labour_req = pd.read_csv(DATA_DIR / 'labour_requirement.csv')
    a_hours = dict(zip(labour_req['sku'], labour_req['labour_hours_per_unit']))
    lab_pol = pd.read_csv(DATA_DIR / 'labour_policy.csv')
    prod_cost = pd.read_csv(DATA_DIR / 'prod_cost_excl_labour.csv')
    base_cost = {(r['sku'], r['factory']): float(r['base_cost_usd']) for _, r in prod_cost.iterrows()}
    
    # 환경 관련
    carbon_prod = pd.read_csv(DATA_DIR / 'carbon_factor_prod.csv')
    # KeyError: 'sku' 해결: 'sku' 컬럼이 없으므로 factory만 key로 사용
    CO2_PROD = {r['factory']: float(r['kg_CO2_per_unit']) for _, r in carbon_prod.iterrows()}
    CO2_TRANS = dict(zip(mode_meta.index, mode_meta['co2_per_km_factor'])) # 트럭 기준 0.4kg/km에 대한 배율

    # 비용 테이블
    inv_cost_df = pd.read_csv(DATA_DIR / 'inv_cost.csv')
    short_cost_df = pd.read_csv(DATA_DIR / 'short_cost.csv')
    HOLD = dict(zip(inv_cost_df['sku'], inv_cost_df['inv_cost_per_day']))
    SHORT = dict(zip(short_cost_df['sku'], short_cost_df['short_cost_per_unit']))

    # 기계고장/휴일
    mfail = pd.read_csv(DATA_DIR / 'machine_failure_log.csv', parse_dates=['start_date', 'end_date'])
    hol = pd.read_csv(DATA_DIR / 'holiday_lookup.csv', parse_dates=['date'])
    HOL = {(r['country'], r['date'].date()): 1 for _, r in hol.iterrows()}

    # 환율
    fx_df = pd.read_csv(DATA_DIR / 'currency.csv', parse_dates=['Date'])
    fx_df = fx_df.rename(columns={'Date': 'date'}).set_index('date').ffill().reset_index()
    fx_long = fx_df.melt(id_vars=['date'], var_name='pair', value_name='usd_per_local_inv')
    fx_long['ccy'] = fx_long['pair'].str.replace('=X', '', regex=False)
    fx_long['usd_per_local'] = 1.0 / fx_long['usd_per_local_inv']
    ctry_ccy = dict(zip(lab_pol['country'], lab_pol['currency']))
    
    fx_map = {}
    for c, ccy in ctry_ccy.items():
        if ccy == 'USD':
            for d in pd.date_range(DATE_START, DATE_END):
                fx_map[(c, d.date())] = 1.0
        else:
            sub = fx_long[fx_long['ccy'] == ccy].set_index('date')['usd_per_local']
            for d in pd.date_range(DATE_START, DATE_END):
                fx_map[(c, d.date())] = sub.asof(d)
    
    # 수요 (과거+미래)
    try:
        with sqlite3.connect(DATA_DIR / 'demand_train.db') as con:
            d_hist = pd.read_sql("SELECT * FROM demand_train", con, parse_dates=['date'])
    except Exception:
        d_hist = pd.DataFrame()
    d_fut = pd.read_csv(DATA_DIR / 'forecast_submission_template.csv', parse_dates=['date'])
    all_dem = pd.concat([d_hist, d_fut.rename(columns={'mean':'demand'})], ignore_index=True)
    all_dem['city'] = all_dem['city'].map(norm_city)
    all_dem = all_dem.drop_duplicates(['date','sku','city'], keep='last').sort_values('date')

    # 도시→창고 매핑
    selected_whs = set(sourcing_policy.keys())
    city_ll = sites_df.groupby('city')[['lat','lon']].mean()
    city_to_wh = {c: min(selected_whs, key=lambda w: haversine_km(city_ll.loc[c,'lat'], city_ll.loc[c,'lon'], site_info.loc[w,'lat'], site_info.loc[w,'lon'])) for c in all_dem['city'].unique() if c in city_ll.index}
    
    dem_tmp = all_dem.copy()
    dem_tmp['warehouse'] = dem_tmp['city'].map(city_to_wh)
    mean_demand_wh_sku = dem_tmp.groupby(['warehouse', 'sku'])['demand'].mean().to_dict()

    # 악천후/유가 급등 할증 준비
    wx_df = pd.read_csv(DATA_DIR / 'weather.csv', parse_dates=['date'])
    weather_by_cc = {(r['country'], r['date'].date()): (r.get('rain_mm',0), r.get('snow_cm',0), r.get('wind_mps',0), r.get('cloud_pct',0)) for _, r in wx_df.iterrows()}
    oil_df = pd.read_csv(DATA_DIR / 'oil_price.csv', parse_dates=['date']).set_index('date').ffill().reset_index()
    oil_df['week_ago'] = oil_df['brent_usd'].shift(7)
    oil_df['jump'] = (oil_df['brent_usd'] - oil_df['week_ago']) / oil_df['week_ago'] >= OIL_JUMP_PCT
    oil_jump_dates = set(pd.to_datetime(oil_df[oil_df['jump']]['date']).dt.date)

    # 주간 용량 캘린더 ffill
    all_weeks = pd.date_range(start=DATE_START, end=DATE_END, freq='W-MON').date
    cap_raw = fac_cap_df.groupby(['factory', 'week'])[['reg_capacity', 'ot_capacity']].sum().reset_index()
    week_reg_cap, week_ot_cap = {}, {}
    for fac, g in cap_raw.groupby('factory'):
        g2 = pd.DataFrame({'week': all_weeks}).merge(g, on='week', how='left').sort_values('week').ffill().fillna(0)
        for _, r in g2.iterrows():
            week_reg_cap[(fac, r['week'])] = float(r['reg_capacity'])
            week_ot_cap[(fac, r['week'])]  = float(r['ot_capacity'])

    # ---------- (2) 상태변수 ----------
    print("\n[3-2] 시뮬레이션 상태 초기화 중...")
    inventory = defaultdict(float)
    for wh in selected_whs:
        for sku in inventory_policy.get(wh, {}).keys():
            inventory[(wh, sku)] = 2000.0

    open_pipeline = defaultdict(float)
    in_transit = defaultdict(list)
    factory_orders = defaultdict(deque)
    week_reg_used = defaultdict(float)
    week_ot_used = defaultdict(float)
    
    mode_tracker = {} 
    transport_cost_tracker = defaultdict(lambda: deque(maxlen=28))

    cost_accum = defaultdict(float)
    total_co2_kg_prod = 0.0
    total_co2_kg_transport = 0.0
    shortages_log, daily_cost_rows = [], []
    dem_week, ship_week = defaultdict(float), defaultdict(float)
    total_dem_all, total_ship_all = 0.0, 0.0
    
    db_rows_all = []

    # ---------- (3) 일일 루프 ----------------
    print("\n[3-3] 일일 시뮬레이션 실행 (2018-01-01 ~ 2024-12-31)...")
    for t_current in pd.date_range(DATE_START, DATE_END, freq='D'):
        if t_current.day == 1: print(f"  -> 진행 중: {t_current.date()}")
        wk_monday, weekday = monday_of(t_current), t_current.weekday()
        
        daily_prod = defaultdict(lambda: {'reg': 0, 'ot': 0})
        daily_ship_fac_wh = []
        daily_ship_wh_city = []
        
        for dest_wh, sku, qty in in_transit.pop(t_current.date(), []):
            inventory[(dest_wh, sku)] += qty
            open_pipeline[(dest_wh, sku)] = max(0.0, open_pipeline[(dest_wh, sku)] - qty)

        today_dem = all_dem[all_dem['date'] == t_current]
        day_short_cost = 0.0
        for _, r in today_dem.iterrows():
            city, sku, dqty = r['city'], r['sku'], float(r['demand'] or 0.0)
            if dqty <= 0: continue
            wh = city_to_wh.get(city)
            if not wh: continue

            avail = inventory.get((wh, sku), 0.0)
            ship = min(avail, dqty)
            shortage = dqty - ship

            if ship > 0:
                inventory[(wh, sku)] -= ship
                daily_ship_wh_city.append({'wh': wh, 'sku': sku, 'qty': ship, 'city': city})
            if shortage > 0:
                day_short_cost += float(SHORT.get(sku, 0.0)) * shortage
                shortages_log.append({'date': t_current.date(), 'city': city, 'sku': sku, 'short_qty': shortage})
            
            dem_week[(city, sku, wk_monday)] += dqty
            ship_week[(city, sku, wk_monday)] += ship
            total_dem_all += dqty
            total_ship_all += ship
        cost_accum['shortage'] += day_short_cost

        for wh, sku_dict in inventory_policy.items():
            for sku, pol in sku_dict.items():
                on_hand = inventory.get((wh, sku), 0.0)
                in_pipe = open_pipeline.get((wh, sku), 0.0)
                rop = float(pol.get('reorder_point', 0))

                if on_hand + in_pipe <= rop:
                    primary_fac = sourcing_policy[wh]['primary_source']['factory']
                    lt_days = sourcing_policy[wh]['primary_source']['lead_time']
                    mean_d = float(mean_demand_wh_sku.get((wh, sku), 0.0))
                    target_S = float(pol.get('safety_stock', 0)) + mean_d * lt_days * 2.0
                    order_qty = max(0.0, target_S - (on_hand + in_pipe))

                    if order_qty > 0:
                        is_domestic = site_info.loc[primary_fac, 'country'] == site_info.loc[wh, 'country']
                        default_mode = "TRUCK" if is_domestic else "SHIP"
                        days_to_depletion = on_hand / max(mean_d, 1e-6)
                        preferred_mode = "AIR" if days_to_depletion <= lt_days and not is_domestic else default_mode
                        factory_orders[primary_fac].append({'wh': wh, 'sku': sku, 'qty': order_qty, 'preferred_mode': preferred_mode})

        mfail_today = {r['factory'] for _, r in mfail.iterrows() if r['start_date'] <= t_current <= r['end_date']}
        day_prod_labor, day_prod_base, day_trans_cost, day_mode_change_fee = 0, 0, 0, 0
        
        for fac, q in list(factory_orders.items()):
            if not q or fac in mfail_today: continue
            
            fac_ctry = site_info.loc[fac, 'country']
            is_holiday = HOL.get((fac_ctry, t_current.date()), 0) == 1 or weekday >= 5
            reg_left = max(0.0, week_reg_cap.get((fac, wk_monday), 0.0) - week_reg_used.get((fac, wk_monday), 0.0))
            ot_left = max(0.0, week_ot_cap.get((fac, wk_monday), 0.0) - week_ot_used.get((fac, wk_monday), 0.0))
            reg_left_today = 0.0 if is_holiday else reg_left
            
            new_q = deque()
            while q and (reg_left_today > 1e-9 or ot_left > 1e-9):
                order = q.popleft()
                to_wh, sku, qty, preferred = order['wh'], order['sku'], order['qty'], order['preferred_mode']
                hrs_per_unit = float(a_hours.get(sku, 0.0))
                if qty <= 1e-9 or hrs_per_unit <= 0: continue

                total_h_need = qty * hrs_per_unit
                reg_h_assign = min(reg_left_today, total_h_need)
                ot_h_assign = min(ot_left, total_h_need - reg_h_assign)
                
                reg_units = np.floor(reg_h_assign / hrs_per_unit) if hrs_per_unit > 0 else 0
                ot_units = np.floor(ot_h_assign / hrs_per_unit) if hrs_per_unit > 0 else 0
                produced_units = reg_units + ot_units
                if produced_units <= 0:
                    new_q.append(order)
                    continue

                base_c = float(base_cost.get((sku, fac), 0.0))
                day_prod_base += base_c * produced_units
                pol = lab_pol[(lab_pol['country'] == fac_ctry) & (lab_pol['year'] == t_current.year)].iloc[0]
                usd_per_local = fx_map.get((fac_ctry, t_current.date()), 1.0)
                reg_wage_usd = pol['regular_wage_local'] * usd_per_local
                reg_h_used = reg_units * hrs_per_unit
                ot_h_used = ot_units * hrs_per_unit
                day_prod_labor += (reg_h_used * reg_wage_usd) + (ot_h_used * reg_wage_usd * pol['ot_mult'])
                # KeyError: 'sku' 해결: fac만 사용해 CO2 배출량 조회
                total_co2_kg_prod += float(CO2_PROD.get(fac, 0.0)) * produced_units

                _, _, dist_km, c_f, c_w = compute_lead_days(site_info, mode_meta, fac, to_wh, "SHIP")
                mode_key = (fac, to_wh, weekday)
                tracker = mode_tracker.get(mode_key)
                
                mode_used = preferred
                if tracker and tracker['lock_end_date'] >= t_current.date():
                    mode_used = tracker['mode']
                
                if tracker and tracker['mode'] != mode_used and tracker['lock_end_date'] < t_current.date():
                    past_costs = transport_cost_tracker.get(mode_key, [])
                    if past_costs:
                        day_mode_change_fee += sum(past_costs) * MODE_CHANGE_FEE_PCT
                
                mode_tracker[mode_key] = {'mode': mode_used, 'lock_end_date': (t_current + timedelta(days=27)).date()}

                lead_days = int(np.ceil(truck_days_from_distance(dist_km) * float(mode_meta.loc[mode_used, 'leadtime_factor'])))
                arr_day = (t_current + timedelta(days=lead_days)).date()

                num_containers = math.ceil(produced_units / CONTAINER_CAPACITY)
                beta = float(mode_meta.loc[mode_used, 'cost_per_km_factor'])
                base_trans_cost = (12.0 * beta) * dist_km * num_containers
                border = BORDER_FEE_PER_SHIPMENT if is_cross_border(c_f, c_w) else 0.0
                wx_mult = WEATHER_MULT_BAD if is_bad_weather(weather_by_cc, c_f, t_current) else 1.0
                oil_mult = OIL_MULT_JUMP if t_current.date() in oil_jump_dates else 1.0
                trans_cost = (base_trans_cost + border) * wx_mult * oil_mult
                day_trans_cost += trans_cost
                transport_cost_tracker[mode_key].append(trans_cost)
                co2_factor_truck = 0.40
                total_co2_kg_transport += dist_km * (co2_factor_truck * CO2_TRANS[mode_used]) * num_containers
                
                in_transit[arr_day].append((to_wh, sku, produced_units))
                open_pipeline[(to_wh, sku)] += produced_units
                week_reg_used[(fac, wk_monday)] += reg_h_used
                week_ot_used[(fac, wk_monday)] += ot_h_used
                reg_left_today = max(0.0, reg_left_today - reg_h_used)
                ot_left = max(0.0, ot_left - ot_h_used)

                daily_prod[fac]['reg'] += reg_units
                daily_prod[fac]['ot'] += ot_units
                daily_ship_fac_wh.append({'fac': fac, 'sku': sku, 'qty': produced_units, 'wh': to_wh, 'mode': mode_used})
                
                leftover_qty = qty - produced_units
                if leftover_qty > 1e-9:
                    order['qty'] = leftover_qty
                    new_q.append(order)

            factory_orders[fac] = new_q + q
            
        cost_accum['production_labor'] += day_prod_labor
        cost_accum['production_base'] += day_prod_base
        cost_accum['transport'] += day_trans_cost
        cost_accum['mode_change_fee'] += day_mode_change_fee

        day_hold_cost = sum(float(HOLD.get(sku, 0.0)) * qv for (_, sku), qv in inventory.items() if qv > 0)
        cost_accum['holding'] += day_hold_cost

        total_day_cost = day_short_cost + day_prod_labor + day_prod_base + day_trans_cost + day_mode_change_fee + day_hold_cost
        daily_cost_rows.append({'date': t_current.date(), 'transport': day_trans_cost, 'holding': day_hold_cost, 'shortage': day_short_cost, 'prod_labor': day_prod_labor, 'prod_base': day_prod_base, 'mode_change': day_mode_change_fee, 'total': total_day_cost})

        # DB 제출 형식 규칙에 맞게 행 생성
        db_rows_today = []
        for fac, prods in daily_prod.items():
            total_reg = int(round(prods['reg']))
            total_ot = int(round(prods['ot']))
            for ship in daily_ship_fac_wh:
                if ship['fac'] == fac:
                    db_rows_today.append({"date": t_current.strftime("%Y-%m-%d"), "factory/warehouse": fac, "sku": ship['sku'], "production_qty": total_reg, "ot_qty": total_ot, "ship_qty": int(round(ship['qty'])), "from": fac, "to": ship['wh'], "mode": ship['mode']})
        for ship in daily_ship_wh_city:
            db_rows_today.append({"date": t_current.strftime("%Y-%m-%d"), "factory/warehouse": ship['wh'], "sku": ship['sku'], "production_qty": 0, "ot_qty": 0, "ship_qty": int(round(ship['qty'])), "from": ship['wh'], "to": ship['city'], "mode": "TRUCK"})
        
        db_rows_all.extend(db_rows_today)

    # ---------------- (4) 최종 비용 계산 및 리포트 저장 ----------------
    print("\n[3-4] 최종 비용 계산 및 DB/리포트 저장 중...")
    
    # DB 저장 (한 번에)
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    submission_df = pd.DataFrame(db_rows_all)
    with sqlite3.connect(DB_NAME) as conn:
        submission_df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # 초기 건설비 추가
    selected_facs = pd.read_csv("selected_factories.csv")['factory'].tolist()
    selected_whs = pd.read_csv("selected_warehouses.csv")['warehouse'].tolist()
    init_build_cost = init_cost_df[init_cost_df['site_id'].isin(selected_facs + selected_whs)]['init_cost_usd'].sum()
    cost_accum['initial_build'] = init_build_cost
    
    # 환경부담금 추가
    total_co2_ton = math.ceil((total_co2_kg_prod + total_co2_kg_transport) / 1000.0)
    env_tax = total_co2_ton * CO2_TAX_PER_TON
    cost_accum['environment_tax'] = env_tax

    pd.DataFrame(daily_cost_rows).to_csv("cost_daily.csv", index=False)
    fr_rows = [{"week_monday": wk.strftime("%Y-%m-%d"), "city": city, "sku": sku, "demand": d, "shipped": ship_week.get((city, sku, wk), 0.0), "fill_rate": ship_week.get((city, sku, wk), 0.0) / d if d > 0 else 0} for (city, sku, wk), d in dem_week.items()]
    pd.DataFrame(fr_rows).to_csv("fr_weekly.csv", index=False)
    
    fr_overall_data = {
        "total_demand": total_dem_all,
        "total_shipped": total_ship_all,
        "fill_rate_overall": total_ship_all / total_dem_all if total_dem_all > 0 else 0,
        "total_co2_ton_rounded_up": total_co2_ton
    }
    fr_overall_data.update(cost_accum)
    fr_overall_data["grand_total_cost"] = sum(cost_accum.values())
    pd.DataFrame([fr_overall_data]).to_csv("fr_overall.csv", index=False)

    pd.DataFrame(shortages_log).to_csv("shortages_log.csv", index=False)

    print(f"\n--- 3단계 시뮬레이션 완료 ---")
    print(f"최종 Fill-Rate: {fr_overall_data['fill_rate_overall']:.4%}")
    print(f"최종 총 비용 (USD): {fr_overall_data['grand_total_cost']:,.2f}")
    print(f"- 제출 DB: {DB_NAME} (table: {TABLE_NAME})")

if __name__ == "__main__":
    main_stage3_simulator()



