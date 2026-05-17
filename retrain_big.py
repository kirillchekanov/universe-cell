import json, math, numpy as np, random
from datetime import datetime, timedelta
from collections import Counter
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import urllib.request

print("RETRAIN BIG — 328 событий + 15 тел + расширенные аспекты\n")

with open("big_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

print(f"События: {len(events)}")
cats = Counter(e["category"] for e in events)
for cat, n in cats.most_common():
    print(f"  {cat:<22} {n:>4}  {'█'*(n//5)}")

# Загружаем солнечные пятна
sunspots = {}
try:
    url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        for line in r.read().decode().split("\n"):
            p = line.strip().split()
            if len(p) >= 4:
                try:
                    yr,mo,sn = int(p[0]),int(p[1]),float(p[3])
                    for d in range(1,32):
                        sunspots[f"{yr:04d}-{mo:02d}-{d:02d}"] = sn
                except: pass
    print(f"\nСолнечные пятна: {len(sunspots)} дней")
except: print("Пятна: недоступны")

KP_STORMS = {"1989-03-13":9.0,"1991-03-24":9.0,"2000-07-15":9.0,
    "2001-03-31":9.0,"2003-10-29":9.0,"2003-11-20":9.0,
    "2015-03-17":8.3,"2024-05-10":9.0,"2024-10-10":9.0}

def get_sn(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    for delta in range(32):
        d = (dt-timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in sunspots: sn=sunspots[d]; return [sn/300.0,1.0 if sn>150 else 0.0,1.0 if sn<10 else 0.0]
    return [0.5,0.0,0.0]

def get_kp(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    mx=0.0
    for sd,kp in KP_STORMS.items():
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days)<=15: mx=max(mx,kp)
    return [mx/9.0, 1.0 if mx>=7 else 0.0]

def get_lunar(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    moon=ephem.Moon(); moon.compute(obs)
    sun=ephem.Sun(); sun.compute(obs)
    dm=float(moon.earth_distance); ds2=float(sun.earth_distance)
    mn=(dm-0.00247)/(0.00272-0.00247)
    sn2=(ds2-0.983)/(1.017-0.983)
    ph=float(moon.phase)/100.0
    return [mn,sn2,ph,1.0 if mn<0.1 else 0.0,1.0 if mn>0.9 else 0.0]

def features(date_str):
    dt = datetime.strptime(date_str,"%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")

    # 9 основных планет
    bodies = [ephem.Sun(),ephem.Moon(),ephem.Mercury(),ephem.Venus(),
              ephem.Mars(),ephem.Jupiter(),ephem.Saturn(),
              ephem.Uranus(),ephem.Neptune()]
    f = []
    lons = {}
    for i,b in enumerate(bodies):
        b.compute(obs)
        lon = float(b.hlong)
        lons[i] = lon
        f += [math.sin(lon), math.cos(lon)]

    # Хирон (вычисляем через эфемериды вручную — приближение)
    # Хирон открыт 1977, период ~50 лет
    # Приближение через эллиптическую орбиту
    yr = dt.year + dt.timetuple().tm_yday/365.25
    chiron_lon = ((yr - 1996.0) / 50.85 * 360) % 360  # перигелий 1996
    f += [math.sin(math.radians(chiron_lon)), math.cos(math.radians(chiron_lon))]

    # Лунные узлы (Раху) — период 18.613 лет
    node_lon = ((yr - 2000.0) / 18.613 * 360) % 360
    f += [math.sin(math.radians(node_lon)), math.cos(math.radians(node_lon))]

    # Лилит — средняя Чёрная Луна, период 8.85 лет
    lilith_lon = ((yr - 2000.0) / 8.85 * 360) % 360
    f += [math.sin(math.radians(lilith_lon)), math.cos(math.radians(lilith_lon))]

    # Расширенные аспекты — все значимые пары
    planet_lons = [float(b.hlong) for b in bodies[:7]]  # Sun-Saturn
    pairs = [(0,4),(0,5),(0,6),(1,5),(1,6),(2,5),(3,4),(3,5),(4,5),(4,6),(5,6)]
    for i,j in pairs:
        asp = abs(math.degrees(planet_lons[i]-planet_lons[j])) % 360
        # Гармоники аспекта
        f += [math.sin(math.radians(asp)),
              math.cos(math.radians(asp)),
              math.sin(math.radians(asp*2))]  # 2я гармоника

    # Лунные узлы через moon
    moon2=ephem.Moon(); moon2.compute(obs)
    f += [math.sin(float(moon2.hlong)+math.pi), math.cos(float(moon2.hlong)+math.pi)]

    # Гравитация
    best = min(grav_data, key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f += [best.get("grav_potential_log",0),
          best.get("angular_momentum_log",0),
          best.get("tidal_force",0)]

    # Геофизика
    f += get_kp(date_str)
    f += get_sn(date_str)
    f += get_lunar(date_str)

    # Год как признак (долгосрочные тренды)
    f += [math.sin(2*math.pi*yr/11),   # 11-летний солнечный цикл
          math.cos(2*math.pi*yr/11),
          math.sin(2*math.pi*yr/18.6), # Лунный узловой цикл
          math.cos(2*math.pi*yr/18.6),
          math.sin(2*math.pi*yr/29.5), # Цикл Сатурна
          math.cos(2*math.pi*yr/29.5)]

    return f

# Тест
test_f = features("2003-10-29")
n_feat = len(test_f)
print(f"\nПризнаков: {n_feat} (было 48, стало {n_feat})")

# Датасет
random.seed(42)
X, y = [], []
errors = 0
for e in events:
    try: X.append(features(e["date"])); y.append(e["category"])
    except: errors += 1

print(f"Событий: {len(y)} (ошибок: {errors})")

start = datetime(1800,1,1)
edates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
n = 0
for _ in range(20000):
    d = start + timedelta(days=random.randint(0,81000))
    if d.year > 2024: continue
    if not any(abs((d-ed).days)<45 for ed in edates):
        try: X.append(features(d.strftime("%Y-%m-%d"))); y.append("neutral"); n+=1
        except: pass
    if n >= len(y)//2: break

le = LabelEncoder(); y_enc = le.fit_transform(y)
X = np.array(X); classes = list(le.classes_)
print(f"Датасет: {len(y)} точек · {X.shape[1]} признаков")
print(f"Классы: {dict(Counter(y))}\n")

# ══ WALK-FORWARD ВАЛИДАЦИЯ ══
SPLIT = 1990  # train 1800-1989, test 1990-2024
events_train = [e for e in events if int(e["date"][:4]) < SPLIT]
events_test  = [e for e in events if int(e["date"][:4]) >= SPLIT]
print(f"Walk-forward: train {len(events_train)} · test {len(events_test)}")

def build_set(evs, start_yr, end_yr):
    Xb, yb = [], []
    edts = [datetime.strptime(e["date"],"%Y-%m-%d") for e in evs]
    for e in evs:
        try: Xb.append(features(e["date"])); yb.append(e["category"])
        except: pass
    st = datetime(start_yr,1,1); en = datetime(end_yr,12,31)
    nn = 0
    for _ in range(10000):
        d = st+timedelta(days=random.randint(0,(en-st).days))
        if not any(abs((d-ed).days)<45 for ed in edts):
            try: Xb.append(features(d.strftime("%Y-%m-%d"))); yb.append("neutral"); nn+=1
            except: pass
        if nn >= len(Xb)//2: break
    return np.array(Xb), yb

print("Строим train/test наборы...")
X_tr, y_tr = build_set(events_train, 1800, SPLIT-1)
X_te, y_te = build_set(events_test,  SPLIT, 2024)

all_labels = list(y_tr) + list(y_te)
le2 = LabelEncoder(); le2.fit(all_labels)
y_tr_enc = le2.transform(y_tr)
y_te_enc = le2.transform(y_te)

model_wf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, random_state=42)
model_wf.fit(X_tr, y_tr_enc)
y_pred = model_wf.predict(X_te)
f1_wf = f1_score(y_te_enc, y_pred, average='weighted', zero_division=0)

print(f"\nWALK-FORWARD (train<{SPLIT}, test {SPLIT}-2024):")
print(f"  F1 = {f1_wf:.3f}")
print(classification_report(y_te_enc, y_pred,
                            target_names=le2.classes_, zero_division=0))

# Cross-val для сравнения
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, random_state=42)
scores = cross_val_score(base, X, y_enc, cv=cv, scoring='f1_weighted')
print(f"Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Разрыв (overfit gap): {scores.mean()-f1_wf:.3f}")

# Финальная модель
model_final = CalibratedClassifierCV(base, cv=3, method='isotonic')
model_final.fit(X, y_enc)
classes_final = list(le.classes_)

# Прогноз
print(f"\nФИНАЛЬНЫЙ ПРОГНОЗ ({n_feat} признаков · {len(events)} событий):")
peaks = [
    ("2026-05-17","сегодня"),
    ("2026-08-29","пик август"),
    ("2026-10-28","октябрь"),
    ("2027-03-21","март 2027"),
    ("2027-09-23","осень 2027"),
    ("2027-12-22","декабрь 2027"),
    ("2028-03-19","март 2028"),
    ("2028-12-16","конец горизонта"),
]
results = {}
for date, label in peaks:
    proba = model_final.predict_proba([features(date)])[0]
    pairs = sorted(zip(classes_final,proba), key=lambda x:-x[1])
    print(f"\n{date} ({label}):")
    for c,p in pairs:
        if p>0.06: print(f"  {c:<22} {p:.1%}  {'▓'*int(p*28)}")
    results[date] = {c:round(float(p),4) for c,p in zip(classes_final,proba)}

with open("big_forecast.json","w") as f:
    json.dump({
        "n_events": len(events),
        "n_features": int(n_feat),
        "f1_walkforward": round(float(f1_wf),3),
        "f1_crossval": round(float(scores.mean()),3),
        "overfit_gap": round(float(scores.mean()-f1_wf),3),
        "split_year": SPLIT,
        "forecast": results
    }, f, indent=2)

print(f"\n✓ big_forecast.json")
print(f"Walk-forward F1 = {f1_wf:.3f} · Cross-val = {scores.mean():.3f}")
