import json, math, numpy as np, random
from datetime import datetime, timedelta
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter

print("RETRAIN — 1718 событий + 52 признака\n")

with open("balanced_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

print(f"События: {len(events)}")
cats = Counter(e["category"] for e in events)
for cat, n in cats.most_common():
    print(f"  {cat:<22} {n:>5}  {'█'*min(n//20,40)}")

# Загружаем данные
import urllib.request, csv, io
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
except Exception as e:
    print(f"Пятна: {e}")

earthquakes = {}
try:
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query"
           "?format=geojson&starttime=1950-01-01&endtime=2025-12-31"
           "&minmagnitude=6.5&orderby=time&limit=5000")
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    for feat in data.get("features",[]):
        props = feat.get("properties",{})
        t = props.get("time",0)
        if t:
            dt = datetime.fromtimestamp(t/1000)
            d = dt.strftime("%Y-%m-%d")
            m = props.get("mag",0)
            if d not in earthquakes or earthquakes[d] < m:
                earthquakes[d] = m
    print(f"Землетрясения: {len(earthquakes)} дней")
except Exception as e:
    print(f"USGS: {e}")

KP_STORMS = {"1989-03-13":9.0,"1991-03-24":9.0,"2000-07-15":9.0,
    "2001-03-31":9.0,"2003-10-29":9.0,"2003-11-20":9.0,
    "2015-03-17":8.3,"2024-05-10":9.0,"2024-10-10":9.0}
DST_STORMS = [("1989-03-13",-589),("2000-07-15",-301),
    ("2001-03-31",-387),("2003-10-29",-383),
    ("2015-03-17",-222),("2024-05-10",-412)]

def get_sn(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    for delta in range(32):
        d = (dt-timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in sunspots:
            sn = sunspots[d]
            return [sn/300.0, 1.0 if sn>150 else 0.0, 1.0 if sn<10 else 0.0]
    return [0.5,0.0,0.0]

def get_quake(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    mx = 0.0
    for delta in range(-7,8):
        d = (dt+timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in earthquakes: mx = max(mx, earthquakes[d])
    return [mx/9.5, 1.0 if mx>=8.0 else 0.5 if mx>=7.0 else 0.0]

def get_kp(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    mx = 0.0
    for sd,kp in KP_STORMS.items():
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days)<=15: mx=max(mx,kp)
    return [mx/9.0, 1.0 if mx>=7 else 0.0]

def get_dst(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    mn = 0.0
    for sd,dv in DST_STORMS:
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days)<=30 and dv<mn: mn=dv
    return [mn/600.0, 1.0 if mn<-300 else 0.0]

def get_lunar(ds):
    dt = datetime.strptime(ds,"%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    moon = ephem.Moon(); moon.compute(obs)
    sun  = ephem.Sun();  sun.compute(obs)
    dm = float(moon.earth_distance)
    ds2 = float(sun.earth_distance)
    mn = (dm-0.00247)/(0.00272-0.00247)
    sn = (ds2-0.983)/(1.017-0.983)
    ph = float(moon.phase)/100.0
    return [mn, sn, ph, 1.0 if mn<0.1 else 0.0, 1.0 if mn>0.9 else 0.0]

def features(date_str):
    dt = datetime.strptime(date_str,"%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    bodies = [ephem.Sun(),ephem.Moon(),ephem.Mercury(),ephem.Venus(),
              ephem.Mars(),ephem.Jupiter(),ephem.Saturn(),
              ephem.Uranus(),ephem.Neptune()]
    f = []
    for b in bodies:
        b.compute(obs)
        f += [math.sin(float(b.hlong)), math.cos(float(b.hlong))]
    jup=ephem.Jupiter(); sat=ephem.Saturn()
    ura=ephem.Uranus(); nep=ephem.Neptune()
    for b in [jup,sat,ura,nep]: b.compute(obs)
    for a,b in [(jup,sat),(jup,ura),(sat,nep),(ura,nep)]:
        asp = abs(math.degrees(float(a.hlong)-float(b.hlong)))%360
        f += [math.sin(math.radians(asp)), math.cos(math.radians(asp))]
    moon = ephem.Moon(); moon.compute(obs)
    f += [math.sin(float(moon.hlong)+math.pi), math.cos(float(moon.hlong)+math.pi)]
    best = min(grav_data, key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f += [best.get("grav_potential_log",0),
          best.get("angular_momentum_log",0),
          best.get("tidal_force",0)]
    f += get_dst(date_str); f += get_kp(date_str)
    f += get_sn(date_str);  f += get_quake(date_str)
    f += get_lunar(date_str)
    f += [0.2, 0.0, 0.0]  # vix placeholder
    return f

print("\nВычисляем признаки...")
random.seed(42)
X, y = [], []
errors = 0
for e in events:
    try:
        X.append(features(e["date"]))
        y.append(e["category"])
    except:
        errors += 1

print(f"Событий обработано: {len(y)} (ошибок: {errors})")

# Нейтральные — 1:1 к событиям (не 2:1 — данных уже много)
start = datetime(1900,1,1)
edates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
n = 0
for _ in range(20000):
    d = start + timedelta(days=random.randint(0,45000))
    if d.year < 1950: continue  # только после 1950
    if not any(abs((d-ed).days)<30 for ed in edates):
        try:
            X.append(features(d.strftime("%Y-%m-%d")))
            y.append("neutral")
            n += 1
        except: pass
    if n >= len(y)//2: break

le = LabelEncoder()
y_enc = le.fit_transform(y)
X = np.array(X)
classes = list(le.classes_)

print(f"Датасет: {len(y)} точек · {X.shape[1]} признаков")
print(f"Классы: {dict(Counter(y))}\n")

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = GradientBoostingClassifier(n_estimators=400, max_depth=5,
                                   learning_rate=0.04, random_state=42)
print("Кросс-валидация (может занять 3-5 мин)...")
scores = cross_val_score(base, X, y_enc, cv=cv, scoring='f1_weighted')

print(f"\nF1 ПРОГРЕСС ПРОЕКТА:")
print(f"  Планеты only:        0.450")
print(f"  + гравитация:        0.569")
print(f"  + геомагнетизм:      0.574")
print(f"  + все источники:     0.600")
print(f"  + 1718 событий:      {scores.mean():.3f} ± {scores.std():.3f}")
delta = scores.mean() - 0.600
print(f"  Прирост от данных:   {'+' if delta>=0 else ''}{delta:.3f}")

# Финальная модель
model = CalibratedClassifierCV(base, cv=3, method='isotonic')
model.fit(X, y_enc)

# Прогноз
print(f"\nФИНАЛЬНЫЙ ПРОГНОЗ (1718 событий):")
peaks = [
    ("2026-08-29","geopolitical пик"),
    ("2026-10-28","natural_disaster"),
    ("2027-09-23","economic"),
    ("2027-12-22","декабрь 2027"),
    ("2028-12-16","tech горизонт"),
]
results = {}
for date, label in peaks:
    proba = model.predict_proba([features(date)])[0]
    pairs = sorted(zip(classes,proba), key=lambda x: -x[1])
    print(f"\n{date} ({label}):")
    for c,p in pairs:
        if p > 0.05:
            print(f"  {c:<22} {p:.1%}  {'▓'*int(p*30)}")
    results[date] = {c:round(float(p),4) for c,p in zip(classes,proba)}

with open("balanced_forecast.json","w") as f:
    json.dump({
        "f1": round(float(scores.mean()),3),
        "f1_std": round(float(scores.std()),3),
        "n_events": len(events),
        "n_features": int(X.shape[1]),
        "forecast": results
    }, f, indent=2)

print(f"\n✓ balanced_forecast.json")
print(f"✓ Датасет: {len(events)} событий · F1: {scores.mean():.3f}")
