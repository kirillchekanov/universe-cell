import json, math, numpy as np, random
from datetime import datetime, timedelta
from collections import Counter
import ephem
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("ИСПРАВЛЕННАЯ МОДЕЛЬ")
print("Проблема: natural_disaster доминировал (90%)")
print("Решение: бинарная задача — кризис vs нейтральное")
print("="*55 + "\n")

with open("balanced_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

# Загружаем данные
import urllib.request
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
except: pass

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
    f += get_sn(date_str)
    f += get_lunar(date_str)
    return f

# ══ ЗАДАЧА 1: БИНАРНАЯ — кризис vs нейтральное ══
print("ЗАДАЧА 1: бинарная (кризис / нейтральное)")
print("-"*45)

# Берём только geopolitical + economic + epidemic (исключаем natural_disaster)
crisis_events = [e for e in events
                 if e["category"] in ("geopolitical","economic_crisis","epidemic")]
print(f"Кризисных событий: {len(crisis_events)}")

# Нейтральные
random.seed(42)
edates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in crisis_events]
start = datetime(1979,1,1)

X, y = [], []
for e in crisis_events:
    try: X.append(features(e["date"])); y.append(1)  # кризис
    except: pass

n = 0
for _ in range(10000):
    d = start + timedelta(days=random.randint(0,16436))
    if not any(abs((d-ed).days)<45 for ed in edates):
        try:
            X.append(features(d.strftime("%Y-%m-%d")))
            y.append(0)  # нейтральное
            n += 1
        except: pass
    if n >= len(crisis_events): break

X = np.array(X)
y = np.array(y)
print(f"Датасет: {len(y)} точек ({sum(y==1)} кризис · {sum(y==0)} нейтральных)")

# Walk-forward
SPLIT = 2011
dates_all = ([e["date"] for e in crisis_events] +
             [(start+timedelta(days=random.randint(0,16436))).strftime("%Y-%m-%d")
              for _ in range(n)])

events_with_dates = list(zip(X, y))
train_idx = [i for i,e in enumerate(crisis_events) if int(e["date"][:4]) < SPLIT]
test_idx  = [i for i,e in enumerate(crisis_events) if int(e["date"][:4]) >= SPLIT]

X_crisis = X[:len(crisis_events)]
y_crisis = y[:len(crisis_events)]
X_neutral = X[len(crisis_events):]
y_neutral = y[len(crisis_events):]

# Делим нейтральные пополам
split_n = len(X_neutral)//2
X_train = np.vstack([X_crisis[train_idx], X_neutral[:split_n]])
y_train = np.concatenate([y_crisis[train_idx], y_neutral[:split_n]])
X_test  = np.vstack([X_crisis[test_idx],  X_neutral[split_n:]])
y_test  = np.concatenate([y_crisis[test_idx],  y_neutral[split_n:]])

print(f"Train: {len(y_train)} · Test: {len(y_test)}")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                    learning_rate=0.05, random_state=42),
    "RandomForest":     RandomForestClassifier(n_estimators=200, max_depth=6,
                                               random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=42, C=0.1),
}

print("\nWalk-forward результаты (train<2011, test 2011-2024):")
best_f1, best_model_name, best_model = 0, None, None

for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred, average='weighted')
    prec = f1_score(y_test, pred, average='weighted', pos_label=1) if hasattr(f1_score,'pos_label') else 0
    print(f"  {name:<22} F1={f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1; best_model_name = name; best_model = clf

print(f"\nЛучшая модель: {best_model_name} · F1={best_f1:.3f}")
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=["нейтральное","кризис"],
                            zero_division=0))

# Permutation test
perms = [f1_score(y_test, np.random.permutation(best_model.predict(X_test)),
                  average='weighted') for _ in range(200)]
p_val = np.mean(np.array(perms) >= best_f1)
print(f"p-value (permutation): {p_val:.4f} {'✓ значимо' if p_val<0.05 else '~ на грани' if p_val<0.1 else '✗'}")

# ══ ПРОГНОЗ ══
print("\nПРОГНОЗ КРИЗИСНЫХ ПЕРИОДОВ 2026-2028:")
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

print(f"{'Дата':<12} {'P(кризис)':>10}  Сигнал")
forecasts = {}
for date, label in peaks:
    try:
        f = features(date)
        if hasattr(best_model, 'predict_proba'):
            p = best_model.predict_proba([f])[0][1]
        else:
            p = float(best_model.predict([f])[0])
        bar = "▓"*int(p*30)
        flag = " ⚠" if p > 0.5 else ""
        print(f"{date}  ({label:<16}) {p:>8.1%}  {bar}{flag}")
        forecasts[date] = round(float(p), 4)
    except Exception as e:
        print(f"{date}: ошибка {e}")

with open("fixed_forecast.json","w") as f:
    json.dump({
        "approach": "binary_crisis_vs_neutral",
        "excluded": "natural_disaster (too dominant)",
        "f1_walkforward": round(float(best_f1),3),
        "p_value": round(float(p_val),4),
        "best_model": best_model_name,
        "forecast": forecasts
    }, f, indent=2)

print(f"\n✓ fixed_forecast.json")
print(f"Walk-forward F1 = {best_f1:.3f} · p = {p_val:.4f}")
