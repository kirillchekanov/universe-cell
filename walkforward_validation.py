import json, math, numpy as np, random
from datetime import datetime, timedelta
from collections import Counter
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

print("WALK-FORWARD ВАЛИДАЦИЯ")
print("Train: 1979-2010 | Test: 2011-2024")
print("="*50 + "\n")

with open("balanced_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

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
except: pass

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
            if d not in earthquakes or earthquakes[d]<m: earthquakes[d]=m
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
    f += [0.2, 0.0, 0.0]
    return f

# Разбиваем по времени
SPLIT_YEAR = 2011
events_train = [e for e in events if int(e["date"][:4]) < SPLIT_YEAR]
events_test  = [e for e in events if int(e["date"][:4]) >= SPLIT_YEAR]

print(f"Train: {len(events_train)} событий (до {SPLIT_YEAR})")
print(f"Test:  {len(events_test)} событий ({SPLIT_YEAR}–2024)\n")

# Нейтральные дни
random.seed(42)
def add_neutral(event_list, start_year, end_year, ratio=1):
    edates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in event_list]
    start = datetime(start_year,1,1)
    end   = datetime(end_year,12,31)
    neutral = []
    attempts = 0
    while len(neutral) < len(event_list)*ratio and attempts < 20000:
        attempts += 1
        days = (end-start).days
        d = start + timedelta(days=random.randint(0,days))
        if not any(abs((d-ed).days)<30 for ed in edates):
            try:
                neutral.append((d.strftime("%Y-%m-%d"), "neutral"))
            except: pass
    return neutral

neutral_train = add_neutral(events_train, 1979, SPLIT_YEAR-1)
neutral_test  = add_neutral(events_test,  SPLIT_YEAR, 2024)

print(f"Нейтральных train: {len(neutral_train)}")
print(f"Нейтральных test:  {len(neutral_test)}")

# Строим X, y
def build_dataset(event_list, neutral_list):
    X, y = [], []
    for e in event_list:
        try: X.append(features(e["date"])); y.append(e["category"])
        except: pass
    for date, cat in neutral_list:
        try: X.append(features(date)); y.append(cat)
        except: pass
    return np.array(X), y

print("\nВычисляем признаки train...")
X_train, y_train = build_dataset(events_train, neutral_train)
print(f"Train: {len(y_train)} точек")

print("Вычисляем признаки test...")
X_test, y_test = build_dataset(events_test, neutral_test)
print(f"Test:  {len(y_test)} точек")

# Общий энкодер
all_labels = y_train + y_test
le = LabelEncoder()
le.fit(all_labels)
classes = list(le.classes_)

y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

# Обучаем ТОЛЬКО на train
print("\nОбучаем на train (1979-2010)...")
model = GradientBoostingClassifier(n_estimators=400, max_depth=5,
                                    learning_rate=0.04, random_state=42)
model.fit(X_train, y_train_enc)

# Предсказываем на test (данные которые модель НЕ видела)
print("Предсказываем на test (2011-2024)...")
y_pred = model.predict(X_test)

f1_wf = f1_score(y_test_enc, y_pred, average='weighted')
f1_cv = 0.689  # из предыдущего шага

print(f"\n{'='*50}")
print(f"РЕЗУЛЬТАТЫ WALK-FORWARD ВАЛИДАЦИИ:")
print(f"{'='*50}")
print(f"  F1 cross-val (оптимистичный): {f1_cv:.3f}")
print(f"  F1 walk-forward (честный):    {f1_wf:.3f}")
print(f"  Разница (overfit):            {f1_cv-f1_wf:+.3f}")

if f1_wf > 0.55:
    verdict = "✓ ХОРОШО — модель обобщает"
elif f1_wf > 0.45:
    verdict = "~ УМЕРЕННО — есть переобучение но сигнал есть"
else:
    verdict = "✗ ПЕРЕОБУЧЕНИЕ — модель не обобщает"
print(f"  Вердикт: {verdict}")

print(f"\nДетальный отчёт по классам (test set):")
print(classification_report(y_test_enc, y_pred,
                            target_names=classes, zero_division=0))

# Дополнительно — permutation test
print("Permutation test (100 итераций)...")
f1_random = []
for _ in range(100):
    y_shuffled = y_test_enc.copy()
    np.random.shuffle(y_shuffled)
    f1_random.append(f1_score(y_test_enc, y_shuffled, average='weighted'))

p_value = np.mean(np.array(f1_random) >= f1_wf)
print(f"  F1 случайного классификатора: {np.mean(f1_random):.3f} ± {np.std(f1_random):.3f}")
print(f"  Наш F1: {f1_wf:.3f}")
print(f"  p-value: {p_value:.4f} {'✓ значимо' if p_value < 0.05 else '✗ незначимо'}")

# Анализ ошибок
print(f"\nАНАЛИЗ ОШИБОК (test set):")
errors = [(y_test[i], classes[y_pred[i]])
          for i in range(len(y_test)) if y_pred[i] != y_test_enc[i]
          and y_test[i] != "neutral"]
err_pairs = Counter(errors)
print("Чаще всего путает:")
for (true, pred), n in err_pairs.most_common(5):
    print(f"  {true:<22} → {pred:<22} ({n}x)")

# Сохраняем
result = {
    "f1_crossval": f1_cv,
    "f1_walkforward": round(float(f1_wf), 3),
    "overfit_gap": round(float(f1_cv - f1_wf), 3),
    "p_value": round(float(p_value), 4),
    "verdict": verdict,
    "train_events": len(events_train),
    "test_events": len(events_test),
    "train_period": f"1979-{SPLIT_YEAR-1}",
    "test_period": f"{SPLIT_YEAR}-2024",
}
with open("walkforward_results.json","w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ walkforward_results.json")
print(f"\nИТОГ: walk-forward F1 = {f1_wf:.3f} · p = {p_value:.4f}")
