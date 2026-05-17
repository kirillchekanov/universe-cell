import json, math, numpy as np, random
from datetime import datetime, timedelta
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("DST + финальный прогноз\n")

with open("gdelt_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

STORMS = [
    ("1989-03-13",-589),("1991-03-24",-298),("1991-11-09",-354),
    ("2000-07-15",-301),("2001-03-31",-387),("2001-11-06",-292),
    ("2003-10-29",-383),("2003-10-30",-401),("2003-11-20",-422),
    ("2004-11-08",-374),("2005-05-15",-247),("2005-08-24",-184),
    ("2010-05-28",-115),("2011-08-05",-113),("2012-03-09",-131),
    ("2012-07-15",-139),("2015-03-17",-222),("2015-06-22",-204),
    ("2017-09-07",-142),("2024-05-10",-412),("2024-10-10",-335),
]

def get_dst(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    min_dst, days_to = 0.0, 999
    for sd, dv in STORMS:
        diff = (dt - datetime.strptime(sd, "%Y-%m-%d")).days
        if abs(diff) <= 30 and dv < min_dst:
            min_dst = dv; days_to = diff
    return [min_dst/600.0,
            days_to/30.0 if days_to < 999 else 0.0,
            1.0 if min_dst<-300 else 0.5 if min_dst<-100 else 0.0]

def features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
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
        asp = abs(math.degrees(float(a.hlong)-float(b.hlong))) % 360
        f += [math.sin(math.radians(asp)), math.cos(math.radians(asp))]
    moon=ephem.Moon(); moon.compute(obs)
    f += [math.sin(float(moon.hlong)+math.pi), math.cos(float(moon.hlong)+math.pi)]
    best = min(grav_data, key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f += [best.get("grav_potential_log",0),
          best.get("angular_momentum_log",0),
          best.get("tidal_force",0)]
    f += get_dst(date_str)
    return f

random.seed(42)
X, y = [], []
for e in events:
    try: X.append(features(e["date"])); y.append(e["category"])
    except: pass

start = datetime(1979,1,1)
edates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
n = 0
for _ in range(5000):
    d = start + timedelta(days=random.randint(0,16436))
    if not any(abs((d-ed).days)<45 for ed in edates):
        try: X.append(features(d.strftime("%Y-%m-%d"))); y.append("neutral"); n+=1
        except: pass
    if n >= len(events)*2: break

le = LabelEncoder()
y_enc = le.fit_transform(y)
X = np.array(X)
classes = list(le.classes_)
print(f"Датасет: {len(y)} точек, {X.shape[1]} признаков\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, random_state=42)
scores = cross_val_score(base, X, y_enc, cv=cv, scoring='f1_weighted')
print(f"F1 baseline (без Dst): 0.569")
print(f"F1 с Dst:              {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Изменение:             {'+' if scores.mean()>0.569 else ''}{scores.mean()-0.569:.3f}\n")

model = CalibratedClassifierCV(base, cv=3, method='isotonic')
model.fit(X, y_enc)

print("ПРОГНОЗ С DST (ключевые даты):")
peaks = [
    ("2026-08-29","geopolitical пик"),
    ("2026-10-28","natural_disaster"),
    ("2027-09-23","economic сигнал"),
    ("2027-12-22","декабрь 2027"),
    ("2028-12-16","tech горизонт"),
]
forecasts = {}
for date, label in peaks:
    proba = model.predict_proba([features(date)])[0]
    pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
    dst_f = get_dst(date)
    dst_note = f" [буря ~{int(dst_f[0]*600)}нТл]" if dst_f[0] < -0.05 else ""
    print(f"\n{date} ({label}){dst_note}:")
    for c, p in pairs:
        if p > 0.04:
            print(f"  {c:<22} {p:.1%}  {'▓'*int(p*30)}")
    forecasts[date] = {c: round(float(p),4) for c,p in zip(classes,proba)}

with open("dst_results.json","w") as f:
    json.dump({
        "f1_baseline": 0.569,
        "f1_with_dst": round(float(scores.mean()),3),
        "improvement": round(float(scores.mean()-0.569),3),
        "feature_count": int(X.shape[1]),
        "forecasts": forecasts,
    }, f, indent=2)

print(f"\n✓ dst_results.json сохранён")
