import json, numpy as np
from datetime import datetime, timedelta
import ephem, math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("КАЛИБРОВАННЫЙ ПРОГНОЗ — финальная модель\n")

with open("quarterly_gravity.json") as f:
    grav_data = json.load(f)

EVENTS = [
    ("1929-10-29","economic_crisis"),("1933-03-04","geopolitical"),
    ("1939-09-01","geopolitical"),("1941-12-07","geopolitical"),
    ("1945-08-06","geopolitical"),("1950-06-25","geopolitical"),
    ("1957-10-04","tech_breakthrough"),("1962-10-16","geopolitical"),
    ("1968-04-04","geopolitical"),("1969-07-20","tech_breakthrough"),
    ("1973-10-17","economic_crisis"),("1979-03-28","natural_disaster"),
    ("1980-05-18","natural_disaster"),("1986-01-28","tech_breakthrough"),
    ("1986-04-26","natural_disaster"),("1987-10-19","economic_crisis"),
    ("1989-11-09","geopolitical"),("1991-01-17","geopolitical"),
    ("1991-12-25","geopolitical"),("1994-01-17","natural_disaster"),
    ("1997-07-02","economic_crisis"),("1998-08-17","economic_crisis"),
    ("2000-01-01","tech_breakthrough"),("2001-09-11","geopolitical"),
    ("2001-12-02","economic_crisis"),("2003-03-20","geopolitical"),
    ("2004-12-26","natural_disaster"),("2005-08-29","natural_disaster"),
    ("2007-06-01","economic_crisis"),("2008-09-15","economic_crisis"),
    ("2010-01-12","natural_disaster"),("2010-04-20","natural_disaster"),
    ("2011-03-11","natural_disaster"),("2013-04-15","geopolitical"),
    ("2014-03-18","geopolitical"),("2015-09-28","geopolitical"),
    ("2016-06-23","geopolitical"),("2017-08-25","natural_disaster"),
    ("2018-11-08","natural_disaster"),("2019-12-31","epidemic"),
    ("2020-03-11","epidemic"),("2020-03-20","economic_crisis"),
    ("2021-01-06","geopolitical"),("2021-11-26","epidemic"),
    ("2022-02-24","geopolitical"),("2022-05-09","economic_crisis"),
    ("2023-02-06","natural_disaster"),("2023-10-07","geopolitical"),
    ("2024-01-01","tech_breakthrough"),
]

def get_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer()
    obs.date = dt.strftime("%Y/%m/%d")
    planets = [ephem.Sun(), ephem.Moon(), ephem.Mercury(), ephem.Venus(),
               ephem.Mars(), ephem.Jupiter(), ephem.Saturn()]
    feats = []
    for p in planets:
        p.compute(obs)
        feats.extend([math.sin(float(p.hlong)), math.cos(float(p.hlong))])
    jup = ephem.Jupiter(); sat = ephem.Saturn()
    jup.compute(obs); sat.compute(obs)
    asp = abs(math.degrees(float(jup.hlong)-float(sat.hlong))) % 360
    feats += [math.sin(math.radians(asp)), math.cos(math.radians(asp))]
    # Гравитация
    best = min(grav_data, key=lambda g:
               abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    feats += [best.get("grav_potential_log",0), best.get("angular_momentum_log",0),
              best.get("tidal_force",0)]
    return feats

import random; random.seed(42)
X, y = [], []
le = LabelEncoder()

for date, label in EVENTS:
    X.append(get_features(date)); y.append(label)

start = datetime(1929,1,1)
for _ in range(400):
    d = start + timedelta(days=random.randint(0,34675))
    ds = d.strftime("%Y-%m-%d")
    if not any(abs((datetime.strptime(e[0],"%Y-%m-%d")-d).days)<30 for e in EVENTS):
        X.append(get_features(ds)); y.append("neutral")

y_enc = le.fit_transform(y)
X = np.array(X)

# SMOTE балансировка
try:
    sm = SMOTE(random_state=42, k_neighbors=1)
    X_bal, y_bal = sm.fit_resample(X, y_enc)
    print(f"SMOTE: {len(y)} → {len(y_bal)} точек")
except Exception as e:
    X_bal, y_bal = X, y_enc
    print(f"SMOTE пропущен: {e}")

# Калиброванная модель
base = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, random_state=42)
model = CalibratedClassifierCV(base, cv=3, method="isotonic")
model.fit(X_bal, y_bal)

classes = le.classes_

# Прогноз на 2026-2028 помесячно
print("\nПОМЕСЯЧНЫЙ ПРОГНОЗ 2026-2028:")
print(f"{'Дата':<12} {'Топ-событие':<22} {'Вероятность':>10}  {'Сигнал'}")

forecast = []
dt = datetime(2026, 5, 1)
end = datetime(2028, 12, 31)
while dt <= end:
    ds = dt.strftime("%Y-%m-%d")
    try:
        feats = get_features(ds)
        proba = model.predict_proba([feats])[0]
        non_neutral = [(p, c) for p, c in zip(proba, classes) if c != "neutral"]
        top_p, top_c = max(non_neutral, key=lambda x: x[0])
        neutral_p = proba[list(classes).index("neutral")]
        forecast.append({
            "date": ds, "neutral": round(float(neutral_p),4),
            "top_class": top_c, "top_prob": round(float(top_p),4),
            "all": {c: round(float(p),4) for c,p in zip(classes,proba)}
        })
        if top_p > 0.08 or neutral_p < 0.75:
            bar = "▓" * int(top_p * 40)
            print(f"{ds:<12} {top_c:<22} {top_p:>9.1%}  {bar}")
    except: pass
    dt += timedelta(days=30)

# Сохраняем
with open("forecast_2026_2028.json","w") as f:
    json.dump(forecast, f, indent=2)

# Финальный график
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor="#07080F")
colors_map = {"geopolitical":"#EF4444","economic_crisis":"#EF9F27",
              "natural_disaster":"#3B82F6","epidemic":"#A855F7",
              "tech_breakthrough":"#27EFB5","neutral":"#1f2937"}

dates_plot = [datetime.strptime(f["date"],"%Y-%m-%d") for f in forecast]
neutral_vals = [f["neutral"] for f in forecast]

ax1.set_facecolor("#0d0d1a")
ax1.fill_between(dates_plot, neutral_vals, alpha=0.3, color="#6B5BDB")
ax1.plot(dates_plot, neutral_vals, color="#6B5BDB", lw=1.5, label="P(neutral)")
ax1.axhline(0.75, color="#888", lw=0.8, linestyle="--", alpha=0.5)
ax1.set_ylim(0,1); ax1.set_ylabel("P(neutral)", color="#888")
ax1.set_title("Калиброванный прогноз 2026–2028", color="white", fontsize=13)
ax1.tick_params(colors="#888")
for sp in ax1.spines.values(): sp.set_edgecolor("#333355")

# Нижний — стакан вероятностей
bottom = np.zeros(len(forecast))
for cls in [c for c in classes if c != "neutral"]:
    vals = np.array([f["all"].get(cls,0) for f in forecast])
    ax2.bar(dates_plot, vals, bottom=bottom, width=25,
            color=colors_map.get(cls,"#888"), alpha=0.85, label=cls)
    bottom += vals

ax2.set_facecolor("#0d0d1a")
ax2.set_ylim(0,0.6); ax2.set_ylabel("P(event)", color="#888")
ax2.tick_params(colors="#888")
ax2.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("forecast_2026_2028.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ forecast_2026_2028.png")
print(f"✓ forecast_2026_2028.json")
