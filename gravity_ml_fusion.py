import json, pickle, numpy as np
from datetime import datetime, timedelta
import ephem, math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("GRAVITY + ML FUSION")
print("Добавляем физику в предиктор событий\n")

# Загружаем гравитационные данные
with open("quarterly_gravity.json") as f:
    grav_data = json.load(f)

print(f"Гравитационных точек: {len(grav_data)}")

# Исторические события (из предыдущей модели)
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
    ("1995-04-01","tech_breakthrough"),  # натальная дата — контрольная точка
    ("1997-07-02","economic_crisis"),("1998-08-17","economic_crisis"),
    ("2000-01-01","tech_breakthrough"),("2001-09-11","geopolitical"),
    ("2001-12-02","economic_crisis"),("2003-03-20","geopolitical"),
    ("2004-12-26","natural_disaster"),("2005-08-29","natural_disaster"),
    ("2007-06-01","economic_crisis"),("2008-09-15","economic_crisis"),
    ("2010-01-12","natural_disaster"),("2010-04-20","natural_disaster"),
    ("2011-03-11","natural_disaster"),("2011-03-15","natural_disaster"),
    ("2013-04-15","geopolitical"),("2014-03-18","geopolitical"),
    ("2015-09-28","geopolitical"),("2016-06-23","geopolitical"),
    ("2017-08-25","natural_disaster"),("2018-11-08","natural_disaster"),
    ("2019-12-31","epidemic"),("2020-03-11","epidemic"),
    ("2020-03-20","economic_crisis"),("2021-01-06","geopolitical"),
    ("2021-11-26","epidemic"),("2022-02-24","geopolitical"),
    ("2022-05-09","economic_crisis"),("2023-02-06","natural_disaster"),
    ("2023-10-07","geopolitical"),("2024-01-01","tech_breakthrough"),
]

def get_planet_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer()
    obs.date = dt.strftime("%Y/%m/%d")
    planets = [ephem.Sun(), ephem.Moon(), ephem.Mercury(), ephem.Venus(),
               ephem.Mars(), ephem.Jupiter(), ephem.Saturn()]
    features = []
    for p in planets:
        p.compute(obs)
        features.extend([float(p.hlong), math.sin(float(p.hlong)),
                         math.cos(float(p.hlong))])
    # Аспекты Юпитер-Сатурн
    jup = ephem.Jupiter(); sat = ephem.Saturn()
    jup.compute(obs); sat.compute(obs)
    aspect = abs(math.degrees(float(jup.hlong) - float(sat.hlong))) % 360
    features.append(math.sin(math.radians(aspect)))
    features.append(math.cos(math.radians(aspect)))
    return features

# Индексируем гравитационные данные по дате
grav_index = {}
for g in grav_data:
    grav_index[g["date"][:7]] = g  # по YYYY-MM

def get_grav_features(date_str):
    ym = date_str[:7]
    # Ищем ближайший квартал
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    best = None
    best_diff = 999
    for g in grav_data:
        gdt = datetime.strptime(g["date"][:10], "%Y-%m-%d")
        diff = abs((gdt - dt).days)
        if diff < best_diff:
            best_diff = diff
            best = g
    if best:
        return [
            best.get("grav_potential_log", 0),
            best.get("grav_force_mag", 0),
            best.get("angular_momentum_log", 0),
            best.get("tidal_force", 0),
        ]
    return [0, 0, 0, 0]

print("Строим датасет с гравитационными признаками...")

X_planets, X_fusion, y = [], [], []
le = LabelEncoder()

for date, event_type in EVENTS:
    pf = get_planet_features(date)
    gf = get_grav_features(date)
    X_planets.append(pf)
    X_fusion.append(pf + gf)
    y.append(event_type)

# Нейтральные дни
import random
random.seed(42)
neutral_dates = []
start = datetime(1929, 1, 1)
for _ in range(300):
    days = random.randint(0, 34675)
    d = start + timedelta(days=days)
    if not any(abs((datetime.strptime(e[0],"%Y-%m-%d")-d).days) < 30 for e in EVENTS):
        neutral_dates.append(d.strftime("%Y-%m-%d"))

for date in neutral_dates[:200]:
    pf = get_planet_features(date)
    gf = get_grav_features(date)
    X_planets.append(pf)
    X_fusion.append(pf + gf)
    y.append("neutral")

y_enc = le.fit_transform(y)

print(f"Датасет: {len(y)} точек, классы: {list(le.classes_)}")

# Сравниваем модели
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_planets = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf_fusion  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
gb_fusion  = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)

scores_p = cross_val_score(rf_planets, X_planets, y_enc, cv=cv, scoring='f1_weighted')
scores_f = cross_val_score(rf_fusion,  X_fusion,  y_enc, cv=cv, scoring='f1_weighted')
scores_g = cross_val_score(gb_fusion,  X_fusion,  y_enc, cv=cv, scoring='f1_weighted')

print(f"\nРезультаты (F1-weighted, 5-fold CV):")
print(f"  Только планеты:          {scores_p.mean():.3f} ± {scores_p.std():.3f}")
print(f"  Планеты + гравитация:    {scores_f.mean():.3f} ± {scores_f.std():.3f}  {'✓ улучшение' if scores_f.mean() > scores_p.mean() else '↓'}")
print(f"  GradientBoosting fusion: {scores_g.mean():.3f} ± {scores_g.std():.3f}")

# Обучаем финальную модель
rf_fusion.fit(X_fusion, y_enc)

# Feature importance
feature_names = (
    [f"{p}_{t}" for p in ["Sun","Moon","Mer","Ven","Mar","Jup","Sat"]
     for t in ["lon","sin","cos"]]
    + ["JS_sin","JS_cos"]
    + ["grav_pot","grav_force","ang_mom","tidal"]
)

importances = rf_fusion.feature_importances_
top_idx = np.argsort(importances)[::-1][:12]

print(f"\nТоп-12 признаков:")
for i in top_idx:
    bar = "█" * int(importances[i] * 200)
    print(f"  {feature_names[i]:20s} {importances[i]:.4f}  {bar}")

# Прогноз на ближайшие пики (из full_gravity результатов)
peak_dates = ["2026-10-18","2027-04-04","2027-03-21","2028-03-19","2027-10-03"]
classes = le.classes_

print(f"\nПРОГНОЗ ДЛЯ ГРАВИТАЦИОННЫХ ПИКОВ:")
predictions = {}
for date in peak_dates:
    pf = get_planet_features(date)
    gf = get_grav_features(date)
    proba = rf_fusion.predict_proba([pf + gf])[0]
    top_class = classes[np.argmax(proba)]
    top_prob = np.max(proba)
    predictions[date] = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    print(f"\n  {date}:")
    for c, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar = "▓" * int(p * 30)
        print(f"    {c:20s} {p:.1%}  {bar}")

# Сохраняем
with open("gravity_ml_fusion.json","w") as f:
    json.dump({
        "scores": {
            "planets_only": round(float(scores_p.mean()),3),
            "fusion": round(float(scores_f.mean()),3),
            "gradient_boost": round(float(scores_g.mean()),3),
        },
        "peak_predictions": predictions,
        "top_features": [feature_names[i] for i in top_idx],
    }, f, indent=2)

# График
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#07080F")

# Сравнение моделей
ax1 = axes[0]
ax1.set_facecolor("#0d0d1a")
models = ["Только\nпланеты", "Планеты +\nгравитация", "Gradient\nBoosting"]
scores = [scores_p.mean(), scores_f.mean(), scores_g.mean()]
errors = [scores_p.std(), scores_f.std(), scores_g.std()]
colors = ["#6B5BDB", "#27EFB5", "#EF9F27"]
bars = ax1.bar(models, scores, color=colors, alpha=0.8, yerr=errors, capsize=5)
ax1.set_title("Сравнение моделей (F1-score)", color="white")
ax1.set_ylabel("F1-weighted", color="#888")
ax1.tick_params(colors="#888")
ax1.set_ylim(0, 1)
for sp in ax1.spines.values(): sp.set_edgecolor("#333355")
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{score:.3f}", ha="center", color="white", fontsize=11, fontweight="bold")

# Feature importance
ax2 = axes[1]
ax2.set_facecolor("#0d0d1a")
top_names = [feature_names[i] for i in top_idx[:10]]
top_vals  = [importances[i] for i in top_idx[:10]]
colors2 = ["#EF9F27" if "grav" in n or "ang" in n or "tidal" in n else "#27EFB5"
           for n in top_names]
ax2.barh(range(10), top_vals[::-1], color=colors2[::-1], alpha=0.8)
ax2.set_yticks(range(10))
ax2.set_yticklabels(top_names[::-1], color="#ccc", fontsize=9)
ax2.set_title("Feature Importance (оранжевый = гравитация)", color="white")
ax2.tick_params(colors="#888")
for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("gravity_ml_fusion.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ gravity_ml_fusion.png")
print(f"✓ gravity_ml_fusion.json")
