import json, numpy as np, math
from datetime import datetime, timedelta
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("RETRAIN — расширенный датасет + Уран + Нептун\n")

with open("gdelt_events.json") as f:
    events = json.load(f)

with open("quarterly_gravity.json") as f:
    grav_data = json.load(f)

def get_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer()
    obs.date = dt.strftime("%Y/%m/%d")

    # 9 тел вместо 7 — добавляем Уран и Нептун
    bodies = [ephem.Sun(), ephem.Moon(), ephem.Mercury(), ephem.Venus(),
              ephem.Mars(), ephem.Jupiter(), ephem.Saturn(),
              ephem.Uranus(), ephem.Neptune()]
    feats = []
    for b in bodies:
        b.compute(obs)
        lon = float(b.hlong)
        feats.extend([math.sin(lon), math.cos(lon)])

    # Ключевые аспекты
    jup = ephem.Jupiter(); sat = ephem.Saturn()
    ura = ephem.Uranus(); nep = ephem.Neptune()
    for b in [jup, sat, ura, nep]:
        b.compute(obs)

    pairs = [(jup, sat), (jup, ura), (sat, nep), (ura, nep)]
    for a, b in pairs:
        asp = abs(math.degrees(float(a.hlong) - float(b.hlong))) % 360
        feats += [math.sin(math.radians(asp)), math.cos(math.radians(asp))]

    # Лунные узлы (приближение через эклиптику)
    moon = ephem.Moon(); moon.compute(obs)
    node_lon = float(moon.hlong) + math.pi  # упрощённо
    feats += [math.sin(node_lon), math.cos(node_lon)]

    # Гравитация
    best = min(grav_data, key=lambda g:
        abs((datetime.strptime(g["date"][:10], "%Y-%m-%d") - dt).days))
    feats += [
        best.get("grav_potential_log", 0),
        best.get("angular_momentum_log", 0),
        best.get("tidal_force", 0),
    ]
    return feats

import random; random.seed(42)
X, y = [], []
le = LabelEncoder()

print("Вычисляем признаки для событий...")
for e in events:
    try:
        X.append(get_features(e["date"]))
        y.append(e["category"])
    except:
        pass

# Нейтральные дни — 2x от событий
print("Добавляем нейтральные дни...")
start = datetime(1979, 1, 1)
event_dates = [datetime.strptime(e["date"], "%Y-%m-%d") for e in events]
neutral_added = 0
attempts = 0
while neutral_added < len(events) * 2 and attempts < 5000:
    attempts += 1
    d = start + timedelta(days=random.randint(0, 16436))
    if not any(abs((d - ed).days) < 45 for ed in event_dates):
        try:
            X.append(get_features(d.strftime("%Y-%m-%d")))
            y.append("neutral")
            neutral_added += 1
        except:
            pass

print(f"Датасет: {len(y)} точек ({len(events)} событий + {neutral_added} нейтральных)")

y_enc = le.fit_transform(y)
X = np.array(X)
classes = le.classes_

print(f"Классы: {list(classes)}")
print(f"Признаков: {X.shape[1]}")

# Кросс-валидация
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, random_state=42)
scores = cross_val_score(base, X, y_enc, cv=cv, scoring='f1_weighted')
print(f"\nF1-weighted (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

# Финальное обучение
model = CalibratedClassifierCV(base, cv=3, method='isotonic')
model.fit(X, y_enc)

# Прогноз помесячно 2026-2028
print("\nПОМЕСЯЧНЫЙ ПРОГНОЗ 2026-2028:")
print(f"{'Дата':<12} {'Событие':<22} {'Вер-ть':>7}  Сигнал")

forecast = []
dt = datetime(2026, 5, 1)
while dt <= datetime(2028, 12, 31):
    ds = dt.strftime("%Y-%m-%d")
    try:
        proba = model.predict_proba([get_features(ds)])[0]
        neutral_p = proba[list(classes).index("neutral")]
        non_neutral = [(p, c) for p, c in zip(proba, classes) if c != "neutral"]
        top_p, top_c = max(non_neutral, key=lambda x: x[0])
        forecast.append({
            "date": ds, "neutral": round(float(neutral_p), 4),
            "top_class": top_c, "top_prob": round(float(top_p), 4),
            "all": {c: round(float(p), 4) for c, p in zip(classes, proba)}
        })
        if top_p > 0.10 or neutral_p < 0.70:
            bar = "▓" * int(top_p * 35)
            flag = " ⚠" if top_p > 0.25 else ""
            print(f"{ds:<12} {top_c:<22} {top_p:>6.1%}  {bar}{flag}")
    except Exception as ex:
        print(f"{ds} ошибка: {ex}")
    dt += timedelta(days=30)

with open("forecast_final.json", "w") as f:
    json.dump(forecast, f, indent=2)

# График
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor="#07080F")
colors_map = {
    "geopolitical":    "#EF4444",
    "economic_crisis": "#EF9F27",
    "natural_disaster":"#3B82F6",
    "epidemic":        "#A855F7",
    "tech_breakthrough":"#27EFB5",
    "neutral":         "#1f2937"
}

dates_plot = [datetime.strptime(f["date"], "%Y-%m-%d") for f in forecast]
neutral_vals = [f["neutral"] for f in forecast]

ax1.set_facecolor("#0d0d1a")
ax1.fill_between(dates_plot, neutral_vals, alpha=0.25, color="#6B5BDB")
ax1.plot(dates_plot, neutral_vals, color="#6B5BDB", lw=2, label="P(neutral)")
ax1.axhline(0.70, color="#EF9F27", lw=0.8, linestyle="--", alpha=0.6, label="порог 70%")
ax1.set_ylim(0, 1)
ax1.set_ylabel("P(neutral)", color="#aaa")
ax1.set_title("Калиброванный прогноз 2026–2028  (9 планет + гравитация)", color="white", fontsize=13)
ax1.tick_params(colors="#888"); ax1.legend(facecolor="#1a1a2e", labelcolor="white")
for sp in ax1.spines.values(): sp.set_edgecolor("#333355")

bottom = np.zeros(len(forecast))
for cls in [c for c in classes if c != "neutral"]:
    vals = np.array([f["all"].get(cls, 0) for f in forecast])
    ax2.bar(dates_plot, vals, bottom=bottom, width=25,
            color=colors_map.get(cls, "#888"), alpha=0.85, label=cls)
    bottom += vals

ax2.set_facecolor("#0d0d1a")
ax2.set_ylim(0, 0.6)
ax2.set_ylabel("P(event)", color="#aaa")
ax2.tick_params(colors="#888")
ax2.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("forecast_final.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ forecast_final.png")
print(f"✓ forecast_final.json")
print(f"\nF1: {scores.mean():.3f} — {'улучшение vs предыдущей модели' if scores.mean() > 0.45 else 'стабильно'}")
