import urllib.request, json, math, numpy as np
from datetime import datetime, timedelta
import ephem

print("DST-ИНДЕКС — геомагнитные бури как признак\n")

# NOAA Dst через World Data Center for Geomagnetism, Kyoto
# Публичный API: https://wdc.kugi.kyoto-u.ac.jp
# Используем OMNI data через NASA CDAweb — бесплатно, без ключа

def fetch_dst_year(year):
    """Загружаем Dst по году через OMNI2 hourly data"""
    url = (f"https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
           f"?activity=retrieve&res=daily&spacecraft=omni2"
           f"&start_date={year}0101&end_date={year}1231"
           f"&vars=40&submit=Submit")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return None

# Пробуем загрузить — если недоступно, используем синтетические данные
print("Пробуем OMNI/NASA API...")
test = fetch_dst_year(2000)

if test and "YEAR" in test:
    print("✓ NASA OMNI доступен")
    USE_REAL = True
else:
    print("NASA OMNI недоступен — используем реальные Dst экстремумы из литературы")
    USE_REAL = False

# Известные экстремальные геомагнитные бури (Dst < -100 нТл)
# Источник: публичные каталоги WDC Kyoto
STORM_CATALOG = [
    # (дата, Dst_min нТл, класс бури)
    ("1989-03-13", -589, "extreme"),   # Quebec blackout
    ("1989-10-20", -268, "severe"),
    ("1991-03-24", -298, "severe"),
    ("1991-11-09", -354, "severe"),
    ("1994-04-17", -201, "severe"),
    ("2000-07-15", -301, "severe"),    # Bastille Day
    ("2000-09-17", -201, "severe"),
    ("2001-03-31", -387, "severe"),
    ("2001-11-06", -292, "severe"),
    ("2003-10-29", -383, "severe"),    # Halloween storms
    ("2003-10-30", -401, "severe"),
    ("2003-11-20", -422, "severe"),
    ("2004-07-26", -197, "severe"),
    ("2004-11-08", -374, "severe"),
    ("2005-01-17", -103, "strong"),
    ("2005-05-15", -247, "severe"),
    ("2005-08-24", -184, "severe"),
    ("2006-12-14", -146, "strong"),
    ("2010-05-28", -115, "strong"),
    ("2011-08-05", -113, "strong"),
    ("2012-03-09", -131, "strong"),
    ("2012-07-15", -139, "strong"),
    ("2013-06-01", -119, "strong"),
    ("2015-03-17", -222, "severe"),    # St. Patrick's Day
    ("2015-06-22", -204, "severe"),
    ("2017-09-07", -142, "strong"),
    ("2019-08-31", -106, "strong"),
    ("2021-10-12", -101, "strong"),
    ("2022-02-03", -96,  "moderate"),
    ("2024-05-10", -412, "extreme"),   # Strongest since 2003
    ("2024-10-10", -335, "severe"),
]

print(f"\nКаталог бурь: {len(STORM_CATALOG)} событий")
print(f"Диапазон Dst: {min(s[1] for s in STORM_CATALOG)} до {max(s[1] for s in STORM_CATALOG)} нТл")

# Строим функцию интерполяции Dst по дате
# Для дат вне каталога — фоновое значение ~0
storm_dict = {s[0]: s[1] for s in STORM_CATALOG}

def get_dst_feature(date_str):
    """Возвращает Dst и производные признаки для даты"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Ищем ближайшую бурю в окне ±30 дней
    min_dst = 0.0
    days_to_storm = 999
    storm_class = 0
    
    for storm_date, dst_val, storm_cls in STORM_CATALOG:
        sd = datetime.strptime(storm_date, "%Y-%m-%d")
        diff = (dt - sd).days
        if abs(diff) <= 30:
            if dst_val < min_dst:
                min_dst = dst_val
                days_to_storm = diff
                storm_class = {"moderate":1,"strong":2,"severe":3,"extreme":4}[storm_cls]
    
    # Нормализуем
    dst_norm = min_dst / 600.0  # делим на макс известное значение
    days_norm = days_to_storm / 30.0 if days_to_storm < 999 else 0
    
    return [dst_norm, days_norm, float(storm_class) / 4.0]

# Тестируем
print("\nТест признаков:")
test_dates = [
    ("2003-10-29", "Halloween storm"),
    ("2015-03-17", "St. Patrick's Day"),
    ("2024-05-10", "May 2024 extreme"),
    ("2020-03-11", "COVID — нет бури"),
    ("2001-09-11", "9/11 — нет бури"),
]
for date, label in test_dates:
    feats = get_dst_feature(date)
    print(f"  {date} ({label}): Dst_norm={feats[0]:.3f} days={feats[1]:.2f} class={feats[2]:.2f}")

# Интегрируем в модель
print("\nПЕРЕОБУЧАЕМ С DST...")

with open("gdelt_events.json") as f:
    events = json.load(f)
with open("quarterly_gravity.json") as f:
    grav_data = json.load(f)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import random

def get_all_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer()
    obs.date = dt.strftime("%Y/%m/%d")
    
    bodies = [ephem.Sun(), ephem.Moon(), ephem.Mercury(), ephem.Venus(),
              ephem.Mars(), ephem.Jupiter(), ephem.Saturn(),
              ephem.Uranus(), ephem.Neptune()]
    feats = []
    for b in bodies:
        b.compute(obs)
        lon = float(b.hlong)
        feats.extend([math.sin(lon), math.cos(lon)])
    
    jup=ephem.Jupiter(); sat=ephem.Saturn()
    ura=ephem.Uranus(); nep=ephem.Neptune()
    for b in [jup,sat,ura,nep]: b.compute(obs)
    for a,b in [(jup,sat),(jup,ura),(sat,nep),(ura,nep)]:
        asp = abs(math.degrees(float(a.hlong)-float(b.hlong))) % 360
        feats += [math.sin(math.radians(asp)), math.cos(math.radians(asp))]
    
    moon=ephem.Moon(); moon.compute(obs)
    node_lon = float(moon.hlong)+math.pi
    feats += [math.sin(node_lon), math.cos(node_lon)]
    
    best = min(grav_data, key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    feats += [best.get("grav_potential_log",0),
              best.get("angular_momentum_log",0),
              best.get("tidal_force",0)]
    
    # DST признаки — новое!
    feats += get_dst_feature(date_str)
    
    return feats

random.seed(42)
X, y = [], []
le = LabelEncoder()

for e in events:
    try:
        X.append(get_all_features(e["date"]))
        y.append(e["category"])
    except: pass

start = datetime(1979,1,1)
event_dates = [datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
neutral_added = 0
attempts = 0
while neutral_added < len(events)*2 and attempts < 5000:
    attempts += 1
    d = start + timedelta(days=random.randint(0,16436))
    if not any(abs((d-ed).days)<45 for ed in event_dates):
        try:
            X.append(get_all_features(d.strftime("%Y-%m-%d")))
            y.append("neutral")
            neutral_added += 1
        except: pass

y_enc = le.fit_transform(y)
X = np.array(X)

print(f"Датасет: {len(y)} точек, {X.shape[1]} признаков (было 31, стало {X.shape[1]})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.05, random_state=42)
scores_new = cross_val_score(base, X, y_enc, cv=cv, scoring='f1_weighted')

print(f"\nF1 без Dst (baseline): 0.569")
print(f"F1 с Dst:              {scores_new.mean():.3f} ± {scores_new.std():.3f}")
delta = scores_new.mean() - 0.569
sign = "+" if delta >= 0 else ""
print(f"Изменение:             {sign}{delta:.3f}  {'✓ улучшение' if delta > 0 else '↓ ухудшение'}")

# Обучаем финальную
model = CalibratedClassifierCV(base, cv=3, method='isotonic')
model.fit(X, le.fit_transform(y))
classes = le.classes_

# Прогноз на пики
print("\nПРОГНОЗ С DST (ключевые даты):")
peak_dates = [
    ("2026-08-29", "пик geopolitical"),
    ("2026-10-28", "пик natural_disaster"),
    ("2027-09-23", "economic сигнал"),
    ("2027-12-22", "декабрь 2027"),
    ("2028-12-16", "tech конец горизонта"),
]
for date, label in peak_dates:
    proba = model.predict_proba([get_all_features(date)])[0]
    neutral_p = proba[list(classes).index("neutral")]
    non_neutral = [(p,c) for p,c in zip(proba,classes) if c!="neutral"]
    top_p, top_c = max(non_neutral, key=lambda x: x[0])
    dst_f = get_dst_feature(date)
    dst_note = f" [буря Dst≈{dst_f[0]*600:.0f}нТл]" if dst_f[0] < -0.1 else ""
    print(f"\n  {date} ({label}){dst_note}:")
    for p,c in sorted(zip(proba,classes),key=lambda x:-x[1]):
        if p > 0.03:
            bar = "▓"*int(p*30)
            print(f"    {c:<22} {p:.1%}  {bar}")

# Сохраняем результат
result = {
    "f1_baseline": 0.569,
    "f1_with_dst": round(float(scores_new.mean()), 3),
    "improvement": round(float(scores_new.mean()-0.569), 3),
    "storm_catalog_size": len(STORM_CATALOG),
    "feature_count": int(X.shape[1]),
}
with open("dst_results.json","w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ dst_results.json")
print(f"Признаков в модели: {X.shape[1]} (планеты + гравитация + Dst)")
