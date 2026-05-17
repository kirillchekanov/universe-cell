import urllib.request, json, math, numpy as np, random
from datetime import datetime, timedelta
from collections import defaultdict
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("KP-ИНДЕКС — реальные геомагнитные данные NOAA\n")

# NOAA исторические Kp данные по годам
# Формат: https://www.ngdc.noaa.gov/stp/GEOMAG/kp_ap.html
# Используем GFZ Potsdam — официальный архив Kp с 1932

def fetch_kp_year(year):
    url = f"https://www.gfz-potsdam.de/fileadmin/gfz/sec23/Kp-ap-Ap-SN-F107_since_1932/Kp_ap_Ap_SN_F107_since_1932.txt"
    return None  # большой файл — качаем иначе

# NOAA FTP архив — помесячные файлы
def fetch_kp_monthly(year, month):
    # WDC Boulder — публичный архив
    url = f"https://services.swpc.noaa.gov/text/daily-geomagnetic-indices.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore")
    except:
        return None

# Качаем текущие + исторические через NOAA API
print("Загружаем Kp данные через NOAA API...")

# 1. Последние 30 дней — точные данные
kp_daily = {}

try:
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    # Агрегируем по дням
    day_kp = defaultdict(list)
    for point in data:
        day = point["time_tag"][:10]
        day_kp[day].append(float(point["estimated_kp"]))
    for day, vals in day_kp.items():
        kp_daily[day] = max(vals)  # дневной максимум
    print(f"  Текущие данные: {len(kp_daily)} дней")
except Exception as e:
    print(f"  Ошибка: {e}")

# 2. Исторические Kp через NOAA 45-day files
try:
    url = "https://services.swpc.noaa.gov/text/daily-geomagnetic-indices.txt"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode("utf-8", errors="ignore")
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#") or len(line) < 10: continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                kp_sum = float(parts[7]) if parts[7] != '-1' else 0
                date_str = f"{year:04d}-{month:02d}-{day:02d}"
                kp_daily[date_str] = kp_sum / 8.0  # нормализуем (макс ~8*9=72)
            except: pass
    print(f"  После 45-day файла: {len(kp_daily)} дней")
except Exception as e:
    print(f"  45-day файл: {e}")

# 3. Исторические экстремумы Kp — расширенный каталог
# Kp=9 (максимум) и Kp>=7 события из научной литературы
KP_STORMS = {
    # (дата: макс Kp за день)
    "1989-03-13": 9.0, "1989-03-14": 8.7, "1989-10-20": 8.3,
    "1991-03-24": 9.0, "1991-11-09": 9.0, "1994-04-17": 8.0,
    "2000-07-15": 9.0, "2000-07-16": 8.3, "2000-09-17": 8.3,
    "2001-03-31": 9.0, "2001-11-06": 8.7, "2003-10-29": 9.0,
    "2003-10-30": 9.0, "2003-11-20": 9.0, "2004-07-26": 8.0,
    "2004-11-08": 9.0, "2005-01-17": 7.3, "2005-05-15": 8.7,
    "2005-08-24": 8.0, "2006-12-14": 7.7, "2010-05-28": 7.3,
    "2011-08-05": 7.3, "2012-03-09": 7.7, "2012-07-15": 7.7,
    "2013-06-01": 7.3, "2015-03-17": 8.3, "2015-06-22": 8.0,
    "2017-09-07": 8.0, "2017-09-08": 7.7, "2019-08-31": 7.3,
    "2021-10-12": 7.3, "2022-02-03": 6.7, "2024-05-10": 9.0,
    "2024-05-11": 8.7, "2024-10-10": 9.0, "2024-10-11": 8.3,
}
for date, kp in KP_STORMS.items():
    if date not in kp_daily:
        kp_daily[date] = kp

print(f"  Итого точек в базе: {len(kp_daily)}")

def get_kp_features(date_str):
    """Kp признаки для даты: макс в окне ±15 дней"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    kp_vals = []
    for delta in range(-15, 16):
        d = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in kp_daily:
            kp_vals.append(kp_daily[d])
    if not kp_vals:
        return [0.0, 0.0, 0.0, 0.0]
    kp_max   = max(kp_vals) / 9.0          # нормализованный пик
    kp_mean  = np.mean(kp_vals) / 9.0      # средний уровень
    kp_storm = 1.0 if max(kp_vals) >= 7 else 0.5 if max(kp_vals) >= 5 else 0.0
    kp_trend = (kp_vals[-1] - kp_vals[0]) / 9.0 if len(kp_vals) > 1 else 0.0
    return [kp_max, kp_mean, kp_storm, kp_trend]

# Тест
print("\nТест Kp признаков:")
tests = [
    ("2003-10-29", "Halloween Kp=9"),
    ("2024-05-10", "May 2024 Kp=9"),
    ("2015-03-17", "St.Patrick Kp=8.3"),
    ("2020-03-11", "COVID — тихо"),
    ("2001-09-11", "9/11 — тихо"),
    ("2026-05-17", "сегодня"),
]
for date, label in tests:
    f = get_kp_features(date)
    print(f"  {date} ({label}): max={f[0]:.2f} mean={f[1]:.2f} storm={f[2]:.1f}")

# Полные признаки
with open("gdelt_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

STORMS_DST = [
    ("1989-03-13",-589),("1991-03-24",-298),("2000-07-15",-301),
    ("2001-03-31",-387),("2003-10-29",-383),("2003-11-20",-422),
    ("2004-11-08",-374),("2015-03-17",-222),("2024-05-10",-412),
    ("2024-10-10",-335),
]
def get_dst(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    min_dst = 0.0
    for sd, dv in STORMS_DST:
        diff = (dt-datetime.strptime(sd,"%Y-%m-%d")).days
        if abs(diff)<=30 and dv<min_dst: min_dst=dv
    return [min_dst/600.0, 1.0 if min_dst<-300 else 0.5 if min_dst<-100 else 0.0]

def all_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    bodies = [ephem.Sun(),ephem.Moon(),ephem.Mercury(),ephem.Venus(),
              ephem.Mars(),ephem.Jupiter(),ephem.Saturn(),
              ephem.Uranus(),ephem.Neptune()]
    f = []
    for b in bodies:
        b.compute(obs)
        f += [math.sin(float(b.hlong)), math.cos(float(b.hlong))]
    jup=ephem.Jupiter();sat=ephem.Saturn()
    ura=ephem.Uranus();nep=ephem.Neptune()
    for b in [jup,sat,ura,nep]: b.compute(obs)
    for a,b in [(jup,sat),(jup,ura),(sat,nep),(ura,nep)]:
        asp=abs(math.degrees(float(a.hlong)-float(b.hlong)))%360
        f+=[math.sin(math.radians(asp)),math.cos(math.radians(asp))]
    moon=ephem.Moon();moon.compute(obs)
    f+=[math.sin(float(moon.hlong)+math.pi),math.cos(float(moon.hlong)+math.pi)]
    best=min(grav_data,key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f+=[best.get("grav_potential_log",0),
        best.get("angular_momentum_log",0),
        best.get("tidal_force",0)]
    f+=get_dst(date_str)
    f+=get_kp_features(date_str)  # новое!
    return f

print("\nПЕРЕОБУЧАЕМ С Kp...")
random.seed(42)
X,y=[],[]
for e in events:
    try: X.append(all_features(e["date"])); y.append(e["category"])
    except: pass

start=datetime(1979,1,1)
edates=[datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
n=0
for _ in range(5000):
    d=start+timedelta(days=random.randint(0,16436))
    if not any(abs((d-ed).days)<45 for ed in edates):
        try: X.append(all_features(d.strftime("%Y-%m-%d"))); y.append("neutral"); n+=1
        except: pass
    if n>=len(events)*2: break

le=LabelEncoder(); y_enc=le.fit_transform(y)
X=np.array(X); classes=list(le.classes_)
print(f"Датасет: {len(y)} точек, {X.shape[1]} признаков")
print(f"  +4 признака Kp (было 34, стало {X.shape[1]})")

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
base=GradientBoostingClassifier(n_estimators=300,max_depth=4,
                                 learning_rate=0.05,random_state=42)
scores=cross_val_score(base,X,y_enc,cv=cv,scoring='f1_weighted')

print(f"\nF1 прогресс:")
print(f"  Только планеты:        0.45 (старт)")
print(f"  + гравитация:          0.569")
print(f"  + Dst каталог:         0.571")
print(f"  + Kp реальные данные:  {scores.mean():.3f} ± {scores.std():.3f}")

model=CalibratedClassifierCV(base,cv=3,method='isotonic')
model.fit(X,y_enc)

print("\nФИНАЛЬНЫЙ ПРОГНОЗ (планеты + гравитация + Dst + Kp):")
peaks=[
    ("2026-08-29","geopolitical пик"),
    ("2026-10-28","natural_disaster"),
    ("2027-09-23","economic"),
    ("2027-12-22","декабрь 2027"),
    ("2028-12-16","tech горизонт"),
]
results={}
for date,label in peaks:
    proba=model.predict_proba([all_features(date)])[0]
    pairs=sorted(zip(classes,proba),key=lambda x:-x[1])
    kp_f=get_kp_features(date)
    kp_note=f" [Kp≈{kp_f[0]*9:.1f}]" if kp_f[0]>0.3 else ""
    print(f"\n{date} ({label}){kp_note}:")
    for c,p in pairs:
        if p>0.04:
            print(f"  {c:<22} {p:.1%}  {'▓'*int(p*30)}")
    results[date]={c:round(float(p),4) for c,p in zip(classes,proba)}

with open("kp_forecast.json","w") as f:
    json.dump({"f1":round(float(scores.mean()),3),
               "features":int(X.shape[1]),
               "kp_points":len(kp_daily),
               "forecast":results},f,indent=2)
print(f"\n✓ kp_forecast.json")
print(f"✓ Kp точек в базе: {len(kp_daily)}")
