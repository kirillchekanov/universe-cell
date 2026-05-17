import urllib.request, json, math, numpy as np, random, csv, io
from datetime import datetime, timedelta
from collections import defaultdict
import ephem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

print("MEGA FEATURES — все открытые источники\n")
print("="*55)

sources = {}

# ══ 1. СОЛНЕЧНЫЕ ПЯТНА (SIDC, с 1749) ══
print("\n[1/6] Солнечные пятна — SIDC Brussels...")
sunspots = {}
try:
    url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        for line in r.read().decode().split("\n"):
            p = line.strip().split()
            if len(p) >= 4:
                try:
                    yr, mo = int(p[0]), int(p[1])
                    sn = float(p[3])
                    for d in range(1, 32):
                        try:
                            sunspots[f"{yr:04d}-{mo:02d}-{d:02d}"] = sn
                        except: pass
                except: pass
    print(f"  ✓ {len(sunspots)} дней · диапазон SN: 0–{max(sunspots.values()):.0f}")
    sources["sunspots"] = len(sunspots)
except Exception as e:
    print(f"  ✗ {e}")

def get_sunspot(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    for delta in range(0, 32):
        d = (dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in sunspots:
            sn = sunspots[d]
            return [sn/300.0,                          # норм значение
                    1.0 if sn > 150 else 0.0,          # солнечный максимум
                    1.0 if sn < 10 else 0.0]           # солнечный минимум
    return [0.5, 0.0, 0.0]

# ══ 2. USGS СЕЙСМИКА (M≥6.5, с 1900) ══
print("\n[2/6] USGS землетрясения M≥6.5...")
earthquakes = {}
try:
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query"
           "?format=csv&starttime=1950-01-01&endtime=2025-12-31"
           "&minmagnitude=6.5&orderby=time&limit=5000")
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            date = row["time"][:10]
            mag = float(row["mag"])
            if date not in earthquakes or earthquakes[date] < mag:
                earthquakes[date] = mag
        except: pass
    print(f"  ✓ {len(earthquakes)} дней с M≥6.5 · макс: M{max(earthquakes.values()):.1f}")
    sources["earthquakes"] = len(earthquakes)
except Exception as e:
    print(f"  ✗ {e}")

def get_quake(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    max_mag = 0.0
    for delta in range(-7, 8):
        d = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in earthquakes:
            max_mag = max(max_mag, earthquakes[d])
    return [max_mag / 9.5,
            1.0 if max_mag >= 8.0 else 0.5 if max_mag >= 7.0 else 0.0]

# ══ 3. F10.7 SOLAR FLUX ══
print("\n[3/6] F10.7 солнечный радиопоток...")
f107 = {}
try:
    url = "https://lasp.colorado.edu/lisird/latis/dap/noaa_radio_flux.csv"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("utf-8", errors="ignore")
    for line in text.split("\n")[1:]:
        p = line.strip().split(",")
        if len(p) >= 2:
            try:
                date = p[0][:10]
                val = float(p[1])
                f107[date] = val
            except: pass
    if f107:
        print(f"  ✓ {len(f107)} дней · диапазон: {min(f107.values()):.0f}–{max(f107.values()):.0f} sfu")
        sources["f107"] = len(f107)
    else:
        raise ValueError("пустой файл")
except Exception as e:
    print(f"  ✗ {e} — используем прокси через солнечные пятна")

def get_f107(date_str):
    if date_str in f107:
        v = f107[date_str] / 300.0
        return [v, 1.0 if f107[date_str] > 200 else 0.0]
    sn = get_sunspot(date_str)[0]
    return [sn * 0.8 + 0.3, 1.0 if sn > 0.5 else 0.0]

# ══ 4. NASA DONKI CME ══
print("\n[4/6] NASA DONKI — корональные выбросы...")
cme_dates = {}
try:
    url = ("https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
           "?startDate=2010-01-01&endDate=2025-12-31")
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    for cme in data:
        try:
            date = cme["startTime"][:10]
            speed = 0
            if cme.get("cmeAnalyses"):
                speeds = [a.get("speed",0) for a in cme["cmeAnalyses"] if a.get("speed")]
                speed = max(speeds) if speeds else 0
            if date not in cme_dates or cme_dates[date] < speed:
                cme_dates[date] = speed
        except: pass
    print(f"  ✓ {len(cme_dates)} дней с CME · макс скорость: {max(cme_dates.values()):.0f} км/с")
    sources["cme"] = len(cme_dates)
except Exception as e:
    print(f"  ✗ {e}")

def get_cme(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    max_speed = 0
    for delta in range(-3, 8):
        d = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in cme_dates:
            max_speed = max(max_speed, cme_dates[d])
    return [max_speed / 3000.0,
            1.0 if max_speed > 1500 else 0.5 if max_speed > 800 else 0.0]

# ══ 5. ЛУННЫЕ РАССТОЯНИЯ (JPL Horizons) ══
print("\n[5/6] Лунные расстояния — вычисляем через ephem...")
def get_lunar_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    moon = ephem.Moon(); moon.compute(obs)
    sun  = ephem.Sun();  sun.compute(obs)
    dist_moon = float(moon.earth_distance)   # а.е.
    dist_sun  = float(sun.earth_distance)    # а.е.
    moon_phase = float(moon.phase) / 100.0
    moon_norm = (dist_moon - 0.00247) / (0.00272 - 0.00247)
    sun_norm  = (dist_sun - 0.983) / (1.017 - 0.983)
    return [moon_norm, sun_norm, moon_phase,
            1.0 if moon_norm < 0.1 else 0.0,   # перигей
            1.0 if moon_norm > 0.9 else 0.0]   # апогей
print("  ✓ 5 признаков: расстояния + фаза + перигей/апогей")
sources["lunar"] = "ephem"

# ══ 6. VIX / S&P500 через Yahoo Finance ══
print("\n[6/6] VIX волатильность — Yahoo Finance...")
vix_data = {}
try:
    # Yahoo Finance — публичный CSV
    import time
    end = int(time.time())
    start_ts = int(datetime(1990,1,1).timestamp())
    url = (f"https://query1.finance.yahoo.com/v7/finance/download/%5EVIX"
           f"?period1={start_ts}&period2={end}&interval=1mo&events=history")
    req = urllib.request.Request(url, headers={
        "User-Agent":"Mozilla/5.0",
        "Accept":"text/html,application/xhtml+xml"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("utf-8", errors="ignore")
    for line in text.split("\n")[1:]:
        p = line.strip().split(",")
        if len(p) >= 5:
            try:
                date = p[0][:7]
                close = float(p[4])
                vix_data[date] = close
            except: pass
    print(f"  ✓ VIX: {len(vix_data)} месяцев · диапазон: {min(vix_data.values()):.0f}–{max(vix_data.values()):.0f}")
    sources["vix"] = len(vix_data)
except Exception as e:
    print(f"  ✗ VIX: {e}")

def get_vix(date_str):
    ym = date_str[:7]
    if ym in vix_data:
        v = vix_data[ym]
        return [v/80.0, 1.0 if v>30 else 0.0, 1.0 if v>50 else 0.0]
    return [0.2, 0.0, 0.0]

# ══ СБОРКА ВСЕХ ПРИЗНАКОВ ══
print("\n" + "="*55)
print("СОБИРАЕМ ВСЕ ПРИЗНАКИ...")

with open("gdelt_events.json") as f: events = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

KP_STORMS = {
    "1989-03-13":9.0,"1991-03-24":9.0,"2000-07-15":9.0,
    "2001-03-31":9.0,"2003-10-29":9.0,"2003-10-30":9.0,
    "2003-11-20":9.0,"2004-11-08":9.0,"2005-05-15":8.7,
    "2015-03-17":8.3,"2015-06-22":8.0,"2017-09-07":8.0,
    "2024-05-10":9.0,"2024-10-10":9.0,
}
DST_STORMS = [
    ("1989-03-13",-589),("2000-07-15",-301),("2001-03-31",-387),
    ("2003-10-29",-383),("2003-11-20",-422),("2015-03-17",-222),
    ("2024-05-10",-412),("2024-10-10",-335),
]

def get_kp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    max_kp = 0.0
    for sd, kp in KP_STORMS.items():
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days) <= 15:
            max_kp = max(max_kp, kp)
    return [max_kp/9.0, 1.0 if max_kp>=7 else 0.0]

def get_dst(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    min_dst = 0.0
    for sd, dv in DST_STORMS:
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days)<=30 and dv<min_dst:
            min_dst=dv
    return [min_dst/600.0, 1.0 if min_dst<-300 else 0.0]

def mega_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer(); obs.date = dt.strftime("%Y/%m/%d")
    
    # Планеты (18 признаков)
    bodies = [ephem.Sun(),ephem.Moon(),ephem.Mercury(),ephem.Venus(),
              ephem.Mars(),ephem.Jupiter(),ephem.Saturn(),
              ephem.Uranus(),ephem.Neptune()]
    f = []
    for b in bodies:
        b.compute(obs)
        f += [math.sin(float(b.hlong)), math.cos(float(b.hlong))]
    
    # Аспекты (8)
    jup=ephem.Jupiter();sat=ephem.Saturn()
    ura=ephem.Uranus();nep=ephem.Neptune()
    for b in [jup,sat,ura,nep]: b.compute(obs)
    for a,b in [(jup,sat),(jup,ura),(sat,nep),(ura,nep)]:
        asp=abs(math.degrees(float(a.hlong)-float(b.hlong)))%360
        f+=[math.sin(math.radians(asp)),math.cos(math.radians(asp))]
    
    # Лунные узлы (2)
    moon=ephem.Moon();moon.compute(obs)
    f+=[math.sin(float(moon.hlong)+math.pi),math.cos(float(moon.hlong)+math.pi)]
    
    # Гравитация (3)
    best=min(grav_data,key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f+=[best.get("grav_potential_log",0),
        best.get("angular_momentum_log",0),
        best.get("tidal_force",0)]
    
    # Геомагнетизм (4)
    f+=get_dst(date_str)
    f+=get_kp(date_str)
    
    # НОВОЕ:
    f+=get_sunspot(date_str)    # 3
    f+=get_quake(date_str)      # 2
    f+=get_f107(date_str)       # 2
    f+=get_cme(date_str)        # 2
    f+=get_lunar_features(date_str)  # 5
    f+=get_vix(date_str)        # 3
    
    return f

# Тест одной даты
test_f = mega_features("2003-10-29")
n_features = len(test_f)
print(f"Признаков на дату: {n_features}")
print(f"  Было: 37 · Стало: {n_features} · Новых: +{n_features-37}")

# Датасет
random.seed(42)
X, y = [], []
for e in events:
    try: X.append(mega_features(e["date"])); y.append(e["category"])
    except: pass

start=datetime(1979,1,1)
edates=[datetime.strptime(e["date"],"%Y-%m-%d") for e in events]
n=0
for _ in range(5000):
    d=start+timedelta(days=random.randint(0,16436))
    if not any(abs((d-ed).days)<45 for ed in edates):
        try: X.append(mega_features(d.strftime("%Y-%m-%d"))); y.append("neutral"); n+=1
        except: pass
    if n>=len(events)*2: break

le=LabelEncoder(); y_enc=le.fit_transform(y)
X=np.array(X); classes=list(le.classes_)
print(f"Датасет: {len(y)} точек · {X.shape[1]} признаков")

# Обучение
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
base=GradientBoostingClassifier(n_estimators=400,max_depth=4,
                                 learning_rate=0.04,random_state=42)
scores=cross_val_score(base,X,y_enc,cv=cv,scoring='f1_weighted')

print(f"\nF1 ПРОГРЕСС:")
print(f"  Планеты only:     0.450")
print(f"  + гравитация:     0.569")
print(f"  + Kp + Dst:       0.574")
print(f"  + ВСЕ источники:  {scores.mean():.3f} ± {scores.std():.3f}")

# Обучаем финал
model=CalibratedClassifierCV(base,cv=3,method='isotonic')
model.fit(X,y_enc)

# Feature importance
fi=model.estimator.feature_importances_
feature_names=(
    [f"planet_{i//2}_{'sin' if i%2==0 else 'cos'}" for i in range(18)]+
    [f"aspect_{i//2}_{'sin' if i%2==0 else 'cos'}" for i in range(8)]+
    ["node_sin","node_cos",
     "grav_pot","ang_mom","tidal",
     "dst_val","dst_storm",
     "kp_max","kp_storm",
     "sunspot_val","sun_max","sun_min",
     "quake_mag","quake_big",
     "f107_val","f107_high",
     "cme_speed","cme_strong",
     "moon_dist","sun_dist","moon_phase","perigee","apogee",
     "vix_val","vix_high","vix_extreme"]
)
top_idx=np.argsort(fi)[::-1][:15]
print(f"\nТОП-15 ПРИЗНАКОВ:")
for i in top_idx:
    name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
    bar = "█"*int(fi[i]*300)
    print(f"  {name:<22} {fi[i]:.4f}  {bar}")

# Прогноз
print(f"\nФИНАЛЬНЫЙ ПРОГНОЗ ({n_features} признаков):")
peaks=[
    ("2026-08-29","geopolitical пик"),
    ("2026-10-28","natural_disaster"),
    ("2027-09-23","economic"),
    ("2027-12-22","декабрь 2027"),
    ("2028-12-16","tech горизонт"),
]
results={}
for date,label in peaks:
    proba=model.predict_proba([mega_features(date)])[0]
    pairs=sorted(zip(classes,proba),key=lambda x:-x[1])
    print(f"\n{date} ({label}):")
    for c,p in pairs:
        if p>0.05:
            print(f"  {c:<22} {p:.1%}  {'▓'*int(p*30)}")
    results[date]={c:round(float(p),4) for c,p in zip(classes,proba)}

with open("mega_forecast.json","w") as f:
    json.dump({
        "f1":round(float(scores.mean()),3),
        "f1_std":round(float(scores.std()),3),
        "n_features":int(X.shape[1]),
        "sources_loaded":sources,
        "top_features":[feature_names[i] if i<len(feature_names) else f"feat_{i}"
                        for i in top_idx],
        "forecast":results
    },f,indent=2)

print(f"\n✓ mega_forecast.json")
print(f"Источников загружено: {len(sources)}")
