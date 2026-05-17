import json, math, numpy as np, random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import ephem, urllib.request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

print("РЕГИОНАЛЬНАЯ МОДЕЛЬ — микро-прогноз по регионам\n")

# Привязка стран к регионам
COUNTRY_REGION = {
    # Европа
    "france":"europe","germany":"europe","uk":"europe","england":"europe",
    "spain":"europe","italy":"europe","russia":"europe","ukraine":"europe",
    "poland":"europe","serbia":"europe","greece":"europe","sweden":"europe",
    "norway":"europe","netherlands":"europe","belgium":"europe","austria":"europe",
    "hungary":"europe","czech":"europe","romania":"europe","bulgaria":"europe",
    "yugoslavia":"europe","kosovo":"europe","bosnia":"europe","croatia":"europe",
    # Ближний Восток
    "israel":"middle_east","palestine":"middle_east","iran":"middle_east",
    "iraq":"middle_east","syria":"middle_east","lebanon":"middle_east",
    "jordan":"middle_east","saudi":"middle_east","yemen":"middle_east",
    "turkey":"middle_east","egypt":"middle_east","libya":"middle_east",
    "kuwait":"middle_east","qatar":"middle_east","uae":"middle_east",
    # Азия
    "china":"asia","japan":"asia","korea":"asia","india":"asia",
    "pakistan":"asia","afghanistan":"asia","vietnam":"asia","cambodia":"asia",
    "thailand":"asia","indonesia":"asia","philippines":"asia","taiwan":"asia",
    "myanmar":"asia","bangladesh":"asia","nepal":"asia","mongolia":"asia",
    # Африка
    "africa":"africa","nigeria":"africa","ethiopia":"africa","somalia":"africa",
    "sudan":"africa","congo":"africa","angola":"africa","mozambique":"africa",
    "rwanda":"africa","kenya":"africa","tanzania":"africa","ghana":"africa",
    "mali":"africa","niger":"africa","chad":"africa","cameroon":"africa",
    # Америки
    "usa":"americas","united states":"americas","america":"americas",
    "mexico":"americas","cuba":"americas","venezuela":"americas",
    "colombia":"americas","peru":"americas","brazil":"americas",
    "argentina":"americas","chile":"americas","panama":"americas",
    "nicaragua":"americas","guatemala":"americas","haiti":"americas",
    # СССР/постсоветское
    "soviet":"eurasia","ussr":"eurasia","georgia":"eurasia",
    "azerbaijan":"eurasia","armenia":"eurasia","chechnya":"eurasia",
    "kazakhstan":"eurasia","uzbekistan":"eurasia","belarus":"eurasia",
}

def get_region(event):
    name = event.get("name","").lower()
    cat = event.get("category","")
    source = event.get("source","")

    # USGS — по координатам не можем, используем название места
    if "usgs" in source:
        for keyword, region in COUNTRY_REGION.items():
            if keyword in name: return region
        # Если нет — глобальный
        return "global"

    # По названию события
    for keyword, region in COUNTRY_REGION.items():
        if keyword in name: return region

    # По категории если нет геолокации
    return "global"

# Загружаем данные
with open("filtered_events.json") as f: all_ev = json.load(f)
with open("quarterly_gravity.json") as f: grav_data = json.load(f)

# Добавляем регион к каждому событию
for e in all_ev:
    e["region"] = get_region(e)

regions = Counter(e["region"] for e in all_ev)
print("Регионы:")
for reg, n in regions.most_common():
    print(f"  {reg:<15} {n:>4}  {'█'*(n//5)}")

# Солнечные пятна
sunspots = {}
try:
    url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        for line in r.read().decode().split("\n"):
            p=line.strip().split()
            if len(p)>=4:
                try:
                    yr,mo,sn=int(p[0]),int(p[1]),float(p[3])
                    for d in range(1,32): sunspots[f"{yr:04d}-{mo:02d}-{d:02d}"]=sn
                except: pass
except: pass

KP={"1989-03-13":9.0,"1991-03-24":9.0,"2000-07-15":9.0,
    "2001-03-31":9.0,"2003-10-29":9.0,"2015-03-17":8.3,
    "2024-05-10":9.0}

def get_sn(ds):
    dt=datetime.strptime(ds,"%Y-%m-%d")
    for d2 in range(32):
        d=(dt-timedelta(days=d2)).strftime("%Y-%m-%d")
        if d in sunspots: sn=sunspots[d]; return [sn/300.0,1.0 if sn>150 else 0.0]
    return [0.5,0.0]

def get_kp(ds):
    dt=datetime.strptime(ds,"%Y-%m-%d")
    mx=0.0
    for sd,kp in KP.items():
        if abs((dt-datetime.strptime(sd,"%Y-%m-%d")).days)<=15: mx=max(mx,kp)
    return mx/9.0

def features(date_str):
    dt=datetime.strptime(date_str,"%Y-%m-%d")
    obs=ephem.Observer(); obs.date=dt.strftime("%Y/%m/%d")
    bodies=[ephem.Sun(),ephem.Moon(),ephem.Mercury(),ephem.Venus(),
            ephem.Mars(),ephem.Jupiter(),ephem.Saturn(),
            ephem.Uranus(),ephem.Neptune()]
    f=[]; lons=[]
    for b in bodies:
        b.compute(obs); lon=float(b.hlong); lons.append(lon)
        f+=[math.sin(lon),math.cos(lon)]
    yr=dt.year+dt.timetuple().tm_yday/365.25
    for period,ref in [(50.85,1996),(18.613,2000),(8.85,2000)]:
        lon=((yr-ref)/period*360)%360
        f+=[math.sin(math.radians(lon)),math.cos(math.radians(lon))]
    pairs=[(0,4),(0,5),(0,6),(1,5),(1,6),(2,5),(3,4),(3,5),(4,5),(4,6),(5,6)]
    for i,j in pairs:
        asp=abs(math.degrees(lons[i]-lons[j]))%360
        f+=[math.sin(math.radians(asp)),math.cos(math.radians(asp)),
            math.sin(math.radians(asp*2))]
    moon=ephem.Moon(); moon.compute(obs)
    f+=[math.sin(float(moon.hlong)+math.pi),math.cos(float(moon.hlong)+math.pi)]
    best=min(grav_data,key=lambda g:
        abs((datetime.strptime(g["date"][:10],"%Y-%m-%d")-dt).days))
    f+=[best.get("grav_potential_log",0),
        best.get("angular_momentum_log",0),
        best.get("tidal_force",0)]
    f+=[get_kp(date_str)]; f+=get_sn(date_str)
    sun2=ephem.Sun(); sun2.compute(obs)
    dm=float(moon.earth_distance); ds2=float(sun2.earth_distance)
    mn=(dm-0.00247)/(0.00272-0.00247)
    f+=[mn,(ds2-0.983)/(1.017-0.983),float(moon.phase)/100.0]
    for period in [11,18.6,29.5,84,165]:
        f+=[math.sin(2*math.pi*yr/period),math.cos(2*math.pi*yr/period)]
    return f

# ══ ГЛОБАЛЬНАЯ МОДЕЛЬ С РЕГИОНОМ КАК ПРИЗНАКОМ ══
print("\n" + "="*55)
print("МОДЕЛЬ: тип события + регион как составной класс")
print("="*55)

# Создаём составные метки: "geopolitical_europe", "epidemic_global" итд
random.seed(42)
by_compound = defaultdict(list)
for e in all_ev:
    if e["region"] in ("europe","middle_east","asia","americas","eurasia","africa"):
        label = f"{e['category']}_{e['region']}"
    else:
        label = e["category"]  # global остаётся без региона
    by_compound[label].append(e)

print("\nСоставные классы:")
for label, evs in sorted(by_compound.items(), key=lambda x:-len(x[1])):
    print(f"  {label:<35} {len(evs):>4}")

# Берём только классы с достаточным количеством примеров
MIN_SAMPLES = 15
valid_labels = {k:v for k,v in by_compound.items() if len(v)>=MIN_SAMPLES}
print(f"\nКлассов с ≥{MIN_SAMPLES} примеров: {len(valid_labels)}")

events_compound = []
for label, evs in valid_labels.items():
    sampled = random.sample(evs, min(len(evs), 60))
    for e in sampled:
        e2 = dict(e); e2["compound_label"] = label
        events_compound.append(e2)

print(f"Событий после выборки: {len(events_compound)}")

# Признаки + нейтральные
X,y=[],[]
for e in events_compound:
    try: X.append(features(e["date"])); y.append(e["compound_label"])
    except: pass

start=datetime(1950,1,1)
edates=[datetime.strptime(e["date"],"%Y-%m-%d") for e in events_compound]
n=0
for _ in range(15000):
    d=start+timedelta(days=random.randint(0,27375))
    if not any(abs((d-ed).days)<45 for ed in edates):
        try: X.append(features(d.strftime("%Y-%m-%d"))); y.append("neutral"); n+=1
        except: pass
    if n>=len(y)//3: break

le=LabelEncoder(); y_enc=le.fit_transform(y)
X=np.array(X); classes=list(le.classes_)
print(f"\nДатасет: {len(y)} точек · {X.shape[1]} признаков")
print(f"Классов: {len(classes)}")

# CV
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
base=GradientBoostingClassifier(n_estimators=300,max_depth=4,
                                 learning_rate=0.05,random_state=42)
scores=cross_val_score(base,X,y_enc,cv=cv,scoring='f1_weighted')
print(f"Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")

# Walk-forward
SPLIT=2006
ev_tr=[e for e in events_compound if int(e["date"][:4])<SPLIT]
ev_te=[e for e in events_compound if int(e["date"][:4])>=SPLIT]

def build(evs,sy,ey):
    Xb,yb=[],[]
    eds=[datetime.strptime(e["date"],"%Y-%m-%d") for e in evs]
    for e in evs:
        try: Xb.append(features(e["date"])); yb.append(e["compound_label"])
        except: pass
    st=datetime(sy,1,1); en=datetime(ey,12,31); nn=0
    for _ in range(8000):
        d=st+timedelta(days=random.randint(0,(en-st).days))
        if not any(abs((d-ed).days)<45 for ed in eds):
            try: Xb.append(features(d.strftime("%Y-%m-%d"))); yb.append("neutral"); nn+=1
            except: pass
        if nn>=len(Xb)//2: break
    return np.array(Xb),yb

X_tr,y_tr=build(ev_tr,1950,SPLIT-1)
X_te,y_te=build(ev_te,SPLIT,2024)
le2=LabelEncoder(); le2.fit(list(y_tr)+list(y_te))
y_tr_e=le2.transform(y_tr); y_te_e=le2.transform(y_te)

clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,
                                learning_rate=0.05,random_state=42)
clf.fit(X_tr,y_tr_e)
pred=clf.predict(X_te)
f1_wf=f1_score(y_te_e,pred,average='weighted',zero_division=0)
print(f"Walk-forward F1: {f1_wf:.3f}")

# Финальная модель
model=CalibratedClassifierCV(base,cv=3,method='isotonic')
model.fit(X,y_enc)

# ══ РЕГИОНАЛЬНЫЙ ПРОГНОЗ ══
print(f"\n{'='*55}")
print("МИКРО-ПРОГНОЗ ПО РЕГИОНАМ 2026-2028")
print("="*55)

peaks=[
    ("2026-08-29","август 2026"),
    ("2027-03-21","март 2027"),
    ("2027-09-23","осень 2027"),
    ("2027-12-22","декабрь 2027"),
    ("2028-12-16","конец горизонта"),
]

all_forecasts = {}
for date, label in peaks:
    proba=model.predict_proba([features(date)])[0]
    pairs=sorted(zip(classes,proba),key=lambda x:-x[1])
    print(f"\n{date} ({label}):")
    # Показываем топ-8 не-neutral
    shown=0
    for c,p in pairs:
        if c!="neutral" and p>0.03 and shown<8:
            region_flag = ""
            if "europe" in c: region_flag="🌍"
            elif "middle_east" in c: region_flag="🌏"
            elif "asia" in c: region_flag="🌏"
            elif "americas" in c: region_flag="🌎"
            elif "africa" in c: region_flag="🌍"
            print(f"  {c:<35} {p:.1%}  {'▓'*int(p*25)}")
            shown+=1
    neutral_p=[p for c,p in pairs if c=="neutral"][0]
    print(f"  {'neutral':<35} {neutral_p:.1%}")
    all_forecasts[date]={c:round(float(p),4) for c,p in zip(classes,proba)}

with open("regional_forecast.json","w") as f:
    json.dump({
        "f1_walkforward":round(float(f1_wf),3),
        "f1_crossval":round(float(scores.mean()),3),
        "n_classes":len(classes),
        "classes":classes,
        "forecast":all_forecasts
    },f,indent=2)

print(f"\n✓ regional_forecast.json")
print(f"Walk-forward F1={f1_wf:.3f} · {len(classes)} региональных классов")
