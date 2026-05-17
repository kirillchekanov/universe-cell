import urllib.request, json
from datetime import datetime
from collections import Counter

print("РАСШИРЕНИЕ ДАТАСЕТА — история + USGS\n")

with open("big_events.json") as f:
    existing = json.load(f)

existing_keys = {(e["date"][:7], e["category"]) for e in existing}
new_events = []

def add(date, cat, name, source):
    key = (date[:7], cat)
    if key not in existing_keys:
        existing_keys.add(key)
        new_events.append({"date":date,"category":cat,"name":name,"source":source,"goldstein":-7.0,"tone":-3.0})

# ══ ИСТОРИЯ ДО 1800 ══
HISTORICAL = [
    # Войны и геополитика
    ("0264-03-01","geopolitical","First Punic War"),
    ("0218-03-01","geopolitical","Second Punic War"),
    ("0044-03-15","geopolitical","Assassination of Caesar"),
    ("0476-09-04","geopolitical","Fall of Western Rome"),
    ("0632-06-08","geopolitical","Arab expansion begins"),
    ("0711-04-19","geopolitical","Muslim conquest Iberia"),
    ("0732-10-10","geopolitical","Battle of Tours"),
    ("1066-10-14","geopolitical","Battle of Hastings"),
    ("1095-11-27","geopolitical","First Crusade"),
    ("1206-03-01","geopolitical","Mongol Empire"),
    ("1215-06-15","geopolitical","Magna Carta"),
    ("1241-04-09","geopolitical","Mongol invasion Europe"),
    ("1258-02-13","geopolitical","Sack of Baghdad"),
    ("1337-05-24","geopolitical","Hundred Years War"),
    ("1356-09-19","geopolitical","Battle of Poitiers"),
    ("1415-10-25","geopolitical","Battle of Agincourt"),
    ("1453-05-29","geopolitical","Fall of Constantinople"),
    ("1455-05-22","geopolitical","Wars of the Roses"),
    ("1492-01-02","geopolitical","Reconquista complete"),
    ("1517-10-31","geopolitical","Protestant Reformation"),
    ("1519-04-21","geopolitical","Spanish conquest Mexico"),
    ("1521-05-21","geopolitical","Fall of Tenochtitlan"),
    ("1571-10-07","geopolitical","Battle of Lepanto"),
    ("1572-08-24","geopolitical","St Bartholomew massacre"),
    ("1588-08-08","geopolitical","Spanish Armada"),
    ("1618-05-23","geopolitical","Thirty Years War"),
    ("1642-08-22","geopolitical","English Civil War"),
    ("1648-10-24","geopolitical","Peace of Westphalia"),
    ("1688-11-05","geopolitical","Glorious Revolution"),
    ("1700-11-30","geopolitical","Great Northern War"),
    ("1756-08-29","geopolitical","Seven Years War"),
    ("1776-07-04","geopolitical","American Independence"),
    ("1789-07-14","geopolitical","French Revolution"),
    ("1793-09-05","geopolitical","Reign of Terror"),
    ("1799-11-09","geopolitical","Napoleon coup"),
    # Экономические кризисы
    ("1340-01-01","economic_crisis","Bardi Peruzzi bankruptcy"),
    ("1557-01-01","economic_crisis","Spanish sovereign default"),
    ("1637-02-05","economic_crisis","Tulip Mania collapse"),
    ("1720-01-01","economic_crisis","South Sea Bubble"),
    ("1720-09-01","economic_crisis","Mississippi Bubble"),
    ("1772-06-01","economic_crisis","Credit Crisis 1772"),
    ("1792-03-09","economic_crisis","Panic of 1792"),
    # Эпидемии
    ("0165-01-01","epidemic","Antonine Plague"),
    ("0249-01-01","epidemic","Plague of Cyprian"),
    ("0541-07-01","epidemic","Plague of Justinian"),
    ("1347-10-01","epidemic","Black Death arrives"),
    ("1348-01-01","epidemic","Black Death peak"),
    ("1489-01-01","epidemic","Typhus epidemic Spain"),
    ("1520-01-01","epidemic","Smallpox kills Aztecs"),
    ("1576-01-01","epidemic","Cocoliztli epidemic"),
    ("1629-01-01","epidemic","Italian plague"),
    ("1656-01-01","epidemic","Naples plague"),
    ("1665-06-01","epidemic","Great Plague of London"),
    ("1720-05-25","epidemic","Plague of Marseille"),
    ("1793-08-01","epidemic","Yellow Fever Philadelphia"),
    # Технологии
    ("1543-02-19","tech_breakthrough","Copernicus heliocentric"),
    ("1492-10-12","tech_breakthrough","Columbus Americas"),
    ("1687-07-05","tech_breakthrough","Newton Principia"),
    ("1769-01-25","tech_breakthrough","Watt steam engine"),
    ("1776-03-01","tech_breakthrough","Wealth of Nations"),
    ("1783-12-17","tech_breakthrough","First balloon flight"),
    ("1796-05-14","tech_breakthrough","Smallpox vaccine"),
    # Природные катастрофы
    ("0365-07-21","natural_disaster","Alexandria earthquake tsunami"),
    ("0526-05-20","natural_disaster","Antioch earthquake"),
    ("1138-10-11","natural_disaster","Aleppo earthquake"),
    ("1556-01-23","natural_disaster","Shaanxi earthquake 830k dead"),
    ("1666-09-02","natural_disaster","Great Fire of London"),
    ("1707-10-28","natural_disaster","Hoei earthquake Japan"),
    ("1755-11-01","natural_disaster","Lisbon Earthquake"),
    ("1783-02-05","natural_disaster","Calabrian earthquake"),
]

n1 = 0
for date, cat, name in HISTORICAL:
    add(date, cat, name, "historical_manual")
    n1 += 1
print(f"Исторических событий до 1800: {n1}")

# ══ USGS M≥7.5 ══
print("\nUSGS землетрясения M≥7.5...")
try:
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query"
           "?format=geojson&starttime=1900-01-01&endtime=2025-12-31"
           "&minmagnitude=7.5&orderby=time&limit=2000")
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    qc = 0
    for feat in data.get("features",[]):
        props = feat.get("properties",{})
        t = props.get("time",0)
        mag = props.get("mag",0)
        if t:
            dt = datetime.fromtimestamp(t/1000)
            add(dt.strftime("%Y-%m-%d"), "natural_disaster",
                f"M{mag} {props.get('place','')[:40]}", "usgs_api")
            qc += 1
    print(f"  ✓ {qc} землетрясений M≥7.5")
except Exception as e:
    print(f"  ✗ {e}")

# ══ ИТОГ ══
all_events = existing + new_events
all_events.sort(key=lambda x: x["date"])

print(f"\nБыло:    {len(existing)}")
print(f"Новых:   {len(new_events)}")
print(f"Итого:   {len(all_events)}")

cats = Counter(e["category"] for e in all_events)
print("\nПо категориям:")
for cat, n in cats.most_common():
    print(f"  {cat:<22} {n:>4}  {'█'*(n//5)}")

dates = sorted([e["date"] for e in all_events])
print(f"\nДиапазон: {dates[0][:4]} → {dates[-1][:4]} ({int(dates[-1][:4])-int(dates[0][:4])} лет)")

with open("mega_events.json","w") as f:
    json.dump(all_events, f, indent=2, ensure_ascii=False)
print(f"\n✓ mega_events.json")
