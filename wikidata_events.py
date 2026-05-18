import urllib.request, json, time
from datetime import datetime
from collections import Counter

print("WIKIDATA SPARQL — исторические события с 1800\n")

ENDPOINT = "https://query.wikidata.org/sparql"

def sparql_query(query, label):
    """Выполняем SPARQL запрос к Wikidata"""
    url = ENDPOINT + "?query=" + urllib.parse.quote(query) + "&format=json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "UniverseCell/1.0 (research project; kirillchekanov@gmail.com)",
        "Accept": "application/sparql-results+json"
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        results = data["results"]["bindings"]
        print(f"  ✓ {label}: {len(results)} событий")
        return results
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return []

import urllib.parse

events = []

# ══ 1. ВОЙНЫ И ВООРУЖЁННЫЕ КОНФЛИКТЫ ══
print("[1/5] Войны и конфликты...")
query_wars = """
SELECT DISTINCT ?event ?eventLabel ?start WHERE {
  ?event wdt:P31/wdt:P279* wd:Q198 .
  ?event wdt:P580 ?start .
  FILTER(?start >= "1800-01-01T00:00:00Z"^^xsd:dateTime)
  FILTER(?start <= "2024-12-31T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?start
LIMIT 2000
"""
results = sparql_query(query_wars, "войны")
for r in results:
    try:
        date = r["start"]["value"][:10]
        events.append({"date": date, "category": "geopolitical",
                       "source": "wikidata_war",
                       "name": r.get("eventLabel",{}).get("value","")[:60]})
    except: pass

time.sleep(2)

# ══ 2. РЕВОЛЮЦИИ ══
print("\n[2/5] Революции...")
query_rev = """
SELECT DISTINCT ?event ?eventLabel ?start WHERE {
  ?event wdt:P31/wdt:P279* wd:Q10931 .
  ?event wdt:P580 ?start .
  FILTER(?start >= "1800-01-01T00:00:00Z"^^xsd:dateTime)
  FILTER(?start <= "2024-12-31T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?start
LIMIT 500
"""
results = sparql_query(query_rev, "революции")
for r in results:
    try:
        date = r["start"]["value"][:10]
        events.append({"date": date, "category": "geopolitical",
                       "source": "wikidata_revolution",
                       "name": r.get("eventLabel",{}).get("value","")[:60]})
    except: pass

time.sleep(2)

# ══ 3. ЭКОНОМИЧЕСКИЕ КРИЗИСЫ ══
print("\n[3/5] Экономические кризисы...")
query_econ = """
SELECT DISTINCT ?event ?eventLabel ?start WHERE {
  { ?event wdt:P31/wdt:P279* wd:Q2143665 . }
  UNION
  { ?event wdt:P31/wdt:P279* wd:Q8161 . }
  ?event wdt:P580 ?start .
  FILTER(?start >= "1800-01-01T00:00:00Z"^^xsd:dateTime)
  FILTER(?start <= "2024-12-31T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?start
LIMIT 500
"""
results = sparql_query(query_econ, "кризисы")
for r in results:
    try:
        date = r["start"]["value"][:10]
        events.append({"date": date, "category": "economic_crisis",
                       "source": "wikidata_econ",
                       "name": r.get("eventLabel",{}).get("value","")[:60]})
    except: pass

time.sleep(2)

# ══ 4. ЭПИДЕМИИ ══
print("\n[4/5] Эпидемии и пандемии...")
query_epid = """
SELECT DISTINCT ?event ?eventLabel ?start WHERE {
  { ?event wdt:P31/wdt:P279* wd:Q133780 . }
  UNION
  { ?event wdt:P31/wdt:P279* wd:Q12136 . }
  ?event wdt:P580 ?start .
  FILTER(?start >= "1800-01-01T00:00:00Z"^^xsd:dateTime)
  FILTER(?start <= "2024-12-31T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?start
LIMIT 500
"""
results = sparql_query(query_epid, "эпидемии")
for r in results:
    try:
        date = r["start"]["value"][:10]
        events.append({"date": date, "category": "epidemic",
                       "source": "wikidata_epidemic",
                       "name": r.get("eventLabel",{}).get("value","")[:60]})
    except: pass

time.sleep(2)

# ══ 5. ТЕХНОЛОГИЧЕСКИЕ ПРОРЫВЫ ══
print("\n[5/5] Научные открытия и изобретения...")
query_tech = """
SELECT DISTINCT ?event ?eventLabel ?start WHERE {
  { ?event wdt:P31/wdt:P279* wd:Q7725634 . }
  UNION
  { ?event wdt:P31 wd:Q483247 . }
  ?event wdt:P571 ?start .
  FILTER(?start >= "1800-01-01T00:00:00Z"^^xsd:dateTime)
  FILTER(?start <= "2024-12-31T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?start
LIMIT 1000
"""
results = sparql_query(query_tech, "изобретения")
for r in results:
    try:
        date = r["start"]["value"][:10]
        events.append({"date": date, "category": "tech_breakthrough",
                       "source": "wikidata_tech",
                       "name": r.get("eventLabel",{}).get("value","")[:60]})
    except: pass

# ══ ИТОГ ══
print(f"\n{'='*50}")
print(f"Загружено: {len(events)} событий")

cats = Counter(e["category"] for e in events)
print("\nПо категориям:")
for cat, n in cats.most_common():
    bar = "█" * min(n//10, 50)
    print(f"  {cat:<22} {n:>5}  {bar}")

# Фильтруем дубли по дате+категории
seen = set()
unique = []
for e in events:
    key = (e["date"][:7], e["category"])  # по месяцу+категории
    if key not in seen:
        seen.add(key)
        unique.append(e)

print(f"\nПосле дедупликации: {len(unique)} событий")

# Сохраняем
with open("wikidata_events.json", "w") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print("✓ wikidata_events.json")
print("\nЗапускай: python3 retrain_wikidata.py")
