import urllib.request, urllib.parse, json, numpy as np, networkx as nx
from pathlib import Path

DATA_DIR = Path(".")
print("Шаг 1: запрашиваю 5000 галактик из SDSS DR17...")

SQL = """SELECT TOP 5000 p.objID, p.ra, p.dec, s.z AS redshift
FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objID
WHERE s.class = 'GALAXY' AND s.z BETWEEN 0.01 AND 0.15 AND s.zWarning = 0 AND p.clean = 1"""

url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?" + \
      urllib.parse.urlencode({"cmd": SQL, "format": "json"})

try:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    rows = data[0]["Rows"]
    print(f"  ✓ получено: {len(rows)} галактик")
    with open("sdss_galaxies.json","w") as f:
        json.dump(rows, f)
except Exception as e:
    print(f"  ✗ ошибка: {e}")
    raise SystemExit(1)

print("Шаг 2: строю граф галактик...")
G = nx.Graph()
for g in rows:
    G.add_node(g["objID"], ra=float(g["ra"]), dec=float(g["dec"]), z=float(g["redshift"]))

coords = np.array([[float(g["ra"]), float(g["dec"])] for g in rows])
ids    = [g["objID"] for g in rows]
edges  = 0
for i in range(len(rows)):
    for j in range(i+1, len(rows)):
        dra  = (coords[i,0]-coords[j,0]) * np.cos(np.radians(coords[i,1]))
        ddec = coords[i,1]-coords[j,1]
        dist = np.sqrt(dra**2+ddec**2)
        if dist < 1.5:
            G.add_edge(ids[i], ids[j], weight=round(1/(dist+0.01),4))
            edges += 1
    if i % 500 == 0:
        print(f"  обработано {i}/{len(rows)}...", end="\r")

nx.write_graphml(G, "galaxy_graph.graphml")
galaxy_metrics = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "clustering": round(nx.average_clustering(G), 4),
    "density":    round(nx.density(G), 4),
    "mean_degree":round(np.mean([d for _,d in G.degree()]), 2),
}
with open("galaxy_metrics.json","w") as f:
    json.dump(galaxy_metrics, f, indent=2)

print("\nШаг 3: сравниваю с клеткой...")
with open("cell_metrics.json") as f:
    cell_metrics = json.load(f)

print("\n" + "="*55)
print(f"{'Метрика':<22} {'Клетка':>10} {'Галактики':>10} {'Схожесть':>10}")
print("-"*55)
keys = ["clustering","density","mean_degree"]
for k in keys:
    c = cell_metrics.get(k,0)
    g = galaxy_metrics.get(k,0)
    ratio = c/g if g else 0
    bar = "✓✓" if 0.5<ratio<2 else "✓" if 0.1<ratio<10 else "✗"
    print(f"{bar} {k:<20} {c:>10.4f} {g:>10.4f} {ratio:>10.3f}")

print("="*55)
print(f"\n  Клетка:    {cell_metrics['nodes']} узлов, {cell_metrics['edges']} рёбер")
print(f"  Галактики: {galaxy_metrics['nodes']} узлов, {galaxy_metrics['edges']} рёбер")

result = {"cell": cell_metrics, "galaxy": galaxy_metrics}
with open("comparison_result.json","w") as f:
    json.dump(result, f, indent=2)

print("\n✓ ГОТОВО!")
print("  → galaxy_graph.graphml")
print("  → comparison_result.json  ← отправь учёным")
