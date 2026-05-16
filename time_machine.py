import urllib.request, urllib.parse, json, networkx as nx, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Скачиваю галактики по временным слоям (z = красное смещение)...")

# Четыре момента времени вселенной
TIME_SLICES = {
    "z=0.25 (3 млрд лет назад)": (0.20, 0.30),
    "z=0.15 (2 млрд лет назад)": (0.10, 0.20),
    "z=0.07 (1 млрд лет назад)": (0.04, 0.10),
    "z=0.02 (сейчас)":           (0.01, 0.04),
}

SDSS_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
slice_metrics = {}

for label, (z_lo, z_hi) in TIME_SLICES.items():
    cache = Path(f"slice_{z_lo:.2f}.json")
    if cache.exists():
        with open(cache) as f: rows = json.load(f)
        print(f"  {label}: загружено из кэша ({len(rows)} галактик)")
    else:
        SQL = f"""SELECT TOP 2000 p.objID, p.ra, p.dec, s.z AS redshift
        FROM PhotoObj p JOIN SpecObj s ON s.bestobjid = p.objID
        WHERE s.class='GALAXY' AND s.z BETWEEN {z_lo} AND {z_hi}
        AND s.zWarning=0 AND p.clean=1"""
        params = urllib.parse.urlencode({"cmd":SQL,"format":"json"})
        try:
            with urllib.request.urlopen(f"{SDSS_URL}?{params}", timeout=30) as r:
                data = json.loads(r.read())
            rows = data[0]["Rows"]
            with open(cache,"w") as f: json.dump(rows,f)
            print(f"  {label}: {len(rows)} галактик")
        except Exception as e:
            print(f"  {label}: ошибка {e}")
            continue

    # Строим граф
    G = nx.Graph()
    for g in rows:
        G.add_node(g["objID"], ra=float(g["ra"]), dec=float(g["dec"]))
    coords = np.array([[float(g["ra"]),float(g["dec"])] for g in rows])
    ids = [g["objID"] for g in rows]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            dra  = (coords[i,0]-coords[j,0])*np.cos(np.radians(coords[i,1]))
            ddec = coords[i,1]-coords[j,1]
            if np.sqrt(dra**2+ddec**2) < 1.5:
                G.add_edge(ids[i],ids[j])

    # Метрики
    L_norm = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eig = np.linalg.eigvalsh(L_norm)
    slice_metrics[label] = {
        "z_mid": (z_lo+z_hi)/2,
        "spectral_mean": round(float(eig.mean()),5),
        "clustering":    round(nx.average_clustering(G),4),
        "nodes":         G.number_of_nodes(),
    }

# Результат
print("\n"+"="*70)
print(f"{'Момент времени':<30} {'spectral_mean':>14} {'clustering':>12} {'узлов':>8}")
print("-"*70)
for label, m in slice_metrics.items():
    print(f"  {label:<28} {m['spectral_mean']:>14.5f} {m['clustering']:>12.4f} {m['nodes']:>8}")
print("="*70)

# График — эволюция F(s) во времени
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Машина времени — эволюция F(s) по красному смещению",
             fontsize=13, color="white")

z_vals = [m["z_mid"] for m in slice_metrics.values()]
sm_vals = [m["spectral_mean"] for m in slice_metrics.values()]
cl_vals = [m["clustering"] for m in slice_metrics.values()]
labels  = [l.split("(")[0].strip() for l in slice_metrics.keys()]

for ax, vals, ylabel, color in [
    (ax1, sm_vals, "spectral_mean", "#5DCAA5"),
    (ax2, cl_vals, "clustering",    "#EF9F27"),
]:
    ax.set_facecolor("#0d1117")
    ax.plot(z_vals, vals, "o-", color=color, linewidth=2,
            markersize=10, markerfacecolor=color)
    for z, v, lbl in zip(z_vals, vals, labels):
        ax.annotate(f"{lbl}\n{v:.4f}", (z,v),
                    textcoords="offset points", xytext=(0,12),
                    ha="center", fontsize=8, color=color)
    ax.set_xlabel("Красное смещение z (больше z = дальше в прошлое)",
                  color="#888888")
    ax.set_ylabel(ylabel, color="#888888")
    ax.set_title(ylabel, color="white")
    ax.tick_params(colors="#888888")
    ax.invert_xaxis()  # прошлое слева, настоящее справа
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

ax1.axhline(0.972, color="#534AB7", linestyle="--",
            linewidth=1, label="уровень клетки")
ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

plt.tight_layout()
plt.savefig("time_machine.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("\n✓ time_machine.png — эволюция F(s) во времени")
print("  Если линия плоская — F(s) стабильна через время")
print("  Если меняется — видим как вселенная эволюционировала")
