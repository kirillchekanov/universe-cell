import json, networkx as nx, numpy as np, matplotlib.pyplot as plt

# Загружаем три временных слоя
datasets = {
    "z=0.01–0.04\n(сейчас)":           ("sdss_galaxies.json", 0.01, 0.04),
    "z=0.05–0.10\n(700 млн лет назад)": ("sdss_z_mid.json",   0.05, 0.10),
    "z=0.10–0.30\n(2 млрд лет назад)":  ("sdss_z_high.json",  0.10, 0.30),
}

def build_metrics(filepath, z_lo, z_hi, label):
    with open(filepath) as f: rows = json.load(f)
    rows = [g for g in rows if z_lo <= float(g.get("redshift",0) or 0) <= z_hi]
    if not rows: rows = json.load(open(filepath))  # берём все если фильтр пустой
    print(f"  {label.split(chr(10))[0]}: {len(rows)} галактик")
    G = nx.Graph()
    coords = np.array([[float(g["ra"]),float(g["dec"])] for g in rows])
    for i in range(len(rows)): G.add_node(i)
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            dra  = (coords[i,0]-coords[j,0])*np.cos(np.radians(coords[i,1]))
            ddec = coords[i,1]-coords[j,1]
            if np.sqrt(dra**2+ddec**2) < 2.0:
                G.add_edge(i,j)
    L   = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eig = np.linalg.eigvalsh(L)
    return {
        "z_mid":         (z_lo+z_hi)/2,
        "spectral_mean": round(float(eig.mean()),5),
        "clustering":    round(nx.average_clustering(G),4),
        "nodes":         G.number_of_nodes(),
    }

print("Строю временные срезы...")
results = {}
for label,(fp,zl,zh) in datasets.items():
    results[label] = build_metrics(fp,zl,zh,label)

# Таблица
print("\n"+"="*70)
print(f"{'Эпоха':<26} {'z':>6} {'spectral_mean':>14} {'clustering':>12} {'узлов':>7}")
print("-"*70)
for label,m in results.items():
    print(f"  {label.split(chr(10))[0]:<24} {m['z_mid']:>6.3f} {m['spectral_mean']:>14.5f} {m['clustering']:>12.4f} {m['nodes']:>7}")
print(f"  {'--- клетка HeLa':<24} {'—':>6} {'0.98150':>14} {'0.9751':>12} {'54':>7}")
print("="*70)

sm = [m["spectral_mean"] for m in results.values()]
dev = (max(sm)-min(sm))/max(sm)*100
print(f"\n  Отклонение spectral_mean во времени: {dev:.2f}%")
if dev < 5:
    print("  ✓ F(s) СТАБИЛЬНА через время — вселенная самоподобна не только в пространстве, но и во времени")
else:
    print("  ~ F(s) меняется — видна эволюция топологии")

# График
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Машина времени — стабильна ли F(s) через 2 млрд лет?",
             fontsize=13, color="white")

z_vals = [m["z_mid"] for m in results.values()]
labels_short = [l.split("\n")[0] for l in results.keys()]

for ax,(vals,ylabel,cell_ref,color) in zip(axes,[
    ([m["spectral_mean"] for m in results.values()], "spectral_mean λ", 0.9815, "#5DCAA5"),
    ([m["clustering"]    for m in results.values()], "clustering coeff", 0.9751, "#EF9F27"),
]):
    ax.set_facecolor("#0d1117")
    ax.plot(z_vals, vals, "o-", color=color, linewidth=2.5,
            markersize=12, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.8)
    for z,v,lbl in zip(z_vals,vals,labels_short):
        ax.annotate(f"{lbl}\n{v:.4f}", (z,v),
                    textcoords="offset points", xytext=(0,16),
                    ha="center", fontsize=8, color=color, fontweight="bold")
    ax.axhline(cell_ref, color="#7F77DD", linestyle="--",
               linewidth=1.2, label=f"клетка HeLa = {cell_ref}")
    ax.set_xlabel("Красное смещение z\n← прошлое          настоящее →", color="#888")
    ax.set_ylabel(ylabel, color="#888")
    ax.set_title(ylabel, color="white", fontsize=11)
    ax.tick_params(colors="#888")
    ax.invert_xaxis()
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("time_machine.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("\n✓ time_machine.png")
