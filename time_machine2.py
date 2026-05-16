import json, networkx as nx, numpy as np, matplotlib.pyplot as plt

print("Загружаю уже скачанные галактики SDSS...")
with open("sdss_galaxies.json") as f:
    all_galaxies = json.load(f)

print(f"  Всего галактик: {len(all_galaxies)}")

# Разбиваем по z на 4 временных слоя
TIME_SLICES = {
    "z=0.10–0.15\n(~1.5 млрд лет назад)": (0.10, 0.15),
    "z=0.07–0.10\n(~1 млрд лет назад)":   (0.07, 0.10),
    "z=0.04–0.07\n(~600 млн лет назад)":  (0.04, 0.07),
    "z=0.01–0.04\n(~400 млн лет назад)":  (0.01, 0.04),
}

slice_metrics = {}

for label, (z_lo, z_hi) in TIME_SLICES.items():
    rows = [g for g in all_galaxies
            if z_lo <= float(g.get("redshift",0) or 0) < z_hi]
    if len(rows) < 20:
        print(f"  {label}: слишком мало галактик ({len(rows)}), пропускаю")
        continue
    print(f"  {label.split(chr(10))[0]}: {len(rows)} галактик")

    G = nx.Graph()
    coords = np.array([[float(g["ra"]), float(g["dec"])] for g in rows])
    ids = list(range(len(rows)))
    for i in ids:
        G.add_node(i, ra=float(rows[i]["ra"]), dec=float(rows[i]["dec"]))

    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            dra  = (coords[i,0]-coords[j,0]) * np.cos(np.radians(coords[i,1]))
            ddec = coords[i,1] - coords[j,1]
            if np.sqrt(dra**2 + ddec**2) < 2.0:
                G.add_edge(i, j)

    L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eig = np.linalg.eigvalsh(L)
    slice_metrics[label] = {
        "z_mid":         (z_lo + z_hi) / 2,
        "spectral_mean": round(float(eig.mean()), 5),
        "clustering":    round(nx.average_clustering(G), 4),
        "nodes":         G.number_of_nodes(),
        "edges":         G.number_of_edges(),
    }

# Таблица
print("\n" + "="*72)
print(f"{'Эпоха':<28} {'spectral_mean':>14} {'clustering':>12} {'узлов':>8}")
print("-"*72)
for label, m in slice_metrics.items():
    short = label.split("\n")[0]
    print(f"  {short:<26} {m['spectral_mean']:>14.5f} {m['clustering']:>12.4f} {m['nodes']:>8}")

# Добавляем клетку как референс
print(f"  {'--- клетка HeLa (референс)':<26} {'0.98150':>14} {'0.9751':>12} {'54':>8}")
print("="*72)

# Смотрим стабильность
sm_vals = [m["spectral_mean"] for m in slice_metrics.values()]
if sm_vals:
    deviation = (max(sm_vals) - min(sm_vals)) / max(sm_vals) * 100
    print(f"\n  Отклонение spectral_mean во времени: {deviation:.2f}%")
    if deviation < 5:
        print("  ✓ F(s) СТАБИЛЬНА — топология не менялась через время")
        print("  → это означает: вселенная самоподобна не только в пространстве, но и во времени")
    else:
        print("  ~ F(s) меняется — видна эволюция топологии")

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Машина времени — эволюция F(s) по красному смещению z",
             fontsize=13, color="white")

z_vals  = [m["z_mid"] for m in slice_metrics.values()]
sm_vals = [m["spectral_mean"] for m in slice_metrics.values()]
cl_vals = [m["clustering"] for m in slice_metrics.values()]

for ax, vals, ylabel, color, cell_ref in [
    (ax1, sm_vals, "spectral_mean λ", "#5DCAA5", 0.9815),
    (ax2, cl_vals, "clustering coeff", "#EF9F27", 0.9751),
]:
    ax.set_facecolor("#0d1117")
    ax.plot(z_vals, vals, "o-", color=color, linewidth=2.5,
            markersize=12, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.8)
    for z, v in zip(z_vals, vals):
        ax.annotate(f"{v:.4f}", (z, v),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, color=color, fontweight="bold")
    ax.axhline(cell_ref, color="#7F77DD", linestyle="--",
               linewidth=1.2, label=f"клетка HeLa = {cell_ref}")
    ax.set_xlabel("Красное смещение z\n← прошлое          настоящее →",
                  color="#888888")
    ax.set_ylabel(ylabel, color="#888888")
    ax.set_title(ylabel, color="white", fontsize=11)
    ax.tick_params(colors="#888888")
    ax.invert_xaxis()
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("time_machine.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("\n✓ time_machine.png сохранён")
