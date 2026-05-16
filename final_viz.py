import json, networkx as nx, matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

with open("cell_metrics.json")   as f: cm = json.load(f)
with open("hela_metrics.json")   as f: hm = json.load(f)
with open("galaxy_metrics.json") as f: gm = json.load(f)

G_mito   = nx.read_graphml("cell_graph.graphml")
G_hela   = nx.read_graphml("hela_graph.graphml")
G_galaxy = nx.read_graphml("galaxy_graph.graphml")

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Вселенная как Клетка — F(s) на трёх датасетах",
             fontsize=15, fontweight="bold", color="white", y=0.98)

spec = gs.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

# Графы
for ax_idx, (G, title, color) in enumerate([
    (G_mito,   "Митохондрия\n(5 узлов)", "#7F77DD"),
    (G_hela,   "Клетка HeLa\n(54 узла)", "#5DCAA5"),
    (G_galaxy.subgraph(list(G_galaxy.nodes())[:80]),
               "Галактики SDSS\n(80 из 4997)", "#EF9F27"),
]):
    ax = fig.add_subplot(spec[0, ax_idx])
    ax.set_facecolor("#0d1117")
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=False,
            node_color=color, node_size=60 if ax_idx==2 else 200,
            edge_color="#ffffff", width=0.5, alpha=0.85)
    ax.set_title(title, fontsize=10, color="white", pad=8)

# Сравнение метрик
ax_bar = fig.add_subplot(spec[1, :])
ax_bar.set_facecolor("#0d1117")

metrics  = ["clustering", "density", "mean_degree"]
labels   = ["Коэф.\nкластеризации", "Плотность\nграфа", "Средняя\nстепень"]
x = np.arange(len(metrics))
w = 0.25

bars = [
    (cm, "Митохондрия (5 узлов)", "#7F77DD"),
    (hm, "HeLa (54 узла)",        "#5DCAA5"),
    (gm, "Галактики (4997)",       "#EF9F27"),
]
for i, (data, label, color) in enumerate(bars):
    vals = [data.get(m, 0) for m in metrics]
    b = ax_bar.bar(x + (i-1)*w, vals, w, label=label, color=color, alpha=0.88)
    for bar, val in zip(b, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=10, color="white")
ax_bar.set_ylabel("Значение", color="white")
ax_bar.tick_params(colors="white")
ax_bar.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
ax_bar.set_title(
    f"clustering: мито={cm['clustering']:.3f}  HeLa={hm['clustering']:.3f}  "
    f"галактики={gm['clustering']:.3f}  →  стабильное совпадение",
    fontsize=10, color="#5DCAA5", pad=10)
for spine in ax_bar.spines.values():
    spine.set_edgecolor("#333355")
ax_bar.set_facecolor("#0d1117")

plt.savefig("fractal_universe_v2.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("✓ Сохранено: fractal_universe_v2.png")
plt.show()
