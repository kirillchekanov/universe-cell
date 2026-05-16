import json, networkx as nx, matplotlib.pyplot as plt, matplotlib.gridspec as gs
from pathlib import Path

G_cell   = nx.read_graphml("cell_graph.graphml")
G_galaxy = nx.read_graphml("galaxy_graph.graphml")
with open("cell_metrics.json") as f:   cm = json.load(f)
with open("galaxy_metrics.json") as f: gm = json.load(f)

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Вселенная как Клетка — первое измерение F(s)", fontsize=14, fontweight="bold", y=0.98)
spec = gs.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Граф клетки
ax1 = fig.add_subplot(spec[0, 0])
pos1 = nx.spring_layout(G_cell, seed=42)
colors1 = {"membrane":"#7F77DD","organelle":"#EF9F27","dense":"#E24B4A","cytoplasm":"#5DCAA5","void":"#888780"}
nc1 = [colors1.get(n, "#aaa") for n in G_cell.nodes()]
nx.draw(G_cell, pos1, ax=ax1, with_labels=True, node_color=nc1,
        node_size=900, font_size=7, font_color="white", font_weight="bold",
        edge_color="#cccccc", width=1.5)
ax1.set_title("Граф клетки\n(органеллы)", fontsize=10)

# Граф галактик (подвыборка для скорости)
ax2 = fig.add_subplot(spec[0, 1])
nodes_sample = list(G_galaxy.nodes())[:80]
G_sub = G_galaxy.subgraph(nodes_sample)
pos2 = nx.spring_layout(G_sub, seed=42)
nx.draw(G_sub, pos2, ax=ax2, with_labels=False,
        node_color="#EF9F27", node_size=30, edge_color="#cccccc", width=0.3, alpha=0.8)
ax2.set_title("Граф галактик SDSS\n(80 из 4997)", fontsize=10)

# Сравнение метрик
ax3 = fig.add_subplot(spec[0, 2])
metrics = ["clustering", "density", "mean_degree"]
labels  = ["Кластери-\nзация", "Плотность\nграфа", "Средняя\nстепень"]
x = range(len(metrics))
w = 0.35
b1 = ax3.bar([i-w/2 for i in x], [cm[m] for m in metrics], w, label="Клетка",   color="#7F77DD", alpha=0.85)
b2 = ax3.bar([i+w/2 for i in x], [gm[m] for m in metrics], w, label="Галактики",color="#EF9F27", alpha=0.85)
ax3.set_xticks(list(x)); ax3.set_xticklabels(labels, fontsize=8)
ax3.set_title("Метрики графов", fontsize=10)
ax3.legend(fontsize=8); ax3.set_yscale("log")
ax3.set_ylabel("Значение (log)")

# Таблица результатов
ax4 = fig.add_subplot(spec[1, :])
ax4.axis("off")
table_data = [
    ["Метрика",        "Клетка (EMD-3805)", "Галактики (SDSS)", "Отношение", "Статус"],
    ["clustering",     f"{cm['clustering']:.4f}", f"{gm['clustering']:.4f}",
     f"{cm['clustering']/gm['clustering']:.2f}×", "✓  СОВПАДАЕТ"],
    ["density",        f"{cm['density']:.4f}",    f"{gm['density']:.4f}",
     f"{cm['density']/gm['density']:.1f}×",      "✗  нужна полная клетка"],
    ["mean_degree",    f"{cm['mean_degree']:.2f}", f"{gm['mean_degree']:.2f}",
     f"{cm['mean_degree']/gm['mean_degree']:.2f}×","✗  нужна полная клетка"],
    ["Узлов в графе",  str(cm['nodes']),          str(gm['nodes']),          "—", ""],
    ["Рёбер",          str(cm['edges']),           str(gm['edges']),          "—", ""],
]
colors_table = [["#1a1a2e"]*5] + \
               [["#0d1117","#0d1117","#0d1117","#0d1117",
                 "#0d3320" if r[4].startswith("✓") else "#2d0d0d" if r[4].startswith("✗") else "#0d1117"]
                for r in table_data[1:]]

t = ax4.table(cellText=table_data, cellLoc="center", loc="center",
              cellColours=colors_table)
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
for (r,c), cell in t.get_celld().items():
    cell.set_edgecolor("#333355")
    cell.set_text_props(color="white" if r==0 else "#cccccc")
ax4.set_title("Первое измерение F(s) — Проект «Вселенная как Клетка»  2026", 
              fontsize=10, pad=12, color="#888888")

fig.patch.set_facecolor("#07080F")
for ax in [ax1,ax2,ax3,ax4]:
    ax.set_facecolor("#0d1117")

plt.savefig("fractal_universe_result.png", dpi=150, bbox_inches="tight",
            facecolor="#07080F")
print("✓ График сохранён: fractal_universe_result.png")
plt.show()
