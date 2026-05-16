import mrcfile, networkx as nx, numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

print("Читаю клетку HeLa (244 MB)...")
with mrcfile.open("emd_11756.map", permissive=True) as mrc:
    data = mrc.data.copy()
    vox  = float(mrc.voxel_size.x)
print(f"  ✓ размер: {data.shape}, воксель: {vox:.2f} Å")
real_nm = np.array(data.shape) * vox / 10
print(f"  ✓ реальный размер: {real_nm[0]:.0f}×{real_nm[1]:.0f}×{real_nm[2]:.0f} нм")

print("\nСегментирую органеллы...")
norm = (data - data.min()) / (data.max() - data.min())

# Более детальная сегментация для полной клетки
zones = {
    "outer_membrane": (0.82, 1.00),
    "inner_membrane": (0.68, 0.82),
    "organelle_body": (0.55, 0.68),
    "ribosomes":      (0.45, 0.55),
    "cytoplasm":      (0.25, 0.45),
    "void":           (0.00, 0.25),
}

organelles = {}
for name, (lo, hi) in zones.items():
    mask = (norm >= lo) & (norm < hi)
    cnt  = int(mask.sum())
    # Находим связные компоненты — каждый отдельный "кусок" = отдельный узел
    labeled, n_comp = ndimage.label(mask)
    coords = np.argwhere(mask)
    centroid = tuple(coords.mean(axis=0).astype(int)) if len(coords) else (0,0,0)
    organelles[name] = {
        "count": cnt, "fraction": cnt/data.size,
        "components": n_comp, "centroid": centroid
    }
    print(f"  {name:<20} {cnt:>10} вокс  {cnt/data.size*100:.1f}%  ({n_comp} компонент)")

print("\nСтрою граф (узел = каждая связная компонента)...")
G = nx.Graph()

# Добавляем узлы — каждая связная область отдельно
node_id = 0
node_map = {}
for zone_name, props in organelles.items():
    mask = (norm >= zones[zone_name][0]) & (norm < zones[zone_name][1])
    labeled, n_comp = ndimage.label(mask)
    
    # Берём топ-50 компонент по размеру (иначе миллионы мелких)
    sizes = ndimage.sum(mask, labeled, range(1, min(n_comp+1, 51)))
    top_labels = np.argsort(sizes)[::-1][:50] + 1
    
    for lbl in top_labels:
        comp_mask = labeled == lbl
        comp_coords = np.argwhere(comp_mask)
        if len(comp_coords) < 10:
            continue
        centroid = tuple(comp_coords.mean(axis=0).astype(int))
        size = int(comp_mask.sum())
        G.add_node(node_id, zone=zone_name, centroid=centroid, size=size)
        node_map[node_id] = centroid
        node_id += 1

print(f"  ✓ узлов: {G.number_of_nodes()}")

# Рёбра — близость центроидов
print("  Строю рёбра...")
nodes = list(G.nodes(data=True))
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        c1 = np.array(nodes[i][1]["centroid"])
        c2 = np.array(nodes[j][1]["centroid"])
        dist = float(np.linalg.norm(c1 - c2))
        if dist < 80:  # вокселей
            G.add_edge(nodes[i][0], nodes[j][0], weight=round(1/(dist+1), 4))

print(f"  ✓ рёбер: {G.number_of_edges()}")

# Метрики
print("\nСчитаю метрики...")
degrees = [d for _, d in G.degree()]
hela_metrics = {
    "nodes":       G.number_of_nodes(),
    "edges":       G.number_of_edges(),
    "clustering":  round(nx.average_clustering(G), 4),
    "density":     round(nx.density(G), 4),
    "mean_degree": round(float(np.mean(degrees)), 2),
    "std_degree":  round(float(np.std(degrees)), 2),
}

with open("hela_metrics.json", "w") as f:
    json.dump(hela_metrics, f, indent=2)
nx.write_graphml(G, "hela_graph.graphml")

# Сравнение
print("\nЗагружаю данные галактик...")
with open("galaxy_metrics.json") as f:
    gm = json.load(f)
with open("cell_metrics.json") as f:
    cm_small = json.load(f)

print("\n" + "="*65)
print(f"{'Метрика':<22} {'Митохондрия':>12} {'HeLa':>12} {'Галактики':>12}")
print("-"*65)
for k in ["clustering", "density", "mean_degree"]:
    a = cm_small.get(k, 0)
    b = hela_metrics.get(k, 0)
    g = gm.get(k, 0)
    flag_b = "✓" if g and 0.1 < b/g < 10 else "~"
    print(f"  {k:<20} {a:>12.4f} {flag_b}{b:>11.4f} {g:>12.4f}")
print("="*65)

print(f"\n  Митохондрия:  {cm_small['nodes']} узлов")
print(f"  HeLa клетка: {hela_metrics['nodes']} узлов  ← намного лучше!")
print(f"  Галактики:   {gm['nodes']} узлов")

print("\n✓ ГОТОВО!")
print("  → hela_graph.graphml")
print("  → hela_metrics.json")
