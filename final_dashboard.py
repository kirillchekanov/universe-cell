import json, math
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

print("ФИНАЛЬНЫЙ ДАШБОРД — Universe as Cell\n")

with open("regional_forecast.json") as f: regional = json.load(f)
with open("natal_forecast.json") as f:   natal = json.load(f)
with open("final_forecast.json") as f:   global_f = json.load(f)

DATES = ["2026-08-29","2027-03-21","2027-09-23","2027-12-22","2028-12-16"]
LABELS = ["Aug 2026","Mar 2027","Sep 2027","Dec 2027","Dec 2028"]

fig = plt.figure(figsize=(18, 14), facecolor="#07080F")
fig.suptitle("Universe as Cell — Gravitational Prediction Model\n"
             f"Walk-forward F1=0.381 · 78 features · 400 events 1950-2025",
             color="white", fontsize=14, y=0.98)

# Цвета
COLORS = {
    "geopolitical":     "#EF4444",
    "economic_crisis":  "#EF9F27",
    "natural_disaster": "#3B82F6",
    "natural_disaster_asia":    "#60A5FA",
    "natural_disaster_europe":  "#93C5FD",
    "natural_disaster_americas":"#BFDBFE",
    "epidemic":         "#A855F7",
    "tech_breakthrough":"#27EFB5",
    "neutral":          "#374151",
}

# ══ ГРАФИК 1: Глобальный прогноз (стакан) ══
ax1 = fig.add_axes([0.04, 0.68, 0.44, 0.26])
ax1.set_facecolor("#0d0d1a")
ax1.set_title("Global forecast — event type probability", color="white", fontsize=11)

x = np.arange(len(DATES))
width = 0.6
bottom = np.zeros(len(DATES))

gf = global_f["forecast"]
cats = ["geopolitical","economic_crisis","epidemic","tech_breakthrough","natural_disaster"]
for cat in cats:
    vals = [gf.get(d,{}).get(cat,0) for d in DATES]
    ax1.bar(x, vals, width, bottom=bottom, color=COLORS.get(cat,"#888"),
            alpha=0.85, label=cat.replace("_"," "))
    bottom += np.array(vals)

ax1.set_xticks(x); ax1.set_xticklabels(LABELS, color="#ccc", fontsize=9)
ax1.set_ylabel("P(event)", color="#888"); ax1.set_ylim(0, 0.7)
ax1.tick_params(colors="#888")
ax1.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", ncol=2)
for sp in ax1.spines.values(): sp.set_edgecolor("#333355")

# ══ ГРАФИК 2: Региональный прогноз ══
ax2 = fig.add_axes([0.54, 0.68, 0.44, 0.26])
ax2.set_facecolor("#0d0d1a")
ax2.set_title("Regional forecast — Asia / Europe / Americas", color="white", fontsize=11)

rf = regional["forecast"]
reg_cats = ["natural_disaster_asia","natural_disaster_europe",
            "natural_disaster_americas","geopolitical","epidemic","tech_breakthrough"]
bottom2 = np.zeros(len(DATES))
for cat in reg_cats:
    vals = [rf.get(d,{}).get(cat,0) for d in DATES]
    ax2.bar(x, vals, width, bottom=bottom2, color=COLORS.get(cat,"#888"),
            alpha=0.85, label=cat.replace("_"," "))
    bottom2 += np.array(vals)

ax2.set_xticks(x); ax2.set_xticklabels(LABELS, color="#ccc", fontsize=9)
ax2.set_ylabel("P(event)", color="#888"); ax2.set_ylim(0, 0.8)
ax2.tick_params(colors="#888")
ax2.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", ncol=2)
for sp in ax2.spines.values(): sp.set_edgecolor("#333355")

# ══ ГРАФИК 3: Натальная дельта стран ══
ax3 = fig.add_axes([0.04, 0.36, 0.92, 0.28])
ax3.set_facecolor("#0d0d1a")
ax3.set_title("Natal delta by country — lower = closer to natal configuration",
              color="white", fontsize=11)

nf = natal["forecast"]
countries_show = ["USA","Russia","China","EU","UK","Germany","France",
                  "Japan","Israel","Ukraine","Iran","Turkey","India","NATO","UN"]
country_x = np.arange(len(countries_show))

for di, (date, label) in enumerate(zip(DATES, LABELS)):
    deltas = [nf.get(date,{}).get(c, 90) for c in countries_show]
    offset = (di - 2) * 0.15
    alpha = 1.0 if di in [1,2,3] else 0.4  # выделяем 2027
    color = ["#6B5BDB","#27EFB5","#EF9F27","#EF4444","#A855F7"][di]
    ax3.bar(country_x + offset, deltas, 0.15, color=color,
            alpha=alpha, label=label)

ax3.set_xticks(country_x)
ax3.set_xticklabels(countries_show, color="#ccc", fontsize=8, rotation=30, ha="right")
ax3.set_ylabel("Δ° от натального", color="#888")
ax3.axhline(70, color="#EF9F27", lw=0.8, linestyle="--", alpha=0.5)
ax3.set_ylim(0, 120)
ax3.tick_params(colors="#888")
ax3.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", labelcolor="white", ncol=5)
for sp in ax3.spines.values(): sp.set_edgecolor("#333355")

# ══ СВОДНАЯ ТАБЛИЦА ══
ax4 = fig.add_axes([0.04, 0.04, 0.92, 0.28])
ax4.set_facecolor("#0d0d1a")
ax4.set_title("Summary — consolidated micro-forecast", color="white", fontsize=11)
ax4.axis("off")

# Собираем топ-3 страны с минимальной дельтой для каждой даты
summary_data = []
for date, label in zip(DATES, LABELS):
    # Глобальный топ
    g = gf.get(date, {})
    top_global = max([(v,k) for k,v in g.items() if k!="neutral"], default=(0,"?"))

    # Региональный топ
    r = rf.get(date, {})
    top_regional = max([(v,k) for k,v in r.items() if k not in ("neutral","")], default=(0,"?"))

    # Натальный топ-3
    n = nf.get(date, {})
    top_natal = sorted(n.items(), key=lambda x:x[1])[:3]
    natal_str = " · ".join([f"{c}({d:.0f}°)" for c,d in top_natal])

    summary_data.append([label,
                          f"{top_global[1].replace('_',' ')} {top_global[0]:.0%}",
                          f"{top_regional[1].replace('_',' ')} {top_regional[0]:.0%}",
                          natal_str])

cols = ["Дата", "Тип (глобально)", "Регион", "Страны (↓ дельта)"]
col_widths = [0.10, 0.18, 0.22, 0.50]
col_x = [0.01, 0.11, 0.29, 0.51]

# Заголовки
for col, cx in zip(cols, col_x):
    ax4.text(cx, 0.92, col, color="#27EFB5", fontsize=9, fontweight="bold",

ax4.axhline(0.88, color="#333355", lw=0.8, xmin=0, xmax=1)

# Данные
for ri, row in enumerate(summary_data):
    y_pos = 0.78 - ri * 0.17
    bg_color = "#0f1020" if ri % 2 == 0 else "#0d0d1a"
    ax4.add_patch(mpatches.FancyBboxPatch((0, y_pos-0.05), 1, 0.15,
        boxstyle="round,pad=0.01", facecolor=bg_color, edgecolor="none",
    for val, cx in zip(row, col_x):
        color = "#EF9F27" if ri == 2 else "#ffffff"  # выделяем осень 2027
        ax4.text(cx, y_pos+0.03, val, color=color, fontsize=8,
                transform=ax4.transAxes, va="center")

plt.savefig("universe_cell_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("✓ universe_cell_dashboard.png")

# Git
import subprocess
result = subprocess.run(
    ["git", "add", "regional_model.py", "regional_forecast.json",
     "natal_countries.py", "natal_forecast.json", "final_dashboard.py",
     "universe_cell_dashboard.png"],
    cwd="/Users/kirillchekanov/universe-cell", capture_output=True, text=True
)
result2 = subprocess.run(
    ["git", "commit", "-m",
     "Add: regional + natal forecast, final dashboard — micro-predictions 2026-2028"],
    cwd="/Users/kirillchekanov/universe-cell", capture_output=True, text=True
)
result3 = subprocess.run(
    ["git", "push"],
    cwd="/Users/kirillchekanov/universe-cell", capture_output=True, text=True
)
print(result2.stdout.strip())
print(result3.stdout.strip() or result3.stderr.strip())
