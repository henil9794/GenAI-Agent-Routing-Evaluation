import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Linguistic Failure Analysis — Agentic RAG Routing",
             fontsize=14, fontweight='bold', y=0.98)

PURPLE = "#534AB7"
TEAL   = "#0F6E56"
CORAL  = "#993C1D"
TEAL2  = "#5DCAA5"
BLUE   = "#378ADD"
GRAY   = "#888780"

# ── Chart 1: Failures by linguistic pattern ──────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
patterns = ["Missing\ntemporal anchor", "Recency signal\nignored", "Dual-source\nambiguity"]
counts   = [8, 6, 2]
colors   = [PURPLE, TEAL, CORAL]
bars = ax1.bar(patterns, counts, color=colors, edgecolor='none', width=0.5)
for bar, val in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 10)
ax1.set_ylabel("Number of failures", fontsize=10)
ax1.set_title("Failures by linguistic pattern", fontsize=11, fontweight='bold', pad=10)
ax1.spines[['top', 'right']].set_visible(False)
ax1.yaxis.grid(True, color='#e0e0e0', zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(axis='x', labelsize=9)

# ── Chart 2: Failures by ambiguity tier (donut) ───────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
tier_vals   = [7, 11]
tier_colors = [PURPLE, TEAL2]
ax2.pie(tier_vals, labels=None, colors=tier_colors,
        startangle=90, wedgeprops=dict(width=0.45))
ax2.set_title("Failures by ambiguity tier", fontsize=11, fontweight='bold', pad=10)
legend_patches = [
    mpatches.Patch(color=PURPLE, label="Tier 1 — 7 (39%)"),
    mpatches.Patch(color=TEAL2,  label="Tier 2 — 11 (61%)")
]
ax2.legend(handles=legend_patches, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)

# ── Chart 3: Failures by router (horizontal bar) ──────────────────────────────
ax3 = fig.add_subplot(2, 1, 2)
routers       = ['rule_based', 'always_local', 'zero_shot_llm',
                 'few_shot_llm', 'cot_llm', 'langgraph_agent']
router_vals   = [3, 3, 3, 3, 3, 3]
router_colors = [GRAY, GRAY, BLUE, BLUE, BLUE, BLUE]
y_pos = np.arange(len(routers))
bars3 = ax3.barh(y_pos, router_vals, color=router_colors, edgecolor='none', height=0.5)
for bar, val in zip(bars3, router_vals):
    ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=10, fontweight='bold')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(routers, fontsize=10)
ax3.set_xlim(0, 5)
ax3.set_xlabel("Number of failures", fontsize=10)
ax3.set_title("Failures by router", fontsize=11, fontweight='bold', pad=10)
ax3.spines[['top', 'right']].set_visible(False)
ax3.xaxis.grid(True, color='#e0e0e0', zorder=0)
ax3.set_axisbelow(True)
legend_patches2 = [
    mpatches.Patch(color=BLUE, label='LLM-based routers'),
    mpatches.Patch(color=GRAY, label='Non-LLM routers')
]
ax3.legend(handles=legend_patches2, loc='lower right', fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("linguistic_failure_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: linguistic_failure_analysis.png")