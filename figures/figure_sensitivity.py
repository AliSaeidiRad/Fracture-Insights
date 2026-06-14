import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Sensitivity Analysis Data
# ==========================

alpha = np.array(
    [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
)

silhouette = np.array(
    [
        0.3999895453453064,
        0.45790502429008484,
        0.26366478204727173,
        0.39153414964675903,
        0.3291137218475342,
        0.18930673599243164,
        0.18102750182151794,
        0.21988385915756226,
        0.12975579500198364,
        0.1270725131034851,
        0.11808037757873535,
        0.04586557298898697,
        0.08933839201927185,
        0.09335707873106003,
        0.08739577233791351,
        0.09167831391096115,
        0.13041724264621735,
        0.09684713184833527,
        0.10820803791284561,
        0.13912485539913177,
        0.147165909409523,
    ]
)

davies_bouldin = np.array(
    [
        1.0278562209497966,
        0.8835621694424229,
        1.2533317881464137,
        1.0774458925152468,
        1.3569170185158725,
        1.4892325455945539,
        1.5406614599419894,
        1.4428876204850265,
        1.9962037612512338,
        1.81496480915854,
        1.8471820911928494,
        2.161070393843754,
        1.9135486630501046,
        1.944839580228503,
        2.0209586758123486,
        2.0193212583925657,
        1.4668727784365638,
        1.9180049733148974,
        1.8626475643630294,
        1.4254151110397044,
        1.7001852125591126,
    ]
)

calinski = np.array(
    [
        37.474769592285156,
        44.38203430175781,
        27.555662155151367,
        30.95620346069336,
        22.52574348449707,
        9.191998481750488,
        8.786787986755371,
        12.49940299987793,
        8.103928565979004,
        8.054288864135742,
        8.228021621704102,
        6.212081432342529,
        7.294126987457275,
        7.215035438537598,
        6.853437423706055,
        6.928976535797119,
        7.7210516929626465,
        7.379587173461914,
        7.306314468383789,
        7.9815521240234375,
        9.001936912536621,
    ]
)

clusters = np.array([5, 6, 6, 5, 5, 6, 6, 4, 2, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 3, 4])

# ==========================
# Best configuration
# ==========================

best_idx = np.argmax(silhouette)
best_alpha = alpha[best_idx]

# ==========================
# Plot
# ==========================

fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# --------------------------
# Silhouette
# --------------------------

ax = axs[0, 0]
ax.plot(alpha, silhouette, marker="o", linewidth=2)
ax.scatter(alpha[best_idx], silhouette[best_idx], s=200, marker="*")
ax.annotate(
    f"Best\nα={best_alpha:.2f}",
    (alpha[best_idx], silhouette[best_idx]),
    xytext=(15, 15),
    textcoords="offset points",
)
ax.set_title("Silhouette Score (Higher is Better)")
ax.set_xlabel("Procrustes Weight α")
ax.set_ylabel("Silhouette")
ax.grid(True, alpha=0.3)

# --------------------------
# Davies-Bouldin
# --------------------------

ax = axs[0, 1]
ax.plot(alpha, davies_bouldin, marker="o", linewidth=2)
best_db = np.argmin(davies_bouldin)

ax.scatter(alpha[best_db], davies_bouldin[best_db], s=200, marker="*")

ax.set_title("Davies–Bouldin Index (Lower is Better)")
ax.set_xlabel("Procrustes Weight α")
ax.set_ylabel("DBI")
ax.grid(True, alpha=0.3)

# --------------------------
# Calinski-Harabasz
# --------------------------

ax = axs[1, 0]
ax.plot(alpha, calinski, marker="o", linewidth=2)

best_ch = np.argmax(calinski)

ax.scatter(alpha[best_ch], calinski[best_ch], s=200, marker="*")

ax.set_title("Calinski–Harabasz Score (Higher is Better)")
ax.set_xlabel("Procrustes Weight α")
ax.set_ylabel("CH Score")
ax.grid(True, alpha=0.3)

# --------------------------
# Optimal Clusters
# --------------------------

ax = axs[1, 1]
ax.step(alpha, clusters, where="mid", linewidth=2)
ax.scatter(alpha, clusters, s=50)

ax.set_title("Optimal Number of Clusters")
ax.set_xlabel("Procrustes Weight α")
ax.set_ylabel("Clusters")
ax.grid(True, alpha=0.3)

# --------------------------
# Global title
# --------------------------

fig.suptitle(
    "Sensitivity Analysis of Similarity Metric Weighting\n"
    "(α = Procrustes, β = Fréchet, α + β = 1)",
    fontsize=16,
    fontweight="bold",
)

plt.savefig("sensitivity_analysis_4panel.tiff", dpi=600, bbox_inches="tight")

plt.show()
