import re
import os
import glob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pc.curve import Curve
from pc.analysis import Analysis


def plot_figure_cmb(results_dir):
    color_curves = {
        "A1": {
            "C1": "red",
            "C2": "gold",
            "C3": "purple",
            "C4": "yellowgreen",
            "C5": "deepskyblue",
            "C6": "darkorange",
            "C7": "hotpink",
            "C8": "turquoise",
            "C9": "sienna",
            "C10": "navy",
            "C11": "limegreen",
            "C12": "darkviolet",
        },
        "A2": {
            "C1": "darkred",
            "C2": "olive",
            "C3": "mediumvioletred",
            "C4": "darkgreen",
            "C5": "steelblue",
            "C6": "chocolate",
            "C7": "deeppink",
            "C8": "teal",
            "C9": "peru",
            "C10": "midnightblue",
            "C11": "forestgreen",
            "C12": "indigo",
        },
        "A3": {
            "C1": "orangered",
            "C2": "orange",
            "C3": "crimson",
            "C4": "slateblue",
            "C5": "dodgerblue",
            "C6": "coral",
            "C7": "fuchsia",
            "C8": "mediumturquoise",
            "C9": "saddlebrown",
            "C10": "royalblue",
            "C11": "springgreen",
            "C12": "blueviolet",
        },
    }

    list_dirs = []

    beta = None
    dir_frechet_only = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_frechet_*")):
        if os.path.isdir(found_dir):
            if "procrustes" not in found_dir:
                dir_frechet_only = found_dir
                beta = re.findall(r"\d+\.\d+|\d+", dir_frechet_only)
                beta = [float(num) for num in beta]
                beta = beta[0]
                list_dirs.append([dir_frechet_only, 0.0, beta])

    alpha = None
    dir_procrustes_only = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_procrustes_*")):
        if os.path.isdir(found_dir):
            if "frechet" not in found_dir:
                dir_procrustes_only = found_dir
                alpha = re.findall(r"\d+\.\d+|\d+", dir_procrustes_only)
                alpha = [float(num) for num in alpha]
                alpha = alpha[0]
                list_dirs.append([dir_procrustes_only, alpha, 0.0])

    coefs = [None, None]  # [alpha, beta]
    dir_both_analysis = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_procrustes_*_frechet_*")):
        if os.path.isdir(found_dir):
            dir_both_analysis = found_dir
            coefs = re.findall(r"\d+\.\d+|\d+", dir_both_analysis)
            coefs = [float(num) for num in coefs]
            list_dirs.append([dir_both_analysis, coefs[0], coefs[1]])

    fig = plt.figure(
        dpi=1200,
        figsize=(
            len(glob.glob(os.path.join(list_dirs[0][0], "*"))) + 16,
            len(list_dirs) + 4,
        ),
        tight_layout=True,
    )

    for idx, analysis in enumerate(list_dirs):
        color_curves_analysis = color_curves[list(color_curves.keys())[idx]]

        analysis_dir = analysis[0]
        analysis_alpha = analysis[1]
        analysis_beta = analysis[2]

        curve_list_lens = []
        curve_list = []

        n_cluster_groups = len(glob.glob(os.path.join(analysis_dir, "*")))

        for i in range(1, 1 + n_cluster_groups):
            cluster_curve_list = [
                Curve(curve).curve().to_numpy()
                for curve in glob.glob(os.path.join(analysis[0], f"*{i}", "*.txt"))
            ]
            curve_list_lens.append(len(cluster_curve_list))
            curve_list += cluster_curve_list

            ax = fig.add_subplot(
                len(list_dirs),
                1 + n_cluster_groups,
                idx * (1 + n_cluster_groups) + i,
                projection="3d",
            )

            ax.set_xlim(-20, 20)
            ax.set_xticks([-20, 0, 20])

            ax.set_ylim(0, 40)
            ax.set_yticks([0, 20, 40])

            ax.set_zlim(0, 30)
            ax.set_zticks([0, 15, 30])

            if i == 1:
                ax.text(
                    x=ax.get_xlim()[0] + 110,
                    y=ax.get_ylim()[0],
                    z=ax.get_zlim()[0] / 2 + ax.get_zlim()[1] / 2 + 10,
                    s=rf"$\alpha$ = {analysis_alpha}, $\beta$ = {analysis_beta}",
                    fontsize=15,
                    fontdict={"fontname": "Times New Roman"},
                )

            for curve in cluster_curve_list:
                ax.plot(
                    curve[:, 0],
                    curve[:, 1],
                    curve[:, 2],
                    c=color_curves_analysis[f"C{i}"],
                )
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
                ax.set_zlabel("z (mm)")
                ax.view_init(azim=70, elev=20)
                if idx == 0:
                    if i == 1:
                        ax.set_title(
                            f"Cluster I", fontdict={"fontname": "Times New Roman"}
                        )
                    elif i == 2:
                        ax.set_title(
                            f"Cluster II", fontdict={"fontname": "Times New Roman"}
                        )
                    elif i == 3:
                        ax.set_title(
                            f"Cluster II", fontdict={"fontname": "Times New Roman"}
                        )
                    elif i == 4:
                        ax.set_title(
                            f"Cluster IV", fontdict={"fontname": "Times New Roman"}
                        )

        if not os.path.exists(
            f".tmp/similarity_matirx_p_{analysis_alpha}_f_{analysis_beta}.npy"
        ):
            analyzer = Analysis(
                curve_list,
                ["procrustes", "frechet"],
                [analysis_alpha, analysis_beta],
            )
            _, _ = analyzer.analysis().cluster(n_cluster_groups)
            matrix = analyzer.similarity_matrix

            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")

            np.save(
                f".tmp/similarity_matirx_p_{analysis_alpha}_f_{analysis_beta}.npy",
                matrix,
            )
        else:
            matrix = np.load(
                f".tmp/similarity_matirx_p_{analysis_alpha}_f_{analysis_beta}.npy"
            )

        ax = fig.add_subplot(
            len(list_dirs), 1 + n_cluster_groups, (idx + 1) * (n_cluster_groups + 1)
        )

        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        sns.heatmap(
            matrix,
            annot=False,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Similarity", "aspect": 10},
            xticklabels=False,
            yticklabels=False,
            vmin=np.min(matrix),
            vmax=np.max(matrix),
            ax=ax,
            square=True,
        )
        if idx == 0:
            ax.set_title("Similarity Matrix")

        for i in range(len(curve_list_lens)):
            x1 = y1 = sum(curve_list_lens[:i])
            x2 = y2 = sum(curve_list_lens[: i + 1])
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color_curves_analysis[f"C{i+1}"],
                facecolor="none",
            )
            ax.add_patch(rect)

    fig.savefig("figure_cmb.tif", format="tif")
