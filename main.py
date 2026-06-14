import os
import math
import glob
import shutil
import argparse
import warnings
from typing import List

import vtk
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pc import Analysis, Curve

PRECISION = 0.1
CLUSTER_RESULT_PREFIX = "Cluster_Result"


def complete_analysis(args):
    list_files = glob.glob(os.path.join(args.dir_curves, "*.txt"))
    array = []
    with tqdm.tqdm(total=len(list_files), desc="Loading Curves") as pbar:
        for curve in list_files:
            array.append(Curve(curve).curve().to_numpy())
            pbar.update()
    t = args.t
    alpha, beta = args.alpha, args.beta

    analysis = Analysis(
        array,
        ["procrustes", "frechet"],
        [alpha, beta],
    )
    _, cluster_labels = analysis.analysis().cluster(t)

    colors = ["r", "g", "b", "k", "y", "cyan", "peru", "green", "navy"]
    c_list = []
    for _p in cluster_labels:
        c_list.append(colors[int(_p) - 1])

    for i in range(1, t + 1):
        print(
            f"cluster {i}: {np.sum(cluster_labels == i)} of {len(cluster_labels)} == {(np.sum(cluster_labels == i)/len(cluster_labels))*100:.2f}%"
        )

    plt.rcParams.update({"font.size": 6})

    n_plots = t + 2  # All curves + clusters + heatmap

    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
    for curve, c in zip(analysis.array, c_list):
        if c:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c=c)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(azim=70, elev=20)

    for i in range(t):
        ax = fig.add_subplot(nrows, ncols, i + 2, projection="3d")
        for curve, c in zip(analysis.array, c_list):
            if colors.index(c) == i:
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c=c)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.view_init(azim=70, elev=20)

    if alpha == 0:
        name = f"frechet_{beta:.2f}"
    elif beta == 0:
        name = f"procrustes_{alpha:.2f}"
    elif alpha != 0 and beta != 0:
        name = f"procrustes_{alpha:.2f}_frechet_{beta:.2f}"

    ax = fig.add_subplot(nrows, ncols, n_plots)
    sns.set_theme(font_scale=0.7)
    matrix = analysis.similarity_matrix
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    sns.heatmap(
        matrix,
        annot=False,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Similarity"},
        xticklabels=False,
        yticklabels=False,
        vmin=np.min(matrix),
        vmax=np.max(matrix),
        ax=ax,
        square=True,
    )

    if os.path.exists(os.path.join(args.output, f"{CLUSTER_RESULT_PREFIX}_{name}")):
        shutil.rmtree(os.path.join(args.output, f"{CLUSTER_RESULT_PREFIX}_{name}"))
    os.makedirs(os.path.join(args.output, f"{CLUSTER_RESULT_PREFIX}_{name}"))

    try:
        _fig_path = os.path.join(args.output, f"{name}_cluster_result.tiff")
        fig.savefig(_fig_path, format="tiff")
        print(f"{_fig_path} exported!!")
    except:
        print(f"couldnt export {_fig_path}!!")

    for file, label in zip(list_files, cluster_labels):
        id = file.split(os.sep)[-1].split(".")[0]
        os.makedirs(
            os.path.join(
                args.output, f"{CLUSTER_RESULT_PREFIX}_{name}", f"Cluster_{label}"
            ),
            exist_ok=True,
        )
        shutil.copy(
            file,
            os.path.join(
                args.output,
                f"{CLUSTER_RESULT_PREFIX}_{name}",
                f"Cluster_{label}",
                f"{id}.txt",
            ),
        )

    return analysis


def calculate_bins(dist: np.ndarray, rule: str = "sturge", mode: int = 1) -> int:
    def sturge_rule(dist: np.ndarray, mode=1):
        if mode == 1:
            bin_count = int(np.ceil(np.log2(len(dist))) + 1)
        elif mode == 2:
            bin_count = int(np.ceil(3.3 * np.log(len(dist))) + 1)
        return bin_count

    def freedman_diaconis_rule(dist: np.ndarray):
        q1 = np.quantile(dist, 0.25)
        q3 = np.quantile(dist, 0.75)
        iqr = q3 - q1
        bin_width = (2 * iqr) / (len(dist) ** (1 / 3))
        bin_count = int(np.ceil((dist.max() - dist.min()) / bin_width))
        return bin_count

    if rule == "sturge":
        return sturge_rule(dist, mode)
    elif rule == "freedman":
        return freedman_diaconis_rule(dist)
    elif rule == "normal":
        return 10
    else:
        raise ValueError(f"{rule} Rule is not valid.")


def create_contours(curves: np.ndarray, rule="sturge"):
    def model(mask):
        points = mask[:, :2]
        kde = gaussian_kde(points.T)
        x_min = MIN["x"]
        y_min = MIN["y"]
        x_max = MAX["x"]
        y_max = MAX["y"]
        _x = np.arange(x_min, x_max, PRECISION)
        _y = np.arange(y_min, y_max, PRECISION)
        X, Y = np.meshgrid(_x, _y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = kde(positions)
        density = density.reshape(X.shape)
        density = (density - density.min()) / (density.max() - density.min())
        return density

    MAX = {
        "x": np.max(curves[:, 0]),
        "y": np.max(curves[:, 1]),
        "z": np.max(curves[:, 2]),
    }
    MIN = {
        "x": np.min(curves[:, 0]),
        "y": np.min(curves[:, 1]),
        "z": np.min(curves[:, 2]),
    }
    histogram = np.histogram(
        curves[:, 2], bins=calculate_bins(curves[:, 2], rule=rule, mode=1)
    )
    LIMITS = []
    contours = []
    for i in tqdm.tqdm(range(1, len(histogram[1])), desc="Creating Contours"):
        upper_limit = (histogram[1][i] + histogram[1][i - 1]) / 2
        lower_limit = histogram[1][i - 1]
        LIMITS.append([lower_limit, upper_limit])
        mask = curves[np.where(curves[:, 2] >= lower_limit)]
        mask = mask[np.where(mask[:, 2] <= upper_limit)]
        contours.append(model(mask))
        lower_limit = upper_limit
        upper_limit = histogram[1][i]
        LIMITS.append([lower_limit, upper_limit])
        mask = curves[np.where(curves[:, 2] >= lower_limit)]
        mask = mask[np.where(mask[:, 2] <= upper_limit)]
        contours.append(model(mask))
    contours.append(model(curves[:, 0:2]))
    return contours, LIMITS, MAX, MIN


def find_precision(number):
    num_str = str(number)
    parts = num_str.split(".")
    if len(parts) == 1:
        return 0
    return len(parts[1])


class set_color:
    def __init__(self, limits, min, max, contours, offset_z, offset_xy, mean, std_dev):
        self.limits = limits
        self.min = min
        self.max = max
        self.contours = contours
        self.offset_z = offset_z
        self.offset_xy = offset_xy
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, x, y, z):
        z_index = None
        for idx, _limits in enumerate(self.limits):
            if z >= _limits[0]:
                if z <= _limits[1]:
                    z_index = idx
                    break
        if z_index is not None:
            if (
                x > self.min["x"] + 2
                and x < self.max["x"] - 2
                and y > self.min["y"] + 2
                and y < self.max["y"] - 2
            ):
                mask = self.contours[z_index - self.offset_z : z_index + self.offset_z]
                if mask:
                    mask = np.concatenate([np.expand_dims(m, 0) for m in mask], 0)
                    # mask = (mask - self.mean)/self.std_dev
                    x_index = round(
                        np.abs(
                            round(self.min["x"], find_precision(PRECISION))
                            - round(x, find_precision(PRECISION))
                        )
                        / PRECISION
                    )
                    y_index = round(
                        np.abs(
                            round(self.min["y"], find_precision(PRECISION))
                            - round(y, find_precision(PRECISION))
                        )
                        / PRECISION
                    )
                    score = np.mean(
                        mask[
                            :,
                            y_index - self.offset_xy : y_index + self.offset_xy,
                            x_index - self.offset_xy : x_index + self.offset_xy,
                        ]
                    )
                    if score is None:
                        score = 0
                    if score <= 0:
                        score = 0
                else:
                    score = 0
            else:
                score = 0
        else:
            score = 0

        score = cm.RdYlBu_r(score)

        return score[0], score[1], score[2]


def plot_stl_with_custom_colors(file_path, set_color, vtk_file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    colors_array = vtk.vtkUnsignedCharArray()
    colors_array.SetNumberOfComponents(1)
    colors_array.SetName("Colors")

    mmin = [np.inf, np.inf, np.inf]
    mmax = [-np.inf, -np.inf, -np.inf]

    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        color = set_color(*point)
        for i in range(3):
            if color[i] > mmax[i]:
                mmax[i] = color[i]
            if color[i] < mmin[i]:
                mmin[i] = color[i]

    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        color = list(set_color(*point))
        for i in range(3):
            color[i] = (color[i] - mmin[i]) / (mmax[i] - mmin[i])
            color[i] = color[i] * 100
        colors_array.InsertNextTuple1(color[0])

    polydata.GetPointData().SetScalars(colors_array)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(vtk_file_path)
    writer.Write()
    print(f"File Exported: {vtk_file_path}")


def export_vtks(args):
    def export(
        list_files: List[str],
        cluster_result: str,
        n_cluster: int,
        vtk_file_path: str,
        stl_file_path: str,
        **kwargs,
    ):
        array = np.array([])
        with tqdm.tqdm(
            total=len(list_files),
            desc=kwargs.get(
                "desc",
                f"Loading Curves for results {cluster_result}, cluster {n_cluster}",
            ),
        ) as pbar:
            for curve in list_files:
                _curve = Curve(curve).curve().to_numpy().reshape(-1, 3)
                array = np.append(array, _curve)
                pbar.update()
        array = array.reshape(-1, 3)
        contours, LIMITS, MAX, MIN = create_contours(array)

        offset_xy = 7
        offset_z = 3
        mean = np.mean(
            np.concatenate(
                [np.expand_dims(contour, 0) for contour in contours]
            ).ravel(),
            0,
        )
        std_dev = np.std(
            np.concatenate(
                [np.expand_dims(contour, 0) for contour in contours]
            ).ravel(),
            0,
        )

        plot_stl_with_custom_colors(
            stl_file_path,
            set_color(LIMITS, MIN, MAX, contours, offset_z, offset_xy, mean, std_dev),
            vtk_file_path,
        )

    output_dir = f"{args.output}_vtks"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for path in glob.glob(
        os.path.join(args.output, f"{CLUSTER_RESULT_PREFIX}_*", "Cluster_*")
    ):
        vtk_file_path = os.path.join(output_dir, f"{path.replace(os.sep, '_')}.vtk")
        if not os.path.exists(vtk_file_path):
            cluster_result = path.split(os.sep)[-2].split(f"{CLUSTER_RESULT_PREFIX}_")[
                -1
            ]
            n_cluster = path.split(os.sep)[-1].split("Cluster_")[-1]

            list_files = glob.glob(os.path.join(path, "*.txt"))
            export(
                list_files, cluster_result, n_cluster, vtk_file_path, args.sample_stl
            )
        else:
            print(f"File already exported: {vtk_file_path}")

    vtk_file_path_all = os.path.join(output_dir, "OUTPUT_ALL")
    list_files = glob.glob(os.path.join(args.dir_curves, "*.txt"))
    if not os.path.exists(vtk_file_path_all):
        export(
            list_files,
            None,
            None,
            vtk_file_path_all,
            args.sample_stl,
            desc="Loading Curves",
        )
    else:
        print(f"File already exported: {vtk_file_path_all}")


def export_plot(args):
    try:
        from figures.figure_cmb import plot_figure_cmb

        plot_figure_cmb(args.output)
    except ImportError as e:
        warnings.warn(
            f"could not import plot_figure_cmb from figurs/figure_cmb.py file\n{e}"
        )

    try:
        from figures.figure_curves import plot_figure_curves

        plot_figure_curves(args.dir_curves)
    except ImportError as e:
        warnings.warn(
            f"could not import plot_figure_curves from figurs/figure_curves.py file\n{e}"
        )

    try:
        from figures.figure_density import plot_figure_density

        plot_figure_density(args.dir_curves)
    except ImportError as e:
        warnings.warn(
            f"could not import plot_figure_density from figurs/figure_density.py file\n{e}"
        )

    try:
        from figures.material_figure_ten_segment import material_figure_ten_segment

        material_figure_ten_segment(
            args.output, args.sample_stl, method=args.mapping_method
        )
    except ImportError as e:
        warnings.warn(
            f"could not import plot_figure_curves from figures.material_figure_ten_segment.py file\n{e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-curves",
        default=os.path.join("Dataset", "Curves"),
        type=str,
        help="Directory to curves.",
    )
    parser.add_argument(
        "--t",
        default=4,
        type=int,
        help="Number of Groups.",
    )
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="Coefficient of procrustes analysis measurements.",
    )
    parser.add_argument(
        "--beta",
        default=1.0,
        type=float,
        help="Coefficient of frechet distance analysis measurements.",
    )
    parser.add_argument(
        "--output", default="output", type=str, help="Directory to save results."
    )
    parser.add_argument(
        "--sample-stl",
        default=os.path.join("Dataset", "Sample", "Sample.stl"),
        type=str,
        help="Path to sample for plotting results on it.",
    )
    parser.add_argument(
        "--only-export",
        action="store_true",
        help="enable it if you just need vtk exports and not doing the analysis.",
    )
    parser.add_argument(
        "--only-plots",
        action="store_true",
        help="enable it if you just need plots of the previous analysis.",
    )
    parser.add_argument(
        "--mapping-method",
        default="nearest",
        type=str,
        help="Mapping method (nearest, ray_casting, sdf)",
    )

    args = parser.parse_args()

    if args.only_export:
        export_vtks(args)

    elif args.only_plots:
        export_plot(args)

    else:
        analysis = complete_analysis(args)
        export_vtks(args)
        export_plot(args)
