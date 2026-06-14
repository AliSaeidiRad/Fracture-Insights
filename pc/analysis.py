import copy
from warnings import warn
from multiprocessing import Pool, cpu_count
from typing import List, Union, Callable, Literal, Optional

import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from pc.similarity import similarity_measurements


def _compute_similarity(args):
    i, j, arr_i, arr_j, measurements, coefficients = args

    sim = similarity_measurements(
        arr_i,
        arr_j,
        measurments=measurements,
        measurments_coefficients=coefficients,
    )

    return i, j, sim


class Analysis:
    def __init__(
        self,
        array: List[np.ndarray],
        measurments: Union[List[str], str, Callable],
        measurments_coefficients: List[float],
        *,
        method: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "complete",
        metric: Literal[
            "braycurtis",
            "canberra",
            "chebyshev",
            "cityblock",
            "correlation",
            "cosine",
            "dice",
            "euclidean",
            "hamming",
            "jaccard",
            "jensenshannon",
            "kulczynski1",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        ] = "euclidean",
        optimal_ordering: bool = False,
        criterion: Literal[
            "inconsistent", "distance", "maxclust", "monocrit", "maxclust_monocrit"
        ] = "maxclust",
        depth: int = 2,
        R: Optional[np.ndarray] = None,
        monocrit: Optional[np.ndarray] = None,
    ) -> None:
        self.original_array = array
        self.array = self.resampling(array)
        self.measurments = measurments
        self.measurments_coefficients = measurments_coefficients
        self.method = method
        self.metric = metric
        self.optimal_ordering = optimal_ordering
        self.criterion = criterion
        self.depth = depth
        self.R = R
        self.monocrit = monocrit

    def resampling(self, array: List[np.ndarray]) -> List[np.ndarray]:
        largest_curve_idx = np.argmax([curve.shape[0] for curve in array])
        target_num_points = array[largest_curve_idx].shape[0]

        resampled_curves = []

        with tqdm.tqdm(total=len(array), desc="Resampling Curves") as pbar:
            for curve in array:
                if curve.shape[0] != target_num_points:
                    f = interp1d(np.linspace(0, 1, curve.shape[0]), curve, axis=0)
                    resampled_curve = f(np.linspace(0, 1, target_num_points))
                    resampled_curves.append(resampled_curve)
                else:
                    resampled_curves.append(curve)
                pbar.update()

        return resampled_curves

    def analysis(self):
        n = len(self.array)
        self.similarity_matrix = np.tril(np.ones((n, n), dtype=np.float32), -1)
        indices = np.tril_indices(
            n, -1
        )  # similarity matrix is a symmetric matrix, so we need only one part lower-triangle or upper-triangle indices

        pairs = list(zip(indices[0].tolist(), indices[1].tolist()))

        tasks = [
            (
                i,
                j,
                self.array[i],
                self.array[j],
                self.measurments,
                self.measurments_coefficients,
            )
            for i, j in pairs
        ]

        with Pool(processes=int(cpu_count() // 2)) as pool:
            for i, j, sim in tqdm.tqdm(
                pool.imap_unordered(_compute_similarity, tasks),
                total=len(tasks),
            ):
                self.similarity_matrix[i][j] = sim

        self.similarity_matrix = (
            self.similarity_matrix + self.similarity_matrix.T
        )  # similarity matrix is symmetric

        self.X = copy.deepcopy(self.similarity_matrix)

        return self

    def cluster(self, t):
        self.t = t

        self.linkage_matrix = linkage(
            squareform(self.X),
            method=self.method,
            metric=self.metric,
            optimal_ordering=self.optimal_ordering,
        )

        self.suggested_optimal_number_clusters()

        self.cluster_labels = fcluster(
            self.linkage_matrix,
            self.t,
            criterion=self.criterion,
            depth=self.depth,
            R=self.R,
            monocrit=self.monocrit,
        )

        self.cluster_results = {}

        try:
            score = silhouette_score(self.X, self.cluster_labels)
            self.cluster_results["silhouette_score"] = score
            print(f"silhouette score: {score}")
        except ValueError as e:
            warn("silhouette score failed")
            warn(str(e))

        try:
            score = davies_bouldin_score(self.X, self.cluster_labels)
            self.cluster_results["davies_bouldin_score"] = score
            print(f"davies bouldin score: {score}")
        except ValueError as e:
            warn("davies bouldin score failed")
            warn(str(e))

        try:
            score = calinski_harabasz_score(self.X, self.cluster_labels)
            self.cluster_results["calinski_harabasz_score"] = score
            print(f"calinski harabasz score: {score}")
        except ValueError as e:
            warn("calinski harabasz score failed")
            warn(str(e))

        return self.linkage_matrix, self.cluster_labels

    def suggested_optimal_number_clusters(self, max_clusters: int = 10):
        silhouette_scores = []
        cluster_ranges = range(2, max_clusters + 1)

        with tqdm.tqdm(
            total=len(cluster_ranges), desc="Finding Optimal Clusters"
        ) as pbar:
            for k in cluster_ranges:
                cluster_labels = fcluster(
                    self.linkage_matrix,
                    k,
                    criterion=self.criterion,
                    depth=self.depth,
                    R=self.R,
                    monocrit=self.monocrit,
                )

                silhouette_scores.append(silhouette_score(self.X, cluster_labels))
                pbar.update()

        self.optimal_clusters = cluster_ranges[np.argmax(silhouette_scores)]
        self.silhouette_scores_differences = silhouette_scores

        print(f"optimal number of clusters: {self.optimal_clusters}")
        print(
            f"silhouette scores from differents clusters number: {self.silhouette_scores_differences}"
        )
