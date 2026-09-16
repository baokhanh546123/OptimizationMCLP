from __future__ import annotations

"""
================================================================================
optimization.py — Multi-Objective Maximal Covering Location Problem (MO-MCLP)
                 ε-constraint solver (OR-Tools CP-SAT)
================================================================================

Bối cảnh Decision Intelligence
------------------------------
Bài toán định vị cơ sở (facility location) cần cân bằng hai mục tiêu mâu thuẫn:
  f1  max covering profit   Σ p_i y_i     (phủ nhu cầu / lợi nhuận)
  f2  min cost              Σ c_j x_j     (chi phí mở cơ sở)

Phương pháp ε-constraint giữ f1 làm objective, biến f2 thành ràng buộc ngân sách:
  max  Σ p_i y_i
  s.t. Σ c_j x_j  ≤  ε
       y_i ≤ Σ_j a_ij x_j
       Σ x_j ≤ P_max   (nếu có)
       x_j, y_i ∈ {0,1}

Quét ε trên [ε_min, ε_max] sinh tập nghiệm Pareto phục vụ ra quyết định.

Early stopping (2 tầng)
-----------------------
1) **Per-solve (giống notebook)**: CP-SAT `relative_gap_limit` + `max_time_in_seconds`
   → dừng một điểm ε khi gap ≤ relative_gap hoặc hết time.
2) **Sweep-level (bổ sung)**: nếu `early_stop=True` và f1 không tăng trong
   `early_stop_patience` điểm ε liên tiếp (sai số `early_stop_tol`) thì dừng
   quét các ε còn lại — tránh giải lại cùng bài P_max khi ngân sách đã dư.

Sau sweep: `re_solve_flagged=True` re-solve các điểm non-monotonic hoặc gap cao.

Ba chế độ (eps_mode)
--------------------
- notebook:    ε_max = Σ c_j, không giảm |J| (đối chiếu ipynb).
- tight:       ε_max = top-P_max costs; auto_reduce theo coverage weight.
- geographic:  Spatial Decomposition — cluster ứng viên (KMeans/PAM),
               giảm cục bộ từng cụm, hợp nghiệm, rồi ε-sweep trên J đã giảm.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

from dataclass.MCLP import MCLP_Data
from utils.load_data import clean_roads
from utils.candidate_grid import *
from core.coverage import build_coverage_matrix_sparse
from core.solver_worker import EpsilonTask, solve_one_epsilon
from model.geo_decomp import (
    candidate_xy_utm,
    kmeans_numpy,
    pam_numpy,
    select_indices_by_cluster,
)


class Optimization:
    __slots__ = (
        "data", "demand_set", "candidate_set", "roads_set", "ward_polygon_wgs84",
        "max_radius_m", "n_points", "time_limit_s", "relative_gap", "n_parallel",
        "workers_per_solve", "use_hint", "non_monotonic_tol", "re_solve_flagged",
        "re_solve_time_limit", "grid_spacing_m", "street_spacing_m", "SCALE",
        "CPU_COUNT", "utm_epsg", "max_candidates", "eps_mode",
        "early_stop", "early_stop_patience", "early_stop_tol", "re_solve_gap_threshold",
        "n_geo_clusters", "geo_cluster_method", "geo_max_per_cluster",
        "_candidate_index_map",
    )

    def __init__(
        self,
        data: Optional[MCLP_Data] = None,
        demand_set=None,
        candidate_set=None,
        roads_set: Optional[str] = None,
        ward_polygon_wgs84=None,
        n_points: int = 15,
        time_limit_s: int = 300,
        relative_gap: float = 0.05,
        n_parallel: int = 4,
        workers_per_solve: int = 2,
        use_hint: bool = True,
        non_monotonic_tol: float = 1e-6,
        re_solve_flagged: bool = True,
        re_solve_time_limit: int = 600,
        re_solve_gap_threshold: float = 20.0,
        early_stop: bool = True,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-6,
        grid_spacing_m: float = 220.0,
        street_spacing_m: float = 80.0,
        max_radius_m: Optional[float] = None,
        SCALE: int = 10 ** 6,
        CPU_COUNT: Optional[int] = None,
        utm_epsg: int = 32648,
        max_candidates: int = 1200,
        eps_mode: str = "notebook",
        n_geo_clusters: int = 6,
        geo_cluster_method: str = "kmeans",
        geo_max_per_cluster: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        relative_gap : float
            Early-stop **per solve** (CP-SAT relative_gap_limit) — giống notebook.
        early_stop : bool
            Early-stop **sweep**: dừng khi f1 bão hòa qua `early_stop_patience` điểm.
        re_solve_flagged : bool
            Sau sweep, re-solve điểm non-monotonic hoặc gap > re_solve_gap_threshold.
        eps_mode : {"notebook", "tight", "geographic"}
        n_geo_clusters : int
            Số cụm không gian (mode geographic).
        geo_cluster_method : {"kmeans", "pam"}
        geo_max_per_cluster : int | None
            Quota ứng viên mỗi cụm; None → chia đều max_candidates.
        """
        self.data = data
        self.demand_set = demand_set
        self.candidate_set = candidate_set
        self.roads_set = roads_set
        self.ward_polygon_wgs84 = ward_polygon_wgs84
        self.max_radius_m = max_radius_m
        self.SCALE = int(SCALE)
        self.n_points = n_points
        self.time_limit_s = time_limit_s
        self.relative_gap = relative_gap
        self.n_parallel = n_parallel
        self.workers_per_solve = workers_per_solve
        self.use_hint = use_hint
        self.non_monotonic_tol = non_monotonic_tol
        self.re_solve_flagged = re_solve_flagged
        self.re_solve_time_limit = re_solve_time_limit
        self.re_solve_gap_threshold = float(re_solve_gap_threshold)
        self.early_stop = bool(early_stop)
        self.early_stop_patience = int(max(1, early_stop_patience))
        self.early_stop_tol = float(early_stop_tol)
        self.grid_spacing_m = grid_spacing_m
        self.street_spacing_m = street_spacing_m
        self.CPU_COUNT = max(1, os.cpu_count() or 1) if CPU_COUNT is None else CPU_COUNT
        self.utm_epsg = utm_epsg
        self.max_candidates = max_candidates
        valid = ("notebook", "tight", "geographic")
        self.eps_mode = eps_mode if eps_mode in valid else "notebook"
        self.n_geo_clusters = max(1, int(n_geo_clusters))
        self.geo_cluster_method = (
            geo_cluster_method if geo_cluster_method in ("kmeans", "pam") else "kmeans"
        )
        self.geo_max_per_cluster = geo_max_per_cluster
        self._candidate_index_map: Optional[np.ndarray] = None

        if self.n_parallel * self.workers_per_solve > self.CPU_COUNT:
            print(
                f"      [CẢNH BÁO] n_parallel({self.n_parallel}) x "
                f"workers_per_solve({self.workers_per_solve}) = "
                f"{self.n_parallel * self.workers_per_solve} > CPU_COUNT({self.CPU_COUNT})"
            )

    def reduce_candidates_by_coverage(
        self,
        data: Optional[MCLP_Data] = None,
        max_candidates: Optional[int] = None,
    ) -> MCLP_Data:
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("data is None")
        K = int(max_candidates if max_candidates is not None else self.max_candidates)
        n_j = data.n_j
        if n_j <= K:
            return data

        weights = np.zeros(n_j, dtype=np.float64)
        for i, js in enumerate(data.coverage_lists):
            pi = float(data.p[i])
            for j in js:
                weights[int(j)] += pi

        order = np.argsort(-weights)
        keep_sorted = np.sort(order[:K].astype(np.int64))

        a_new = data.a[:, keep_sorted].tocsr()
        c_new = data.c[keep_sorted]
        new_data = MCLP_Data(
            p=data.p.copy(),
            a=a_new,
            c=c_new,
            P_max=data.P_max,
            budget=data.budget,
            must_cover=data.must_cover.copy() if data.must_cover is not None else None,
        )
        self._candidate_index_map = keep_sorted.copy()
        print(
            f"[OK] Candidate reduction: |J| {n_j} → {new_data.n_j} "
            f"(top by coverage weight, max_candidates={K})"
        )
        self.data = new_data
        return new_data

    def reduce_candidates_geographic(
        self,
        data: Optional[MCLP_Data] = None,
        n_clusters: Optional[int] = None,
        method: Optional[str] = None,
        max_per_cluster: Optional[int] = None,
    ) -> MCLP_Data:
        """Spatial Decomposition: cluster candidates → local top-k → merge.

        Giữ nguyên ý nghĩa covering; chỉ thay J bằng J' ⊆ J chọn theo cụm + weight.
        """
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("data is None")

        k = int(n_clusters if n_clusters is not None else self.n_geo_clusters)
        method = (method or self.geo_cluster_method).lower()
        if method not in ("kmeans", "pam"):
            method = "kmeans"

        n_j = data.n_j
        if n_j <= 1:
            return data

        k = min(k, n_j)
        xy = candidate_xy_utm(self.candidate_set, self.utm_epsg)
        if xy.shape[0] != n_j:
            raise ValueError(
                f"candidate_set size ({xy.shape[0]}) != data.n_j ({n_j})"
            )

        if method == "pam" and n_j > 3000:
            print(f"[GEO] |J|={n_j} lớn → fallback KMeans (PAM chậm).")
            method = "kmeans"

        labels = pam_numpy(xy, k) if method == "pam" else kmeans_numpy(xy, k)

        weights = np.zeros(n_j, dtype=np.float64)
        for i, js in enumerate(data.coverage_lists):
            pi = float(data.p[i])
            for j in js:
                weights[int(j)] += pi

        if max_per_cluster is not None:
            quota = int(max_per_cluster)
        elif self.geo_max_per_cluster is not None:
            quota = int(self.geo_max_per_cluster)
        else:
            quota = max(1, int(np.ceil(self.max_candidates / k)))

        keep_arr = select_indices_by_cluster(labels, weights, quota)
        a_new = data.a[:, keep_arr].tocsr()
        c_new = data.c[keep_arr]
        new_data = MCLP_Data(
            p=data.p.copy(),
            a=a_new,
            c=c_new,
            P_max=data.P_max,
            budget=data.budget,
            must_cover=data.must_cover.copy() if data.must_cover is not None else None,
        )
        self._candidate_index_map = keep_arr.copy()
        print(
            f"[OK] Geographic reduction ({method}): |J| {n_j} → {new_data.n_j} "
            f"(n_clusters={k}, quota/cluster≈{quota})"
        )
        for c in range(k):
            sz = int(np.sum(labels == c))
            tk = int(np.sum(np.isin(keep_arr, np.where(labels == c)[0])))
            if sz:
                print(f"      cluster {c}: size={sz} → keep={tk}")
        self.data = new_data
        return new_data
