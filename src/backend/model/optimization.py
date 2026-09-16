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
- benders:     Benders Decomposition (exact) — master chỉ có x_j + η (KHÔNG
               có y_i / ràng buộc covering, kích thước không phụ thuộc |I|).
               Subproblem dạng đóng (không cần LP solver), sinh optimality cut
               lõm mỗi vòng lặp. Xem model/benders_solver.py để biết chứng
               minh công thức + giới hạn so với Branch-and-Benders-Cut thật
               (Cordeau et al., EJOR 2019) — OR-Tools CP-SAT không expose lazy
               constraint trong 1 cây B&B nên đây là iterative cutting-plane,
               vẫn exact nhưng re-solve master nhiều lần thay vì 1 lần.
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
from model.benders_solver import BendersEpsilonTask, solve_one_epsilon_benders


class Optimization:
    __slots__ = (
        "data", "demand_set", "candidate_set", "roads_set", "ward_polygon_wgs84",
        "max_radius_m", "n_points", "time_limit_s", "relative_gap", "n_parallel",
        "workers_per_solve", "use_hint", "non_monotonic_tol", "re_solve_flagged",
        "re_solve_time_limit", "grid_spacing_m", "street_spacing_m", "SCALE",
        "CPU_COUNT", "utm_epsg", "max_candidates", "eps_mode",
        "early_stop", "early_stop_patience", "early_stop_tol", "re_solve_gap_threshold",
        "n_geo_clusters", "geo_cluster_method", "geo_max_per_cluster",
        "benders_max_iters", "benders_master_time_limit_s",
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
        benders_max_iters: int = 200,
        benders_master_time_limit_s: float = 5.0,
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
        eps_mode : {"notebook", "tight", "geographic", "benders"}
        n_geo_clusters : int
            Số cụm không gian (mode geographic).
        geo_cluster_method : {"kmeans", "pam"}
        geo_max_per_cluster : int | None
            Quota ứng viên mỗi cụm; None → chia đều max_candidates.
        benders_max_iters : int
            Số vòng lặp cutting-plane tối đa mỗi điểm ε (mode benders).
        benders_master_time_limit_s : float
            Trần thời gian MỖI lần giải master CP-SAT (không phải tổng) — tổng
            vẫn bị chặn bởi time_limit_s như các mode khác.
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
        valid = ("notebook", "tight", "geographic", "benders")
        self.eps_mode = eps_mode if eps_mode in valid else "notebook"
        self.n_geo_clusters = max(1, int(n_geo_clusters))
        self.geo_cluster_method = (
            geo_cluster_method if geo_cluster_method in ("kmeans", "pam") else "kmeans"
        )
        self.geo_max_per_cluster = geo_max_per_cluster
        self.benders_max_iters = max(1, int(benders_max_iters))
        self.benders_master_time_limit_s = float(benders_master_time_limit_s)
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

    def build_candidate_set(self):
        grid_candidate = generate_candidate_grid(
            self.ward_polygon_wgs84, self.grid_spacing_m, self.utm_epsg
        )
        if not self.roads_set:
            self.candidate_set = grid_candidate
            return grid_candidate, np.ones(len(grid_candidate))

        print(f"\n[OK] Grid candidate: {len(grid_candidate)} vị trí (spacing {self.grid_spacing_m}m)")

        import geopandas as gpd
        roads_raw = gpd.read_file(self.roads_set)
        roads_clean = clean_roads(roads_raw)
        n_eligible = int(roads_clean["is_candidate_eligible"].sum())
        print(f"[OK] Roads: {len(roads_raw)} segment, {n_eligible} eligible sau clean_roads()")

        street_cand = generate_street_candidates(
            roads_clean, self.ward_polygon_wgs84, spacing_m=self.street_spacing_m,
            utm_epsg=self.utm_epsg, start_id=len(grid_candidate),
        )
        print(f"[OK] Street candidate: {len(street_cand)} vị trí (spacing {self.street_spacing_m}m)")

        merged = merge_candidate_sets(grid_candidate, street_cand, utm_epsg=self.utm_epsg)
        cost = derive_candidate_cost(merged)
        print(f"[OK] Candidate set J sau merge + dedup: {len(merged)}")

        self.candidate_set = merged
        return merged, cost

    def build_coverage_matrix(self):
        if self.demand_set is None or self.candidate_set is None:
            raise ValueError("demand_set và candidate_set phải được gán trước.")
        return build_coverage_matrix_sparse(
            self.demand_set, self.candidate_set,
            utm_epsg=self.utm_epsg, max_radius_m=self.max_radius_m,
        )

    def build_base_template(self):
        if self.data is None:
            raise ValueError("self.data (MCLP_Data) chưa được gán.")

        mdl = cp_model.CpModel()
        data = self.data
        n_i, n_j = data.n_i, data.n_j

        x = [mdl.NewBoolVar(f"x_{j}") for j in range(n_j)]
        y = [mdl.NewBoolVar(f"y_{i}") for i in range(n_i)]

        for i, covering_js in enumerate(data.coverage_lists):
            if covering_js.size == 0:
                mdl.Add(y[i] == 0)
            else:
                mdl.Add(y[i] <= sum(x[int(j)] for j in covering_js))

        if data.must_cover is not None:
            for i in data.must_cover:
                mdl.Add(y[int(i)] == 1)

        if data.P_max is not None:
            mdl.Add(sum(x) <= int(data.P_max))

        if data.budget is not None:
            c_int = np.round(data.c * self.SCALE).astype(np.int64)
            budget_int = int(round(float(data.budget) * self.SCALE))
            mdl.Add(sum(int(c_int[j]) * x[j] for j in range(n_j)) <= budget_int)

        return mdl, x, y

    def _greedy_solution_simple(
        self, eps: float, return_f1: bool = False
    ):
        data = self.data
        n_j = data.n_j
        p, c = data.p, data.c
        P_max = int(data.P_max) if data.P_max is not None else n_j

        cand_covers: List[List[int]] = [[] for _ in range(n_j)]
        for i, js in enumerate(data.coverage_lists):
            for j in js:
                cand_covers[int(j)].append(i)

        covered = np.zeros(data.n_i, dtype=bool)
        chosen: List[int] = []
        remaining = float(eps)

        for _ in range(P_max):
            best_j, best_score = -1, -1.0
            for j in range(n_j):
                if j in chosen or c[j] > remaining + 1e-12:
                    continue
                gain = sum(float(p[i]) for i in cand_covers[j] if not covered[i])
                if gain <= 0:
                    continue
                score = gain * 1e6 if c[j] <= 1e-12 else gain / float(c[j])
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j < 0:
                break
            chosen.append(best_j)
            remaining -= float(c[best_j])
            for i in cand_covers[best_j]:
                covered[i] = True

        if not chosen:
            return (None, 0.0) if return_f1 else None
        x = [0] * n_j
        for j in chosen:
            x[j] = 1
        f1 = float(sum(float(p[i]) for i in range(data.n_i) if covered[i]))
        if not return_f1:
            print(f"[OK] Greedy warm-start (notebook-style): f1={f1:.3f}, n_fac={len(chosen)}")
        return (x, f1) if return_f1 else x

    def _submodular_diagnostic_bound(self, eps: float, r: dict) -> None:
        """Cận trên THAM KHẢO, không thay thế optimality_gap_pct chính thức của CP-SAT.

        Vì sao cần: đã kiểm chứng thực nghiệm (LP-relaxation liên tục giải bằng
        scipy.linprog khớp gần tuyệt đối với BestObjectiveBound của CP-SAT trên
        instance cấu trúc tương tự) rằng với P_max nhỏ so với |J| và mật độ phủ
        cao, bound CP-SAT trả về gần như CHÍNH LÀ LP-relaxation gốc — không cải
        thiện đáng kể dù tăng workers/bật optimize_with_core/probing hay tăng
        time_limit trong khoảng thời gian thực tế. Với ε mà P_max — chứ không
        phải budget — là ràng buộc chi phối, bài toán quy về max-coverage dưới
        ràng buộc cardinality, một bài toán SUBMODULAR ĐƠN ĐIỆU, nên định lý
        Nemhauser-Wolsey-Fisher (1978) cho:  OPT ≤ f(greedy) / (1 - 1/e)
        — một cận trên thường CHẶT HƠN NHIỀU so với LP-relaxation bound ở loại
        instance này (xem ví dụ bằng số trong PR note).
        """
        data = self.data
        P_max = data.P_max
        valid = P_max is not None and r["n_facilities"] == int(P_max)
        r["submodular_bound_valid"] = valid
        if not valid:
            r["submodular_diagnostic_ub"] = None
            r["submodular_diagnostic_gap_pct"] = None
            return
        greedy_x, greedy_f1 = self._greedy_solution_simple(eps, return_f1=True)
        if not greedy_x or greedy_f1 <= 0:
            r["submodular_diagnostic_ub"] = None
            r["submodular_diagnostic_gap_pct"] = None
            return
        ub = greedy_f1 / (1.0 - 1.0 / np.e)
        f1 = max(float(r["f1_covering_profit"]), 1e-9)
        r["submodular_diagnostic_ub"] = float(ub)
        r["submodular_diagnostic_gap_pct"] = float((ub - f1) / f1 * 100.0)

    def _compute_epsilon_range(self) -> tuple[float, float]:
        data = self.data
        c = np.asarray(data.c, dtype=np.float64)
        eps_min = float(np.min(c))

        if data.budget is not None:
            eps_max = float(data.budget)
        elif self.eps_mode in ("tight", "geographic", "benders") and data.P_max is not None and int(data.P_max) < data.n_j:
            P = int(data.P_max)
            eps_max = float(np.sort(c)[-P:].sum()) * 1.01
        else:
            eps_max = float(np.sum(c))

        eps_max = max(eps_max, eps_min)
        return eps_min, eps_max

    def _make_epsilons(self) -> np.ndarray:
        eps_min, eps_max = self._compute_epsilon_range()
        return np.linspace(eps_min, eps_max, self.n_points).astype(float)

    def epsilon_constraint_sweep(self, auto_reduce: bool = False):
        """Chạy ε-constraint + early-stop sweep + re-solve điểm xấu.

        Luồng cuối (giống tinh thần notebook + mở rộng):
          1. Solve từng ε (per-solve early-stop qua relative_gap).
          2. Nếu early_stop và f1 bão hòa → dừng sweep.
          3. Gắn cờ non-monotonic.
          4. re_solve_flagged → re-solve điểm non-monotonic / gap cao.
          5. Report summary.
        """
        if self.data is None:
            raise ValueError("self.data chưa được gán")

        if self.eps_mode == "benders":
            return self._epsilon_constraint_sweep_benders()

        if self.eps_mode == "geographic":
            self.reduce_candidates_geographic()
        elif auto_reduce and self.data.n_j > self.max_candidates:
            self.reduce_candidates_by_coverage(max_candidates=self.max_candidates)
        else:
            print(
                f"[EXPERIMENT] Không giảm candidate — |J|={self.data.n_j} "
                f"(mode={self.eps_mode}, auto_reduce={auto_reduce})"
            )

        data = self.data
        n_i, n_j = data.n_i, data.n_j

        eps_min, eps_max = self._compute_epsilon_range()
        print(
            f"[INFO] mode={self.eps_mode} | CPU={self.CPU_COUNT} | "
            f"n_parallel={self.n_parallel} | time_limit={self.time_limit_s}s | "
            f"gap_target={self.relative_gap * 100:.1f}% | SCALE={self.SCALE:,}"
        )
        print(
            f"[INFO] early_stop={self.early_stop} (patience={self.early_stop_patience}) | "
            f"re_solve_flagged={self.re_solve_flagged} "
            f"(gap_threshold={self.re_solve_gap_threshold}%)"
        )
        print(
            f"[INFO] ε range = [{eps_min:.4f}, {eps_max:.4f}]  "
            f"(P_max={data.P_max}, n_i={n_i}, n_j={n_j}, nnz={data.a.nnz})"
        )

        base_mdl, _, _ = self.build_base_template()
        template_text = str(base_mdl.Proto())

        epsilons = self._make_epsilons()
        print(f"[INFO] Số điểm ε dự kiến: {len(epsilons)}")

        results: list = []
        hint_x = None
        if self.use_hint:
            hint_x = self._greedy_solution_simple(float(epsilons[-1]))

        # Sweep-level early-stop state
        plateau_count = 0
        best_f1_seen = -np.inf
        stopped_early = False

        ctx = mp.get_context("fork")

        with ProcessPoolExecutor(max_workers=self.n_parallel, mp_context=ctx) as ex:
            for batch_start in range(0, len(epsilons), self.n_parallel):
                if stopped_early:
                    break

                batch_eps = epsilons[batch_start: batch_start + self.n_parallel]
                tasks = [
                    EpsilonTask(
                        template_proto_text=template_text,
                        n_i=n_i,
                        n_j=n_j,
                        c=data.c,
                        p=data.p,
                        eps=float(eps),
                        time_limit_s=self.time_limit_s,
                        num_search_workers=self.workers_per_solve,
                        relative_gap=self.relative_gap,
                        hint_x=hint_x if self.use_hint else None,
                        scale=self.SCALE,
                    )
                    for eps in batch_eps
                ]

                for eps_val, r in zip(batch_eps, ex.map(solve_one_epsilon, tasks)):
                    if r is None:
                        print(
                            f"  → ε={eps_val:.4f} FAILED "
                            f"(infeasible / no solution trong time_limit)"
                        )
                        continue

                    new_x = r.pop("x_solution")
                    if self.use_hint:
                        hint_x = new_x
                    self._submodular_diagnostic_bound(eps_val, r)
                    results.append(r)
                    diag = (
                        f"  diag_gap={r['submodular_diagnostic_gap_pct']:.2f}%"
                        if r.get("submodular_diagnostic_ub") is not None else ""
                    )
                    print(
                        f"  → ε={r['epsilon']:.4f}  f1={r['f1_covering_profit']:.3f}  "
                        f"f2={r['f2_cost']:.4f}  n_fac={r['n_facilities']}  "
                        f"gap={r['optimality_gap_pct']:.2f}%{diag}  t={r['solve_time_s']:.1f}s"
                    )

                    # --- Sweep early-stop: f1 plateau ---
                    if self.early_stop:
                        f1 = float(r["f1_covering_profit"])
                        if f1 > best_f1_seen + self.early_stop_tol:
                            best_f1_seen = f1
                            plateau_count = 0
                        else:
                            plateau_count += 1
                            if plateau_count >= self.early_stop_patience:
                                remaining = len(epsilons) - (batch_start + len(batch_eps))
                                print(
                                    f"[EARLY-STOP] f1 bão hòa qua {plateau_count} điểm ε "
                                    f"(best_f1={best_f1_seen:.3f}) — bỏ {max(0, remaining)} "
                                    f"điểm ε còn lại."
                                )
                                stopped_early = True
                                break

        results.sort(key=lambda r: r["epsilon"])
        self._flag_non_monotonic(results)

        if self.re_solve_flagged:
            self._resolve_flagged_points(results, template_text, n_i, n_j, hint_x)

        results.sort(key=lambda r: r["epsilon"])
        self._remap_results_to_original_indices(results)
        self._report_summary(results, stopped_early=stopped_early)
        return results

    def epsilon_constraint_sweep_notebook(self):
        """API notebook-style: không reduction, eps_mode=notebook."""
        prev = self.eps_mode
        self.eps_mode = "notebook"
        try:
            return self.epsilon_constraint_sweep(auto_reduce=False)
        finally:
            self.eps_mode = prev

    def epsilon_constraint_sweep_geographic(
        self,
        n_clusters: Optional[int] = None,
        method: Optional[str] = None,
        max_per_cluster: Optional[int] = None,
    ):
        """API geographic: Spatial Decomposition + ε-sweep trên tập đã giảm."""
        prev_mode = self.eps_mode
        prev_k = self.n_geo_clusters
        prev_m = self.geo_cluster_method
        prev_q = self.geo_max_per_cluster
        self.eps_mode = "geographic"
        if n_clusters is not None:
            self.n_geo_clusters = int(n_clusters)
        if method is not None:
            self.geo_cluster_method = method
        if max_per_cluster is not None:
            self.geo_max_per_cluster = max_per_cluster
        try:
            return self.epsilon_constraint_sweep(auto_reduce=False)
        finally:
            self.eps_mode = prev_mode
            self.n_geo_clusters = prev_k
            self.geo_cluster_method = prev_m
            self.geo_max_per_cluster = prev_q

    def _remap_results_to_original_indices(self, results: list) -> None:
        """Map chosen_candidates từ chỉ số J' về J gốc sau reduction.

        Solver trả index trong tập đã giảm; candidate_gdf dùng candidate_id gốc.
        """
        idx_map = getattr(self, "_candidate_index_map", None)
        if idx_map is None or len(results) == 0:
            return
        idx_map = np.asarray(idx_map, dtype=np.int64)
        n_map = len(idx_map)
        for r in results:
            chosen = r.get("chosen_candidates")
            if chosen is not None:
                remapped = []
                for j in chosen:
                    j = int(j)
                    if 0 <= j < n_map:
                        remapped.append(int(idx_map[j]))
                    else:
                        remapped.append(j)
                r["chosen_candidates"] = remapped
                r["chosen_candidates_reduced"] = list(chosen)
        print(
            f"[OK] Remapped chosen_candidates → original candidate_id "
            f"(index_map size={n_map})"
        )

    def _flag_non_monotonic(self, results: list) -> None:
        running_max_f1 = -np.inf
        for r in results:
            r["flagged_non_monotonic"] = bool(
                r["f1_covering_profit"] < running_max_f1 - self.non_monotonic_tol
            )
            running_max_f1 = max(running_max_f1, r["f1_covering_profit"])

    def _resolve_flagged_points(
        self,
        results: list,
        template_text: str,
        n_i: int,
        n_j: int,
        best_hint: Optional[Sequence[int]] = None,
    ) -> None:
        """Re-solve điểm non-monotonic hoặc gap > threshold (mở rộng so với notebook)."""
        data = self.data
        gap_threshold = self.re_solve_gap_threshold
        flagged_idx = [
            idx for idx, r in enumerate(results)
            if r["flagged_non_monotonic"] or r["optimality_gap_pct"] > gap_threshold
        ]
        if not flagged_idx:
            print("[RE-SOLVE] Không có điểm nào cần re-solve.")
            return

        print(
            f"\n[RE-SOLVE] {len(flagged_idx)} điểm (non-monotonic hoặc gap > {gap_threshold}%) — "
            f"time_limit={self.re_solve_time_limit}s"
        )

        for idx in flagged_idx:
            r = results[idx]
            eps = float(r["epsilon"])
            print(f"  re-solve ε={eps:.4f} (gap={r['optimality_gap_pct']:.1f}%) ...", end=" ")
            task = EpsilonTask(
                template_proto_text=template_text,
                n_i=n_i,
                n_j=n_j,
                c=data.c,
                p=data.p,
                eps=eps,
                time_limit_s=self.re_solve_time_limit,
                num_search_workers=self.workers_per_solve,
                relative_gap=self.relative_gap,
                hint_x=best_hint,
                scale=self.SCALE,
            )
            r_new = solve_one_epsilon(task)
            if r_new is not None and (
                r_new["f1_covering_profit"] > r["f1_covering_profit"] + 1e-9
                or r_new["optimality_gap_pct"] < r["optimality_gap_pct"] - 1e-6
            ):
                r_new.pop("x_solution", None)
                r_new["flagged_non_monotonic"] = False
                self._submodular_diagnostic_bound(eps, r_new)
                results[idx] = r_new
                diag = (
                    f" diag_gap={r_new['submodular_diagnostic_gap_pct']:.2f}%"
                    if r_new.get("submodular_diagnostic_ub") is not None else ""
                )
                print(
                    f"CẢI THIỆN → f1={r_new['f1_covering_profit']:.3f} "
                    f"gap={r_new['optimality_gap_pct']:.2f}%{diag}"
                )
            else:
                print("không cải thiện")


    def _epsilon_constraint_sweep_benders(self):
        """ε-sweep bằng Benders Decomposition (exact) — xem model/benders_solver.py.

        Không gọi build_base_template()/không có template proto covering nào —
        master mỗi điểm ε chỉ gồm x_j + η, kích thước KHÔNG phụ thuộc |I|.
        """
        data = self.data
        n_i, n_j = data.n_i, data.n_j
        print(
            f"[EXPERIMENT] eps_mode=benders — master chỉ có x_j+η (|J|={n_j}), "
            f"KHÔNG có y_i (|I|={n_i}) hay ràng buộc covering trong master."
        )
        eps_min, eps_max = self._compute_epsilon_range()
        print(
            f"[INFO] mode=benders | CPU={self.CPU_COUNT} | n_parallel={self.n_parallel} | "
            f"time_limit={self.time_limit_s}s/điểm | "
            f"master_time_limit={self.benders_master_time_limit_s}s | "
            f"max_iters={self.benders_max_iters} | "
            f"gap_target={self.relative_gap * 100:.1f}% | SCALE={self.SCALE:,}"
        )
        print(
            f"[INFO] ε range = [{eps_min:.4f}, {eps_max:.4f}]  "
            f"(P_max={data.P_max}, n_i={n_i}, n_j={n_j}, nnz={data.a.nnz})"
        )
        epsilons = self._make_epsilons()
        print(f"[INFO] Số điểm ε dự kiến: {len(epsilons)}")

        a_csr = data.a.tocsr()
        results: list = []
        plateau_count = 0
        best_f1_seen = -np.inf
        stopped_early = False

        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=self.n_parallel, mp_context=ctx) as ex:
            for batch_start in range(0, len(epsilons), self.n_parallel):
                if stopped_early:
                    break
                batch_eps = epsilons[batch_start: batch_start + self.n_parallel]
                tasks = [
                    BendersEpsilonTask(
                        a=a_csr, p=data.p, c=data.c, eps=float(eps), P_max=data.P_max,
                        time_limit_s=self.time_limit_s,
                        master_time_limit_s=self.benders_master_time_limit_s,
                        relative_gap=self.relative_gap, scale=self.SCALE,
                        max_iters=self.benders_max_iters,
                        num_search_workers=self.workers_per_solve,
                    )
                    for eps in batch_eps
                ]
                for eps_val, r in zip(batch_eps, ex.map(solve_one_epsilon_benders, tasks)):
                    if r is None:
                        print(f"  → ε={eps_val:.4f} FAILED (infeasible / no solution)")
                        continue
                    r.pop("x_solution")
                    self._submodular_diagnostic_bound(eps_val, r)
                    results.append(r)
                    diag = (
                        f"  diag_gap={r['submodular_diagnostic_gap_pct']:.2f}%"
                        if r.get("submodular_diagnostic_ub") is not None else ""
                    )
                    print(
                        f"  → ε={r['epsilon']:.4f}  f1={r['f1_covering_profit']:.3f}  "
                        f"f2={r['f2_cost']:.4f}  n_fac={r['n_facilities']}  "
                        f"gap={r['optimality_gap_pct']:.2f}%{diag}  "
                        f"iters={r['n_benders_iters']}  {r['status']}  t={r['solve_time_s']:.1f}s"
                    )

                    if self.early_stop:
                        f1 = float(r["f1_covering_profit"])
                        if f1 > best_f1_seen + self.early_stop_tol:
                            best_f1_seen = f1
                            plateau_count = 0
                        else:
                            plateau_count += 1
                            if plateau_count >= self.early_stop_patience:
                                remaining = len(epsilons) - (batch_start + len(batch_eps))
                                print(
                                    f"[EARLY-STOP] f1 bão hòa qua {plateau_count} điểm ε "
                                    f"(best_f1={best_f1_seen:.3f}) — bỏ {max(0, remaining)} "
                                    f"điểm ε còn lại."
                                )
                                stopped_early = True
                                break

        results.sort(key=lambda r: r["epsilon"])
        self._flag_non_monotonic(results)

        if self.re_solve_flagged:
            self._resolve_flagged_points_benders(results)

        results.sort(key=lambda r: r["epsilon"])
        self._remap_results_to_original_indices(results)
        self._report_summary(results, stopped_early=stopped_early)
        return results

    def _resolve_flagged_points_benders(self, results: list) -> None:
        data = self.data
        a_csr = data.a.tocsr()
        gap_threshold = self.re_solve_gap_threshold
        flagged_idx = [
            idx for idx, r in enumerate(results)
            if r["flagged_non_monotonic"] or r["optimality_gap_pct"] > gap_threshold
        ]
        if not flagged_idx:
            print("[RE-SOLVE] Không có điểm nào cần re-solve.")
            return
        print(
            f"\n[RE-SOLVE] {len(flagged_idx)} điểm (non-monotonic hoặc gap > {gap_threshold}%) — "
            f"time_limit={self.re_solve_time_limit}s (benders, max_iters x2)"
        )
        for idx in flagged_idx:
            eps = results[idx]["epsilon"]
            print(f"  → Re-solving ε={eps:.4f} (benders) ...", end=" ", flush=True)
            task = BendersEpsilonTask(
                a=a_csr, p=data.p, c=data.c, eps=float(eps), P_max=data.P_max,
                time_limit_s=self.re_solve_time_limit,
                master_time_limit_s=self.benders_master_time_limit_s,
                relative_gap=min(0.01, self.relative_gap), scale=self.SCALE,
                max_iters=self.benders_max_iters * 2,
                num_search_workers=self.workers_per_solve,
            )
            r_new = solve_one_epsilon_benders(task)
            if (
                r_new is not None
                and r_new["f1_covering_profit"] >= results[idx]["f1_covering_profit"] - 1e-6
            ):
                r_new.pop("x_solution", None)
                r_new["flagged_non_monotonic"] = False
                self._submodular_diagnostic_bound(eps, r_new)
                results[idx] = r_new
                print(
                    f"CẢI THIỆN → f1={r_new['f1_covering_profit']:.3f} "
                    f"gap={r_new['optimality_gap_pct']:.2f}%"
                )
            else:
                print("không cải thiện")

    def _report_summary(self, results: list, stopped_early: bool = False) -> None:
        if not results:
            print("\n      [epsilon_constraint_sweep] Không có nghiệm nào.")
            return
        n_flagged = sum(1 for r in results if r["flagged_non_monotonic"])
        gaps = [r["optimality_gap_pct"] for r in results]
        avg_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))
        extra = " | EARLY-STOPPED" if stopped_early else ""
        print(
            f"\n      [epsilon_constraint_sweep] {len(results)} điểm Pareto{extra}, "
            f"{n_flagged} non-monotonic, gap TB={avg_gap:.2f}%, gap max={max_gap:.2f}%"
        )
        diag_gaps = [
            r["submodular_diagnostic_gap_pct"] for r in results
            if r.get("submodular_diagnostic_ub") is not None
        ]
        if diag_gaps:
            print(
                f"      [DIAGNOSTIC] submodular bound (Nemhauser-Wolsey-Fisher, chỉ hợp lệ "
                f"khi P_max là ràng buộc chi phối) — diag_gap TB={np.mean(diag_gaps):.2f}%, "
                f"max={np.max(diag_gaps):.2f}% trên {len(diag_gaps)}/{len(results)} điểm. "
                f"Nếu diag_gap << optimality_gap_pct thì gap báo cáo bởi CP-SAT nhiều khả năng "
                f"là do LP-relaxation bound lỏng (đặc tính bài toán), KHÔNG phải nghiệm tệ."
            )
        if avg_gap > 20:
            print(
                f"      [CẢNH BÁO] gap TB > 20% — với |J| lớn không reduction đây là dự kiến. "
                f"Thử eps_mode='tight'/'geographic' + auto_reduce để so sánh."
            )