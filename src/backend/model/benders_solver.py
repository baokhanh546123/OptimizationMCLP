"""Benders Decomposition cho MO-MCLP (biến thể ε-constraint tại 1 điểm ε).

================================================================================
Ý tưởng toán học
================================================================================
Bài toán gốc tại một ε cố định:
    max  Σ_i p_i y_i
    s.t. Σ_j c_j x_j ≤ ε
         Σ x_j ≤ P_max                (nếu có)
         y_i ≤ Σ_{j∈N(i)} x_j ,  0 ≤ y_i ≤ 1
         x_j ∈ {0,1}

Với x CỐ ĐỊNH, subproblem trên y TÁCH RỜI theo từng demand i (mỗi y_i chỉ có
đúng 1 ràng buộc), nên có nghiệm và dual DẠNG ĐÓNG — không cần gọi LP solver:

    y_i*(x)      = 1{ Σ_{j∈N(i)} x_j ≥ 1 }              (i được phủ hay không)
    λ_i*(x)      = p_i · 1{ Σ_{j∈N(i)} x_j = 0 }         (dual/subgradient)

(đã verify bằng scipy.linprog trên instance ngẫu nhiên: dual khớp tuyệt đối,
xem benders_cut_check.py trong PR — sai số 0.0).

f(x) = Σ_i p_i·min(1, Σ_j a_ij x_j) là hàm LÕM theo x (tổng của min(affine,
const) là lõm), nên với BẤT KỲ x̄ nào, (f(x̄), λ(x̄)) cho một OPTIMALITY CUT
hợp lệ TOÀN CỤC (đúng với MỌI x khả thi, không chỉ quanh x̄):

    η  ≤  f(x̄) + Σ_j m_j(x̄) · (x_j − x̄_j),      m_j(x̄) = Σ_i a_ij λ_i(x̄)

Với x̄_j=1 thì j đã bật, mọi i mà j phủ đã có cov≥1 nên λ_i=0 cho các i đó
→ m_j(x̄)=0 tự động khi x̄_j=1. Cut rút gọn còn (chỉ tính trên j đang TẮT):

    η  ≤  f(x̄) + Σ_{j : x̄_j=0} m_j(x̄) · x_j

MASTER problem chỉ có biến x_j (n_j biến) + η — HOÀN TOÀN KHÔNG có biến y_i
(n_i biến) hay ràng buộc covering nào. Đây là điểm mạnh cốt lõi khi |I| rất
lớn: kích thước master không phụ thuộc |I|, chỉ phụ thuộc |J|.

================================================================================
Giới hạn cần biết (khác với Cordeau et al., EJOR 2019)
================================================================================
Cordeau et al. dùng Branch-and-Benders-Cut: cắt được thêm NGAY trong lúc
solver đang chạy 1 cây B&B duy nhất (qua lazy constraint callback của
Gurobi/CPLEX). OR-Tools CP-SAT (Python API, bản hiện tại) KHÔNG expose cơ chế
lazy-constraint-trong-1-cây tương đương. Module này cài **classical iterative
Benders / Kelley's cutting-plane**: mỗi vòng lặp giải LẠI master (đã có thêm
cut mới) từ đầu, dùng AddHint để warm-start. Vẫn EXACT (hội tụ đúng optimum,
chứng minh bằng η_bar == f(x̄) tại hội tụ — η_bar luôn là cận trên toàn cục
hợp lệ nhờ cut lõm ở trên), nhưng chậm hơn bản lazy-cut một cây trên bài cực
lớn vì phải re-solve master nhiều lần thay vì 1 lần duy nhất.
"""
from __future__ import annotations

import time
from typing import List, NamedTuple, Optional

import numpy as np
from ortools.sat.python import cp_model
from scipy import sparse


def covering_gain_and_marginals(
    a_csr: sparse.csr_matrix, p: np.ndarray, x_bar: np.ndarray
):
    """Subproblem dạng đóng: trả (f(x̄), marginal gain m_j mỗi candidate, covered mask)."""
    cov = a_csr.dot(x_bar.astype(np.float64))
    covered = cov >= 0.5
    f_val = float(p[covered].sum())
    lam = np.where(covered, 0.0, p)
    m = a_csr.T.dot(lam)
    return f_val, np.asarray(m).ravel(), covered


class BendersEpsilonTask(NamedTuple):
    a: sparse.csr_matrix          # (n_i, n_j) coverage — KHÔNG đi qua CP-SAT template
    p: np.ndarray
    c: np.ndarray
    eps: float
    P_max: Optional[int]
    time_limit_s: float           # tổng ngân sách thời gian cho cả vòng lặp Benders
    master_time_limit_s: float    # trần thời gian MỖI lần giải master
    relative_gap: float
    scale: int
    max_iters: int
    num_search_workers: int


def solve_one_epsilon_benders(task: BendersEpsilonTask) -> Optional[dict]:
    """Giải 1 điểm ε bằng iterative Benders. Cùng shape kết quả với
    core.solver_worker.solve_one_epsilon để cắm thẳng vào epsilon_constraint_sweep.
    """
    n_j = task.a.shape[1]
    scale = int(task.scale)
    c_arr = np.asarray(task.c, dtype=np.float64)
    p_arr = np.asarray(task.p, dtype=np.float64)
    a_csr = task.a.tocsr() if not sparse.isspmatrix_csr(task.a) else task.a

    c_int = np.round(c_arr * scale).astype(np.int64)
    eps_int = int(round(float(task.eps) * scale))
    eta_ub_int = int(round(float(p_arr.sum()) * scale)) + 1

    mdl = cp_model.CpModel()
    x = [mdl.NewBoolVar(f"x_{j}") for j in range(n_j)]
    eta = mdl.NewIntVar(0, eta_ub_int, "eta")

    mdl.Add(sum(int(c_int[j]) * x[j] for j in range(n_j)) <= eps_int)
    if task.P_max is not None:
        mdl.Add(sum(x) <= int(task.P_max))
    mdl.Maximize(eta)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = max(1, task.num_search_workers)
    solver.parameters.log_search_progress = False

    gap_tol_int = max(1, int(round(task.relative_gap * scale)))
    best_f, best_x = -1.0, None
    last_bound = float(eta_ub_int)
    t0 = time.time()
    n_iters = 0
    converged = False

    for it in range(max(1, task.max_iters)):
        n_iters = it + 1
        remaining = task.time_limit_s - (time.time() - t0)
        if remaining <= 0:
            break
        solver.parameters.max_time_in_seconds = max(0.05, min(task.master_time_limit_s, remaining))

        if best_x is not None:
            mdl.ClearHints()
            for j in range(n_j):
                mdl.AddHint(x[j], int(best_x[j]))

        status = solver.Solve(mdl)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        x_bar = np.array([solver.Value(x[j]) for j in range(n_j)], dtype=np.int64)
        eta_bar_int = int(solver.Value(eta))
        last_bound = float(solver.BestObjectiveBound())

        f_val, m, _covered = covering_gain_and_marginals(a_csr, p_arr, x_bar.astype(np.float64))
        if f_val > best_f + 1e-9:
            best_f, best_x = f_val, x_bar

        f_int = int(round(f_val * scale))
        if eta_bar_int <= f_int + gap_tol_int:
            converged = True
            break

        m_int = np.round(m * scale).astype(np.int64)
        terms = [(j, int(m_int[j])) for j in range(n_j) if x_bar[j] == 0 and m_int[j] > 0]
        # η ≤ f(x̄) + Σ_{j: x̄_j=0} m_j·x_j  — hệ số ở j đã bật (x̄_j=1) luôn =0 (chứng minh ở docstring)
        mdl.Add(eta <= f_int + sum(coef * x[j] for j, coef in terms))

    solve_time = time.time() - t0
    if best_x is None:
        return None

    chosen = [j for j in range(n_j) if best_x[j]]
    f2 = float(c_arr[chosen].sum())
    denom = max(abs(best_f), 1.0)
    gap_pct = abs(last_bound / scale - best_f) / denom * 100.0

    return {
        "epsilon": float(task.eps),
        "f1_covering_profit": best_f,
        "f2_cost": f2,
        "n_facilities": len(chosen),
        "chosen_candidates": chosen,
        "status": "BENDERS_CONVERGED" if converged else "BENDERS_MAX_ITERS_OR_TIME",
        "optimality_gap_pct": gap_pct,
        "solve_time_s": solve_time,
        "n_benders_iters": n_iters,
        "x_solution": best_x.tolist(),
    }
