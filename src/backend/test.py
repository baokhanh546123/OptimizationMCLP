"""
test.py — End-to-end smoke pipeline cho MO-MCLP (Xuân Hương ward).

Luồng:
  1. Load boundary + demand + roads (path tương đối từ repo root)
  2. Sinh candidate set (grid + street)
  3. Build MCLP_Data (sparse coverage)
  4. ε-constraint sweep (early_stop + re_solve); eps_mode="geographic" thêm
     bước Spatial Decomposition (KMeans/PAM) trước sweep — xem
     Optimization.reduce_candidates_geographic() trong model/optimization.py.
  5. (Tuỳ chọn) export bản đồ Pareto HTML / JPEG

Chạy từ repo root:
  PYTHONPATH=src/backend python src/backend/test.py
  PYTHONPATH=src/backend python src/backend/test.py --mode tight --export-map
  PYTHONPATH=src/backend python src/backend/test.py --n-points 8 --time-limit 120
  PYTHONPATH=src/backend python src/backend/test.py --mode geographic --n-cls 8 --mode-cls kmeans
  PYTHONPATH=src/backend python src/backend/test.py --mode geographic --n-cls 6 --mode-cls pam --geo-max-per-cluster 40
  PYTHONPATH=src/backend python src/backend/test.py --mode benders --benders-max-iters 100 --benders-master-time-limit 3
  PYTHONPATH=src/backend python src/backend/test.py --mode tight --early-stop-patience 3 --re-solve-gap-threshold 15
  PYTHONPATH=src/backend python src/backend/test.py --mode notebook --no-hint --max-radius 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

from dataclass.MCLP import MCLP_Data
from model.optimization import Optimization
from utils.candidate_grid import build_candidate_set
from utils.load_data import load_places


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data"

BOUNDARY_PATH = DATA_DIR / "bounary" / "boundary.geojson"
DEMAND_PATH = DATA_DIR / "Xuanhuongward" / "Xuan Huong Wards_featured.geojson"
ROADS_PATH = DATA_DIR / "Xuanhuongward" / "Xuan Huong Wards_roads.geojson"
OUTPUT_MAP_DIR = REPO_ROOT / "outputs" / "maps"


def _check_data_files() -> None:
    missing = [p for p in (BOUNDARY_PATH, DEMAND_PATH, ROADS_PATH) if not p.exists()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Thiếu file dữ liệu (chạy từ repo có thư mục data/):\n{lines}\n"
            f"REPO_ROOT={REPO_ROOT}"
        )


def load_ward_polygon(boundary_path: Path = BOUNDARY_PATH, index: int = 2):
    """Đọc boundary GeoJSON và lấy polygon ward theo index (mặc định 2 = Xuân Hương)."""
    wards_df = gpd.read_file(boundary_path)
    if index < 0 or index >= len(wards_df):
        raise IndexError(
            f"boundary index={index} ngoài phạm vi [0, {len(wards_df) - 1}]"
        )
    return wards_df.geometry.iloc[index]


def build_pipeline(
    *,
    p_max: int = 3,
    grid_spacing_m: float = 220.0,
    street_spacing_m: float = 80.0,
    ward_index: int = 2,
):
    """Load data → candidate → MCLP_Data."""
    _check_data_files()

    print(f"[INFO] REPO_ROOT = {REPO_ROOT}")
    print(f"[INFO] DATA_DIR  = {DATA_DIR}")

    wards_polygon = load_ward_polygon(BOUNDARY_PATH, index=ward_index)
    print(f"[OK] Ward polygon (boundary index={ward_index})")

    demand_gdf = load_places(str(DEMAND_PATH))
    print(f"[OK] Demand POIs: {len(demand_gdf)}")

    roads_path = str(ROADS_PATH)
    candidate_gdf, candidate_cost = build_candidate_set(
        roads=roads_path,
        ward_polygon_wgs84=wards_polygon,
        grid_spacing_m=grid_spacing_m,
        street_spacing_m=street_spacing_m,
    )

    mclp = MCLP_Data.from_geodata(
        demand_gdf,
        candidate_gdf,
        candidate_cost=candidate_cost,
        P_max=p_max,
    )
    avg_cover = float(mclp.a.sum(axis=1).mean())
    print(
        f"\n[OK] Coverage matrix a_ij shape: {mclp.a.shape}, "
        f"trung bình mỗi POI được phủ bởi {avg_cover:.1f} candidate, "
        f"nnz={mclp.a.nnz}"
    )
    return demand_gdf, candidate_gdf, roads_path, wards_polygon, mclp


def run_sweep(
    mclp: MCLP_Data,
    demand_gdf,
    candidate_gdf,
    roads_path: str,
    wards_polygon,
    *,
    mode: str = "notebook",
    n_points: int = 12,
    time_limit_s: int = 300,
    relative_gap: float = 0.05,
    n_parallel: int = 1,
    workers_per_solve: int = 4,
    early_stop: bool = True,
    re_solve_flagged: bool = True,
    auto_reduce: bool = False,
    max_candidates: int = 1200,
    re_solve_time_limit: int = 600,
    n_geo_clusters: int = 6,
    geo_cluster_method: str = "kmeans",
    geo_max_per_cluster: int | None = None,
    benders_max_iters: int = 200,
    benders_master_time_limit_s: float = 5.0,
    early_stop_patience: int = 2,
    early_stop_tol: float = 1e-6,
    re_solve_gap_threshold: float = 20.0,
    use_hint: bool = True,
    max_radius_m: float | None = None,
    scale: int = 10 ** 6,
    utm_epsg: int = 32648,
):
    """Chạy ε-constraint sweep và in bảng kết quả.

    mode="geographic": Spatial Decomposition (KMeans/PAM) chạy TRƯỚC sweep.
    mode="benders": Benders Decomposition (exact) — xem model/benders_solver.py.
    """
    opt = Optimization(
        data=mclp,
        demand_set=demand_gdf,
        candidate_set=candidate_gdf,
        roads_set=roads_path,
        ward_polygon_wgs84=wards_polygon,
        n_points=n_points,
        time_limit_s=time_limit_s,
        relative_gap=relative_gap,
        n_parallel=n_parallel,
        workers_per_solve=workers_per_solve,
        use_hint=use_hint,
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        re_solve_flagged=re_solve_flagged,
        re_solve_time_limit=re_solve_time_limit,
        re_solve_gap_threshold=re_solve_gap_threshold,
        max_candidates=max_candidates,
        max_radius_m=max_radius_m,
        SCALE=scale,
        utm_epsg=utm_epsg,
        eps_mode=mode,
        n_geo_clusters=n_geo_clusters,
        geo_cluster_method=geo_cluster_method,
        geo_max_per_cluster=geo_max_per_cluster,
        benders_max_iters=benders_max_iters,
        benders_master_time_limit_s=benders_master_time_limit_s,
    )

    extra_geo = (
        f"  n_cls={n_geo_clusters}  mode_cls={geo_cluster_method}"
        f"{f'  geo_max_per_cluster={geo_max_per_cluster}' if geo_max_per_cluster else ''}"
        if mode == "geographic" else ""
    )
    extra_benders = (
        f"  benders_max_iters={benders_max_iters}  benders_master_time_limit={benders_master_time_limit_s}s"
        if mode == "benders" else ""
    )
    print(
        f"\n[RUN] ε-sweep  mode={mode}  n_points={n_points}  "
        f"time_limit={time_limit_s}s  early_stop={early_stop}  "
        f"re_solve={re_solve_flagged}  auto_reduce={auto_reduce}{extra_geo}{extra_benders}"
    )
    results = opt.epsilon_constraint_sweep(auto_reduce=auto_reduce)

    has_iters = any(r.get("n_benders_iters") is not None for r in results)
    has_diag = any(r.get("submodular_diagnostic_gap_pct") is not None for r in results)
    print("\n" + "=" * (90 if has_iters or has_diag else 78))
    header = (
        f"{'ε':>10}  {'f1':>10}  {'f2':>8}  {'n_fac':>5}  "
        f"{'gap%':>7}  {'flag':>5}"
    )
    if has_diag:
        header += f"  {'diag%':>7}"
    if has_iters:
        header += f"  {'iters':>5}"
    header += "  status"
    print(header)
    print("-" * len(header))
    for r in results:
        line = (
            f"{r['epsilon']:10.4f}  "
            f"{r['f1_covering_profit']:10.3f}  "
            f"{r['f2_cost']:8.4f}  "
            f"{r['n_facilities']:5d}  "
            f"{r['optimality_gap_pct']:7.2f}  "
            f"{str(r.get('flagged_non_monotonic', False))!s:>5}"
        )
        if has_diag:
            dg = r.get("submodular_diagnostic_gap_pct")
            line += f"  {dg:7.2f}" if dg is not None else f"  {'—':>7}"
        if has_iters:
            ni = r.get("n_benders_iters")
            line += f"  {ni:5d}" if ni is not None else f"  {'—':>5}"
        line += f"  {r.get('status', '')}"
        print(line)
    print("=" * len(header))
    return results


def export_maps(
    candidate_gdf,
    results: list,
    demand_gdf,
    *,
    map_mode: str = "dot",
    only_trusted: bool = False,
):
    """Export Pareto map HTML + JPEG vào outputs/maps/."""
    try:
        from visualize.visualize_plot import export_pareto_map
    except ImportError as exc:
        print(f"[WARN] Không import được visualize ({exc}) — bỏ qua export map.")
        return {}

    paths = export_pareto_map(
        candidate_gdf=candidate_gdf,
        sweep_results=results,
        demand_gdf=demand_gdf,
        out_dir=OUTPUT_MAP_DIR,
        stem="xuanhuong_pareto",
        formats=("html", "jpeg"),
        mode=map_mode,
        only_trusted=only_trusted,
    )
    for kind, path in paths.items():
        print(f"[OK] Map {kind}: {path}")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MO-MCLP end-to-end test pipeline (Xuân Hương)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_mode = p.add_argument_group("Solver mode")
    g_mode.add_argument(
        "--mode",
        choices=("notebook", "tight", "geographic", "benders"),
        default="notebook",
        help=(
            "eps_mode: notebook=Σc; tight=top-P_max; geographic=Spatial Decomposition; "
            "benders=Benders Decomposition exact"
        ),
    )

    g_data = p.add_argument_group("Data / candidates")
    g_data.add_argument("--p-max", type=int, default=3, help="P_max — số cơ sở tối đa")
    g_data.add_argument("--ward-index", type=int, default=2, help="Index polygon ward trong boundary")
    g_data.add_argument("--grid-spacing", type=float, default=220.0, help="Grid candidate spacing (m)")
    g_data.add_argument("--street-spacing", type=float, default=80.0, help="Street candidate spacing (m)")
    g_data.add_argument(
        "--max-radius", type=float, default=None, dest="max_radius_m",
        help="Bán kính phủ tối đa (m); None = default coverage builder",
    )
    g_data.add_argument("--utm-epsg", type=int, default=32648, help="UTM EPSG (default 32648 = UTM 48N)")
    g_data.add_argument(
        "--auto-reduce", action="store_true",
        help="Giảm |J| theo coverage weight — chỉ khi --mode != geographic",
    )
    g_data.add_argument("--max-candidates", type=int, default=1200, help="Trần |J| khi auto-reduce / quota geo")

    g_eps = p.add_argument_group("ε-sweep")
    g_eps.add_argument("--n-points", type=int, default=12, help="Số điểm ε trên Pareto front")
    g_eps.add_argument("--time-limit", type=int, default=300, dest="time_limit_s", help="Time limit mỗi điểm ε (s)")
    g_eps.add_argument("--relative-gap", type=float, default=0.05, help="CP-SAT relative_gap_limit")
    g_eps.add_argument("--n-parallel", type=int, default=1, help="Số điểm ε giải song song")
    g_eps.add_argument("--workers", type=int, default=4, dest="workers_per_solve", help="CP-SAT workers mỗi điểm ε")
    g_eps.add_argument("--scale", type=int, default=10**6, help="SCALE integer CP-SAT")
    g_eps.add_argument("--no-hint", action="store_true", help="Tắt greedy warm-start hint")

    g_stop = p.add_argument_group("Early-stop / re-solve")
    g_stop.add_argument("--no-early-stop", action="store_true", help="Tắt sweep early-stop khi f1 bão hòa")
    g_stop.add_argument(
        "--early-stop-patience", type=int, default=2, dest="early_stop_patience",
        help="Số điểm ε liên tiếp f1 không tăng trước khi dừng sweep",
    )
    g_stop.add_argument(
        "--early-stop-tol", type=float, default=1e-6, dest="early_stop_tol",
        help="Sai số tuyệt đối khi so f1 cho early-stop",
    )
    g_stop.add_argument("--no-re-solve", action="store_true", help="Tắt re-solve điểm non-monotonic / gap cao")
    g_stop.add_argument(
        "--re_solve-time-limit", type=int, default=120, dest="re_solve_time_limit",
        help="Time limit (s) mỗi điểm khi re-solve",
    )
    g_stop.add_argument(
        "--re-solve-gap-threshold", type=float, default=20.0, dest="re_solve_gap_threshold",
        help="Re-solve nếu optimality_gap_pct > ngưỡng này (%%)",
    )

    g_geo = p.add_argument_group("Geographic decomposition (--mode geographic)")
    g_geo.add_argument("--n-cls", type=int, default=6, dest="n_geo_clusters", help="Số cụm không gian")
    g_geo.add_argument(
        "--mode-cls", choices=("kmeans", "pam"), default="kmeans", dest="geo_cluster_method",
        help="kmeans (Lloyd) hoặc pam (k-medoids; fallback kmeans nếu |J|>3000)",
    )
    g_geo.add_argument(
        "--geo-max-per-cluster", type=int, default=None, dest="geo_max_per_cluster",
        help="Quota candidate mỗi cụm (mặc định: max_candidates / n_cls)",
    )

    g_ben = p.add_argument_group("Benders (--mode benders)")
    g_ben.add_argument(
        "--benders-max-iters", type=int, default=200, dest="benders_max_iters",
        help="Số vòng cutting-plane tối đa mỗi điểm ε",
    )
    g_ben.add_argument(
        "--benders-master-time-limit", type=float, default=5.0, dest="benders_master_time_limit_s",
        help="Trần thời gian MỖI lần giải master CP-SAT (giây)",
    )

    g_map = p.add_argument_group("Export map")
    g_map.add_argument("--export-map", action="store_true", help="Export Pareto map HTML + JPEG")
    g_map.add_argument("--map-mode", choices=("heatmap", "dot"), default="dot")
    g_map.add_argument("--only-trusted", action="store_true", help="Chỉ vẽ điểm không flagged_non_monotonic")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    demand_gdf, candidate_gdf, roads_path, wards_polygon, mclp = build_pipeline(
        p_max=args.p_max,
        grid_spacing_m=args.grid_spacing,
        street_spacing_m=args.street_spacing,
        ward_index=args.ward_index,
    )

    results = run_sweep(
        mclp,
        demand_gdf,
        candidate_gdf,
        roads_path,
        wards_polygon,
        mode=args.mode,
        n_points=args.n_points,
        time_limit_s=args.time_limit_s,
        relative_gap=args.relative_gap,
        n_parallel=args.n_parallel,
        workers_per_solve=args.workers_per_solve,
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_tol=args.early_stop_tol,
        re_solve_flagged=not args.no_re_solve,
        re_solve_time_limit=args.re_solve_time_limit,
        re_solve_gap_threshold=args.re_solve_gap_threshold,
        auto_reduce=args.auto_reduce,
        max_candidates=args.max_candidates,
        n_geo_clusters=args.n_geo_clusters,
        geo_cluster_method=args.geo_cluster_method,
        geo_max_per_cluster=args.geo_max_per_cluster,
        benders_max_iters=args.benders_max_iters,
        benders_master_time_limit_s=args.benders_master_time_limit_s,
        use_hint=not args.no_hint,
        max_radius_m=args.max_radius_m,
        scale=args.scale,
        utm_epsg=args.utm_epsg,
    )

    if not results:
        print("[WARN] Không có nghiệm Pareto nào.")
        return 1

    if args.export_map:
        export_maps(
            candidate_gdf,
            results,
            demand_gdf,
            map_mode=args.map_mode,
            only_trusted=args.only_trusted,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
