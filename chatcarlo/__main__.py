"""CLI — AI（Claude Code）が非対話で叩ける入口。

  python -m chatcarlo validate <scene.yaml>
  python -m chatcarlo preview  <scene.yaml> [-o out.html]
  python -m chatcarlo trace    <scene.yaml> [-n 200] [--seed 42] [-o out.html]
  python -m chatcarlo animate  <scene.yaml> [-n 200] [--seed 42] [-o out.html]
  python -m chatcarlo plot     <dose.npz> [--scene scene.yaml] [--quantity dose|h10] [-o out.png]
  python -m chatcarlo xs <材料...> [--emin 10] [--emax 150] [-o out.png]
"""
from __future__ import annotations

import argparse
import sys


def cmd_validate(args) -> int:
    from .scene import load_scene
    scene = load_scene(args.scene)
    for w in scene.warnings:
        print(f"[警告] {w}")
    if scene.ok:
        print(f"OK: {args.scene} は有効なシーンです（物体 {len(scene.raw['geometry'])} 個）")
        return 0
    for e in scene.errors:
        print(f"[エラー] {e}", file=sys.stderr)
    return 1


def cmd_preview(args) -> int:
    from .preview import write_html
    from .scene import load_scene
    scene = load_scene(args.scene)
    if not scene.ok:
        for e in scene.errors:
            print(f"[エラー] {e}", file=sys.stderr)
        return 1
    for w in scene.warnings:
        print(f"[警告] {w}")
    out = args.out or args.scene.rsplit(".", 1)[0] + "_preview.html"
    write_html(scene, out, title=args.title or f"ChatCarlo — {args.scene}")
    print(f"プレビューを書き出しました: {out}")
    return 0


_TRACE_MAX_N = 2000


def cmd_trace(args) -> int:
    import numpy as np

    from .geometry import Geometry
    from .preview import write_html
    from .scene import load_scene
    from .source import sample_source_photons
    from .trajectory import TrajectoryRecorder, trajectories_to_json
    from .transport import transport_photons

    scene = load_scene(args.scene)
    if not scene.ok:
        for e in scene.errors:
            print(f"[エラー] {e}", file=sys.stderr)
        return 1
    for w in scene.warnings:
        print(f"[警告] {w}")

    n = int(args.n)
    if n > _TRACE_MAX_N:
        print(f"[警告] -n={n} は大きすぎるため{_TRACE_MAX_N}にクランプします"
              "（HTML描画性能のため）")
        n = _TRACE_MAX_N

    rng = np.random.default_rng(args.seed)
    geometry = Geometry(scene.raw["geometry"])
    pos, dirv, energy = sample_source_photons(scene.raw["source"], n, rng)
    recorder = TrajectoryRecorder()
    transport_photons(pos, dirv, energy, geometry, rng, recorder=recorder)
    trajectories = trajectories_to_json(recorder)

    out = args.out or args.scene.rsplit(".", 1)[0] + "_trace.html"
    write_html(scene, out, title=args.title or f"ChatCarlo trace — {args.scene}",
               trajectories=trajectories)
    print(f"光子{n}個の軌跡を書き出しました: {out}")
    return 0


_ANIMATE_MAX_N = 2000


def cmd_animate(args) -> int:
    import numpy as np

    from .animate import CATEGORY_LABELS, categorize_trajectories, write_html
    from .geometry import Geometry
    from .scene import load_scene
    from .source import sample_source_photons
    from .trajectory import TrajectoryRecorder, trajectories_to_json
    from .transport import transport_photons

    scene = load_scene(args.scene)
    if not scene.ok:
        for e in scene.errors:
            print(f"[エラー] {e}", file=sys.stderr)
        return 1
    for w in scene.warnings:
        print(f"[警告] {w}")

    n = int(args.n)
    if n > _ANIMATE_MAX_N:
        print(f"[警告] -n={n} は大きすぎるため{_ANIMATE_MAX_N}にクランプします"
              "（HTML描画性能のため）")
        n = _ANIMATE_MAX_N

    rng = np.random.default_rng(args.seed)
    geometry = Geometry(scene.raw["geometry"])
    pos, dirv, energy = sample_source_photons(scene.raw["source"], n, rng)
    recorder = TrajectoryRecorder()
    transport_photons(pos, dirv, energy, geometry, rng, recorder=recorder)
    trajectories = trajectories_to_json(recorder)

    counts = categorize_trajectories(trajectories)
    summary = " / ".join(f"{CATEGORY_LABELS[k]} {v}" for k, v in counts.items())
    print(f"転帰分類: {summary}")

    out = args.out or args.scene.rsplit(".", 1)[0] + "_anim.html"
    write_html(scene, out, trajectories,
               title=args.title or f"ChatCarlo animation — {args.scene}")
    print(f"光子{n}個のアニメーションを書き出しました: {out}")
    return 0


def cmd_run(args) -> int:
    from .diagnostics import (background_medium_warning, batch_shortage_message,
                               dose_map_Gy, grid_reliability_summary,
                               max_voxel_position_cm, near_source_air_warning,
                               unreliable_max_warning)
    from .geometry import Geometry
    from .scene import load_scene
    from .transport import run_transport

    scene = load_scene(args.scene)
    if not scene.ok:
        for e in scene.errors:
            print(f"[エラー] {e}", file=sys.stderr)
        return 1
    for w in scene.warnings:
        print(f"[警告] {w}")

    import os
    n_workers = args.workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    engine = getattr(args, "engine", "numpy")
    kernel_chunks = getattr(args, "kernel_chunks", 0)
    kernel_max_segments = getattr(args, "kernel_max_segments_per_history", 16)
    if kernel_chunks < 0:
        print("[エラー] --kernel-chunks は0以上で指定してください", file=sys.stderr)
        return 1
    track_uncertainty = not args.no_uncertainty
    n_histories_int = int(args.n_histories)
    if args.dose_grid and n_workers >= 2:
        # 並列時はワーカーごとに線量グリッドを持ち、親へpickleで集約するため、
        # メモリは概ね(ワーカー数+1)倍になる。部屋規模×細解像度では容易に数GBを
        # 超える（例: chest_roomの1cm解像度で約1.6GB/ワーカー）ため事前に見積もる。
        # ボクセルあたりのバイト数（chatcarlo/tally.pyのVoxelGrid）:
        #   統計なし: kerma + h10 = 8*2 = 16
        #   統計あり: 上記 + kerma_sum2 + h10_sum2 (8*2) + n_batches_hit (int32=4)
        #             + スナップショット2枚 (8*2) = 52（約3.25倍）
        # 親プロセスはスナップショットを確保しない（end_batchを呼ばないため遅延確保
        # されない）ので52は親側の過大評価だが、警告としては安全側に倒す。
        import numpy as np
        from .tally import VoxelGrid
        geometry_est = Geometry(scene.raw["geometry"])
        grid_est = VoxelGrid.from_bbox(geometry_est.bbox_min, geometry_est.bbox_max,
                                        args.resolution)
        bytes_per_voxel = 52 if track_uncertainty else 16
        est_gb = int(np.prod(grid_est.shape)) * bytes_per_voxel * (n_workers + 1) / 1e9
        if est_gb > 4.0:
            hint = ("解像度を粗くするか、workers数を減らすか、"
                    "--no-uncertainty で統計マップを無効化することを検討してください。"
                    if track_uncertainty else
                    "解像度を粗くするかworkers数を減らすことを検討してください。")
            print(f"[警告] --dose-grid（解像度{args.resolution}cm, グリッド形状"
                  f"{grid_est.shape}）と--workers {n_workers}の併用で、線量グリッドの"
                  f"メモリ使用量が概算{est_gb:.1f}GBに達します。{hint}")
    try:
        result = run_transport(scene, n_histories=n_histories_int, seed=args.seed,
                               batch_size=args.batch_size,
                               dose_grid=args.dose_grid, grid_resolution_cm=args.resolution,
                               n_workers=n_workers, track_uncertainty=track_uncertainty,
                               engine=engine, kernel_chunks=kernel_chunks,
                               kernel_max_segments_per_history=kernel_max_segments)
    except ValueError as exc:
        message = str(exc)
        if engine == "kernel" and args.dose_grid and "区間バッファ" in message:
            message += " --kernel-max-segments-per-history を増やして再実行してください。"
        print(f"[エラー] {message}", file=sys.stderr)
        return 1
    if engine == "kernel":
        import numba
        configured = kernel_chunks or min(numba.get_num_threads(), 8)
        final_batch_histories = n_histories_int % args.batch_size or min(args.batch_size, n_histories_int)
        effective = min(configured, final_batch_histories)
        print(f"kernel chunks: 設定値={configured}, 最終バッチ実効値={effective}")
    print(f"histories: {result.n_histories:,}")
    print(f"吸収（光電）割合: {result.fraction_absorbed:.4f}")
    print(f"脱出割合: {result.fraction_escaped:.4f}")
    print(f"平均相互作用回数/光子: {result.mean_scatter_events:.4f}")
    print(f"蛍光X線放出イベント数: {result.n_fluorescence:,}")

    if not track_uncertainty:
        print("統計誤差: --no-uncertainty により無効化されています"
              "（材料別SEM・グリッドの相対誤差マップは出力されません）。")
    else:
        shortage = batch_shortage_message(result.n_batches, n_histories_int, args.batch_size)
        if shortage:
            print(f"統計誤差: {shortage}")

    show_scalar_stats = track_uncertainty and result.n_batches >= 2
    print("材料別吸収エネルギー [MeV/history合計]:")
    for name, e_mev in sorted(result.energy_deposited_MeV.items(), key=lambda kv: -kv[1]):
        if show_scalar_stats:
            r = result.energy_deposited_rel_err.get(name, float("nan"))
            sem = result.energy_deposited_sem_MeV.get(name, float("nan"))
            if r == r:  # not nan（材料の真の付与エネルギーがゼロでない）
                print(f"  {name}: {e_mev:.6g}  (± {sem:.3g} MeV, 相対 {r * 100:.2f}%)")
                continue
        print(f"  {name}: {e_mev:.6g}")

    if result.n_photons_real is not None:
        src_cal = scene.raw["source"]
        if src_cal.get("ctdi_vol_mGy") is not None:
            print(f"線量校正: CTDIvol={src_cal['ctdi_vol_mGy']:g} mGy "
                  f"（{src_cal.get('ctdi_phantom', 'body')}ファントム）基準の実効光子数 "
                  f"= {result.n_photons_real:.6g}")
        else:
            print(f"光子数校正: mAs={src_cal['mas']:g} で照射野を通過する実光子数 "
                  f"= {result.n_photons_real:.6g}")

    if args.dose_grid:
        import numpy as np
        geometry = Geometry(scene.raw["geometry"])
        dose = dose_map_Gy(result.grid, geometry)
        h10 = result.grid.h10_map_pSv()
        n_histories = result.n_histories
        # scene.source.mas未指定時は1historyあたりの値のみ（相対値）を出力する。
        # mas指定時は実光子数でスケールした絶対値[Gy]・[pSv]も出す。
        # dose_per_history/h10_per_historyは既にn_historiesで割って「1光子あたり」に
        # なっているので、ここでの係数は実光子数そのもの（n_historiesでは割らない）。
        scale = result.n_photons_real if result.n_photons_real is not None else None
        dose_per_history = dose / n_histories
        h10_per_history = h10 / n_histories

        # R_dose≡R_kerma・R_h10≡R_h10track（docs/plan_statistical_uncertainty.md設計判断5、
        # いずれも決定的な定数倍のため）。track_uncertainty=Falseや M<2 でもnanで安全に返る。
        grid_n_batches = result.grid.n_batches if track_uncertainty else 0
        dose_R = result.grid.kerma_relative_error()
        h10_R = result.grid.h10_relative_error()
        n_hit = (result.grid.n_batches_hit if track_uncertainty
                 else np.zeros(result.grid.shape, dtype=np.int32))
        show_grid_stats = track_uncertainty and grid_n_batches >= 2
        dose_idx = np.unravel_index(int(np.argmax(dose_per_history)), dose_per_history.shape)
        h10_idx = np.unravel_index(int(np.argmax(h10_per_history)), h10_per_history.shape)

        print(f"線量グリッド: shape={result.grid.shape}, "
              f"resolution={result.grid.voxel_size_cm}cm")

        dose_line = f"  最大吸収線量 [Gy/history]: {dose_per_history.max():.6g}"
        if show_grid_stats:
            dose_line += (f"  (相対誤差 R={float(dose_R[dose_idx]):.3f}, "
                          f"寄与バッチ {int(n_hit[dose_idx])}/{grid_n_batches})")
        print(dose_line)

        print(f"  総カーマ: {result.grid.total_kerma_MeV():.6g} MeV "
              f"({result.grid.total_kerma_MeV() / n_histories * 1000:.6g} keV/history)")

        h10_line = f"  最大H*(10) [pSv/history]: {h10_per_history.max():.6g}"
        if show_grid_stats:
            h10_line += (f"  (相対誤差 R={float(h10_R[h10_idx]):.3f}, "
                         f"寄与バッチ {int(n_hit[h10_idx])}/{grid_n_batches})")
        print(h10_line)

        if scale is not None:
            cal = "CTDIvol校正済み" if scene.raw["source"].get("ctdi_vol_mGy") is not None else "mAs校正済み"
            print(f"  最大吸収線量 [Gy]（{cal}）: {(dose_per_history.max() * scale):.6g}")
            print(f"  最大H*(10) [pSv]（{cal}）: {(h10_per_history.max() * scale):.6g}")

        if show_grid_stats:
            summary = grid_reliability_summary(dose_R, n_hit, grid_n_batches)
            h10_summary = grid_reliability_summary(h10_R, n_hit, grid_n_batches)
            print(f"  グリッド統計: 線量R<0.10 {summary['frac_low_r'] * 100:.1f}% / "
                  f"H*(10)R<0.10 {h10_summary['frac_low_r'] * 100:.1f}% / "
                  f"寄与バッチ0（未到達） {summary['frac_unreached'] * 100:.1f}%")
            dose_reliab = unreliable_max_warning(float(dose_R[dose_idx]), int(n_hit[dose_idx]), grid_n_batches)
            if dose_reliab:
                print(f"[警告] 最大吸収線量: {dose_reliab}")
            h10_reliab = unreliable_max_warning(float(h10_R[h10_idx]), int(n_hit[h10_idx]), grid_n_batches)
            if h10_reliab:
                print(f"[警告] 最大H*(10): {h10_reliab}")

        # 「最大」統計が非物理的な位置（背景=空気ボクセル）に落ちていないかを診断する。
        # 詳細はdocs/lessons_learned.md参照。
        src_pos = scene.raw["source"]["position"]
        dose_pos = max_voxel_position_cm(result.grid, dose_per_history)
        dose_mat = str(geometry.material_at(dose_pos[None, :])[0])
        dose_warn = background_medium_warning(dose_mat, geometry.background)
        if dose_warn:
            print(f"[警告] {dose_warn}")

        h10_pos = max_voxel_position_cm(result.grid, h10_per_history)
        h10_mat = str(geometry.material_at(h10_pos[None, :])[0])
        h10_dist = float(np.linalg.norm(h10_pos - np.asarray(src_pos, dtype=float)))
        nearest_obj = geometry.nearest_object_distance_cm(src_pos)
        h10_warn = near_source_air_warning(h10_mat, geometry.background, h10_dist, nearest_obj)
        if h10_warn:
            print(f"[警告] {h10_warn}")
        if args.dose_out:
            save_kwargs = dict(
                dose_per_history_Gy=dose_per_history,
                h10_per_history_pSv=h10_per_history,
                kerma_keV=result.grid.kerma_keV,
                h10_track_pSv_cm3=result.grid.h10_track_pSv_cm3,
                origin_cm=result.grid.origin_cm,
                voxel_size_cm=result.grid.voxel_size_cm,
                shape=np.array(result.grid.shape),
            )
            if scale is not None:
                save_kwargs["dose_Gy"] = dose_per_history * scale
                save_kwargs["h10_pSv"] = h10_per_history * scale
            if track_uncertainty:
                # SEM = R * mean。Rは量の換算（密度・校正係数）に対して不変な決定的
                # 定数倍のため（設計判断5）、絶対値・相対値どちらの出力にもこのまま使える。
                save_kwargs["rel_err_dose"] = dose_R
                save_kwargs["rel_err_h10"] = h10_R
                save_kwargs["sem_dose_per_history_Gy"] = dose_R * dose_per_history
                save_kwargs["sem_h10_per_history_pSv"] = h10_R * h10_per_history
                save_kwargs["n_batches"] = np.array(grid_n_batches)
                save_kwargs["n_batches_hit"] = n_hit
            np.savez(args.dose_out, **save_kwargs)
            print(f"線量グリッドを書き出しました: {args.dose_out}")
    return 0


def cmd_plot(args) -> int:
    from .plotting import plot_dose_npz

    out = args.out or args.npz.rsplit(".", 1)[0] + "_maps.png"
    ok = plot_dose_npz(args.npz, out, quantity=args.quantity, axis=args.axis,
                        pos_cm=args.pos, scene_path=args.scene)
    if not ok:
        print("[エラー] 線量グリッドが全てゼロのため描画できません", file=sys.stderr)
        return 1
    print(f"線量マップを書き出しました: {out}")
    return 0


def cmd_xs(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .materials import mu_en_rho, mu_rho, resolve

    e = np.logspace(np.log10(args.emin), np.log10(args.emax), 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in args.materials:
        name, rho, _ = resolve(m)
        ax1.loglog(e, mu_rho(m, e), label=f"{m} (ρ={rho:.3g})")
        ax2.loglog(e, mu_en_rho(m, e), label=m)
    ax1.set_title("Mass attenuation μ/ρ  [cm²/g]")
    ax2.set_title("Mass energy-absorption μen/ρ  [cm²/g]")
    for ax in (ax1, ax2):
        ax.set_xlabel("Photon energy [keV]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ChatCarlo cross-section data (xraylib / EPDL, NIST-consistent)")
    fig.tight_layout()
    out = args.out or "xs.png"
    fig.savefig(out, dpi=140)
    print(f"断面積プロットを書き出しました: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="chatcarlo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="scene.yaml を検証")
    pv.add_argument("scene")
    pv.set_defaults(func=cmd_validate)

    pp = sub.add_parser("preview", help="ジオメトリープレビューHTMLを生成")
    pp.add_argument("scene")
    pp.add_argument("-o", "--out")
    pp.add_argument("--title")
    pp.set_defaults(func=cmd_preview)

    pt = sub.add_parser("trace", help="光子軌跡を3D可視化するHTMLを生成（小history）")
    pt.add_argument("scene")
    pt.add_argument("-n", type=int, default=200, help="軌跡を記録する光子数（既定200、上限2000）")
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("-o", "--out")
    pt.add_argument("--title")
    pt.set_defaults(func=cmd_trace)

    pa = sub.add_parser("animate", help="教育用の光子軌跡アニメーションHTMLを生成（小history）")
    pa.add_argument("scene")
    pa.add_argument("-n", type=int, default=200, help="軌跡を記録する光子数（既定200、上限2000）")
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("-o", "--out")
    pa.add_argument("--title")
    pa.set_defaults(func=cmd_animate)

    pr = sub.add_parser("run", help="光子輸送を実行")
    pr.add_argument("scene")
    pr.add_argument("-n", "--n-histories", type=float, default=1e5)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--dose-grid", action="store_true", help="ボクセル吸収線量タリーを有効化")
    pr.add_argument("--resolution", type=float, default=5.0, help="線量グリッド解像度[cm]（既定5cm）")
    pr.add_argument("--dose-out", help="線量グリッドを.npzに書き出すパス")
    pr.add_argument("--batch-size", type=int, default=200_000,
                     help="統計バッチ＝輸送バッチのhistory数（既定200,000）。"
                          "統計バッチ数M=ceil(n_histories/batch_size)が2未満だと"
                          "相対誤差Rが推定できない（nanになり、対処法がメッセージで"
                          "出る）ため、小さいnで統計を見たい場合に下げる。"
                          "同一seedでもbatch_sizeを変えると結果はビット一致しない"
                          "（乱数の消費ブロック長が変わるため。統計的には同等）。")
    pr.add_argument("--no-uncertainty", action="store_true",
                     help="統計不確かさ（相対誤差マップ・材料別SEM）の積算を無効化する。"
                          "既定は有効。--dose-grid併用時は線量グリッドのメモリが"
                          "ボクセルあたり16→52バイト（約3.25倍）になるため、"
                          "細解像度×多ワーカーでメモリが足りない場合の逃げ道。"
                          "物理結果（線量値そのもの）は有効/無効でビット一致する。")
    pr.add_argument("--workers", type=int, default=1,
                     help="並列ワーカー数（既定1=直列、0=CPU数自動）。マルチプロセスで"
                          "n_historiesを分散し物理は変えない。ワーカーごとにimportと"
                          "断面積テーブル構築の固定費（数百ms〜1秒程度）がかかるため、"
                          "n_historiesが小さいと逆に遅くなる場合がある。同一seedでも"
                          "workers数を変えると結果はビット一致しない（統計的には同等。"
                          "docs/plan_phase3_parallel.md参照）。--dose-grid併用時は"
                          "線量グリッドをワーカーごとに持つためメモリが(workers+1)倍"
                          "になる点に注意（細解像度では数GB級、超過見込み時は警告を出す）")
    pr.add_argument("--engine", choices=["numpy", "kernel"], default="numpy",
                    help="輸送エンジン（既定numpy、kernelは単色parallel・box限定のopt-in）")
    pr.add_argument("--kernel-chunks", type=int, default=0,
                    help="kernelのNumbaチャンク数（0=自動、既定はmin(numba threads, 8)）")
    pr.add_argument("--kernel-max-segments-per-history", type=int, default=16,
                    help="kernel dose-grid用の履歴あたり区間バッファ上限（既定16）")
    pr.set_defaults(func=cmd_run)

    ppl = sub.add_parser("plot", help="線量/H*(10)マップの断面図を生成")
    ppl.add_argument("npz")
    ppl.add_argument("-o", "--out")
    ppl.add_argument("--scene", help="指定するとジオメトリー輪郭を断面に重ねる")
    ppl.add_argument("--quantity", choices=["dose", "h10", "relerr-dose", "relerr-h10"],
                      default="dose",
                      help="relerr-*は相対誤差Rマップ（統計未到達ボクセルは別色でマスク）。"
                           "--no-uncertaintyで生成した.npzには無い")
    ppl.add_argument("--axis", choices=["x", "y", "z"], default=None,
                      help="未指定なら最大値ボクセルを通る3断面（既定）")
    ppl.add_argument("--pos", type=float, default=None, help="--axis指定時の断面位置[cm]")
    ppl.set_defaults(func=cmd_plot)

    px = sub.add_parser("xs", help="断面積カーブを描画")
    px.add_argument("materials", nargs="+")
    px.add_argument("--emin", type=float, default=10.0)
    px.add_argument("--emax", type=float, default=150.0)
    px.add_argument("-o", "--out")
    px.set_defaults(func=cmd_xs)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
