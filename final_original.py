#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캡스톤디자인 2조 · 최종 확정 알고리즘 + 통계적 신뢰구간(Wilson CI & Moving Block Bootstrap) 통합판

과제: 낙동강 칠서 지점 유해남조류 >= 1,000 cells/mL 위험 3일 선행예측 (XGBoost + SHAP + 통계적 신뢰구간)

[기존 확정 원칙 유지 (final_mode_BAEK.py 기반)]
 1) 2024~2025년 자료는 로드 직후 봉인(모델링·평가 모두 금지), 2016~2023년(445건)만 사용
 2) 교수님 추천 7변수 구조:
    Temp_last + Solar_mean_7d + DOY_sin + Flow_mean_3d + SPI30d_gamma + Dry_days_lt1mm + TOC 1종
 3) 하이퍼파라미터 = B6 설정 고정 (주피터 커널 멈춤 방지를 위해 n_jobs=1 적용)
 4) 임계값 = 성해영 절차: 2020~2023 확장형 시간순 검증(212건)에서 목표 재현율(0.85) 이상 중 FPR 최소
 5) SHAP(TreeSHAP, XGBoost 내장 pred_contribs / pred_interactions)으로 변수 기여와 TOC×가뭄 상호작용 시각화

[신규 추가된 2가지 통계적 신뢰구간 모듈]
 A) 시스템 오경보율(FDR·FPR) & 탐지율(Recall): 윌슨 스코어(Wilson Score) 95% 신뢰구간
 B) 개별 예측 원점수(Score) 불확실성: 이동 블록 부트스트랩(Moving Block Bootstrap, L=6, B=500, 95% CI)
    - 부트스트랩 내부 scale_pos_weight는 원본 학습셋 값으로 고정하여 인위적 로그오즈 변동 차단
    - 보정되지 않은 원점수이므로 '발생확률'이 아닌 '모델 원점수 95% 불확실성 구간'으로 표기

[실행 방법]
 1) 주피터 노트북 셀에서 실행:
    from final_model_with_CI import run_pipeline
    py_tbl, zone_tbl, oof_mbb = run_pipeline(
        input_path="공통데이터.xlsx", output_dir="final_out_ci", n_boot=500, block_size=6
    )
    (※ 빠른 동작 점검 시 n_boot=50으로 먼저 확인 후 최종 산출 시 n_boot=500 권장)

 2) 터미널(CLI)에서 실행:
    python final_model_with_CI.py --input "공통데이터.xlsx" --output final_out_ci --n-boot 500 --block-size 6

필요 패키지: numpy, pandas, scipy, openpyxl, scikit-learn, xgboost(>=1.7), matplotlib
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 0. 확정 설정
# ------------------------------------------------------------------
MODEL_SHEET = "모델 공통데이터"
DAILY_SHEET = "일별 원자료"
TARGET = "Risk_ge1000"

SEALED_FROM_YEAR = 2024
VALIDATION_YEARS = [2020, 2021, 2022, 2023]

BASE_FEATURES = [
    "Temp_last",        # 최근 실측 수온            (백승재·윤민준)
    "Solar_mean_7d",    # 7일 평균 일사량           (류다현·윤민준, 교수님 적극 고려)
    "DOY_sin",          # 연중 위치(sin)            (백승재·윤민준; cos는 수온과 r=-0.876으로 제외)
    "Flow_mean_3d",     # 3일 평균 유량             (류지원: 7일보다 우수, 윤민준: 단기 수문 일원화)
    "SPI30d_gamma",     # 30일 표준강수지수(장기 가뭄)
    "Dry_days_lt1mm",   # 1mm 미만 연속 건조일수(단기 가뭄)
]
DROUGHT_FEATURES = ["SPI30d_gamma", "Dry_days_lt1mm"]

TOC_CANDIDATES = ["TOC_last", "TOC_state_mean_7d"]
TOC_RULE_TOLERANCE = 0.01

# B6 하이퍼파라미터 고정 (주피터 OpenMP 충돌 방지를 위해 n_jobs=1)
XGB_PARAMS = dict(
    n_estimators=160, max_depth=2, learning_rate=0.035,
    min_child_weight=5, reg_lambda=6.0, subsample=0.85, colsample_bytree=0.85,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    random_state=42, n_jobs=1,
)
SEED = 42
STABILITY_SEEDS = list(range(10))

TARGET_RECALL = 0.85
SENSITIVITY_RECALLS = [0.80, 0.85, 0.90]
COST_RATIOS = [1, 3, 5]

# 신뢰구간 추정 설정 (기본값: L=6, B=500, 95% CI)
CI_LEVEL = 0.95
MBB_BLOCK_SIZE = 6
MBB_N_BOOT = 500


# ------------------------------------------------------------------
# 1. 데이터 로드 · 봉인 · 보조 TOC 열 생성
# ------------------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {os.path.abspath(path)}\n"
            f"현재 작업 폴더({os.getcwd()})에 파일을 두거나 --input 경로를 확인하세요."
        )
    d = pd.read_excel(path, sheet_name=MODEL_SHEET)
    for c in ["target_date", "cutoff_date"]:
        d[c] = pd.to_datetime(d[c])
    d = d.sort_values("target_date").reset_index(drop=True)
    d["year"] = d["target_date"].dt.year

    if "DOY_sin" not in d.columns:
        doy = d["cutoff_date"].dt.dayofyear
        d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)

    if "TOC_weighted7" not in d.columns or "TOC_age_actual" not in d.columns:
        try:
            daily = pd.read_excel(path, sheet_name=DAILY_SHEET)
            daily["Date"] = pd.to_datetime(daily["Date"])
            toc = daily.set_index("Date")["TOC"].dropna()
            w7, age_list = [], []
            for cutoff in d["cutoff_date"]:
                num = den = 0.0
                last_age = np.nan
                for age in range(8):
                    day = cutoff - pd.Timedelta(days=age)
                    if day in toc.index and np.isfinite(toc.loc[day]):
                        if np.isnan(last_age):
                            last_age = age
                        if age <= 6:
                            w = 2.0 ** (-age / 3.0)
                            num += w * float(toc.loc[day])
                            den += w
                w7.append(num / den if den > 0 else np.nan)
                age_list.append(last_age)
            if "TOC_weighted7" not in d.columns:
                d["TOC_weighted7"] = w7
            if "TOC_age_actual" not in d.columns:
                d["TOC_age_actual"] = age_list
        except Exception as e:
            print(f"[알림] 일별 TOC 기반 열 생성 불가 ({e})")

    sealed = d["year"] >= SEALED_FROM_YEAR
    print(f"[봉인] {SEALED_FROM_YEAR}년 이후 {int(sealed.sum())}건은 모델링·평가에 사용하지 않음")
    dev = d.loc[~sealed].reset_index(drop=True)
    return d, dev


def check_columns(dev, toc_candidates):
    need = BASE_FEATURES + ["target_date", "cutoff_date", TARGET]
    miss = [c for c in need if c not in dev.columns]
    if miss:
        raise ValueError(f"필수 열 없음: {miss}")
    avail = [c for c in toc_candidates if c in dev.columns]
    if TOC_CANDIDATES[0] not in avail:
        raise ValueError("기본 TOC 열(TOC_last)이 없습니다.")
    return avail


# ------------------------------------------------------------------
# 2. 모델 · 확장형 시간순 검증 · 윌슨 스코어 신뢰구간
# ------------------------------------------------------------------
def build_model(y_train, seed=SEED, fixed_spw=None):
    """fixed_spw가 주어지면 부트스트랩 표본 변동과 무관하게 원본 scale_pos_weight를 고정 적용."""
    p = dict(XGB_PARAMS)
    p["random_state"] = seed
    if fixed_spw is not None:
        p["scale_pos_weight"] = float(fixed_spw)
    else:
        p["scale_pos_weight"] = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return XGBClassifier(**p)


def rolling_oof(dev, features, seed=SEED):
    """확장형 시간순 검증: 각 검증연도의 첫 입력 마감일보다 앞선 목표일만 학습."""
    rows = []
    for yr in VALIDATION_YEARS:
        ev = dev["year"] == yr
        if ev.sum() == 0:
            continue
        tr = dev["target_date"] < dev.loc[ev, "cutoff_date"].min()
        ytr = dev.loc[tr, TARGET].to_numpy()
        m = build_model(ytr, seed).fit(dev.loc[tr, features], ytr)
        p = m.predict_proba(dev.loc[ev, features])[:, 1]
        part = dev.loc[ev, ["target_date", "year", TARGET]].copy()
        part["score"] = p
        part["n_train"] = int(tr.sum())
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def yearly_ap(oof):
    out = {}
    for yr, g in oof.groupby("year"):
        y = g[TARGET].to_numpy()
        out[int(yr)] = average_precision_score(y, g["score"]) if 0 < y.sum() < len(y) else np.nan
    return out


def summarize(name, oof):
    ap = yearly_ap(oof)
    vals = np.array([v for v in ap.values() if np.isfinite(v)])
    y = oof[TARGET].to_numpy()
    return {
        "model": name,
        **{f"AP_{k}": round(v, 4) for k, v in ap.items()},
        "AP_mean": round(vals.mean(), 4),
        "AP_sd": round(vals.std(ddof=1), 4) if len(vals) > 1 else np.nan,
        "AP_pooled": round(average_precision_score(y, oof["score"]), 4),
        "ROC_AUC_pooled": round(roc_auc_score(y, oof["score"]), 4),
    }


def wilson_ci(k, n, ci_level=CI_LEVEL):
    """이항 비율(FDR, FPR, Recall, Precision)의 윌슨 스코어 신뢰구간."""
    if n <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * n)) / n)) / denom
    return round(float(max(0.0, center - half)), 4), round(float(min(1.0, center + half)), 4)


def confusion_at(y, s, t):
    pred = s >= t
    tp = int((pred & (y == 1)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics_row(y, s, t, ci_level=CI_LEVEL):
    tp, fp, fn, tn = confusion_at(y, s, t)
    n_pos = tp + fn
    n_neg = fp + tn
    n_alert = tp + fp
    rec = tp / n_pos if n_pos else np.nan
    fpr = fp / n_neg if n_neg else np.nan
    prec = tp / n_alert if n_alert else np.nan
    fdr = fp / n_alert if n_alert else np.nan

    rec_lo, rec_hi = wilson_ci(tp, n_pos, ci_level)
    fpr_lo, fpr_hi = wilson_ci(fp, n_neg, ci_level)
    fdr_lo, fdr_hi = wilson_ci(fp, n_alert, ci_level)

    pct = int(round(ci_level * 100))
    return {
        "threshold": round(float(t), 6),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": round(rec, 4),
        f"Recall_CI{pct}_lo": rec_lo, f"Recall_CI{pct}_hi": rec_hi,
        "FPR": round(fpr, 4),
        f"FPR_CI{pct}_lo": fpr_lo, f"FPR_CI{pct}_hi": fpr_hi,
        "Precision": round(prec, 4),
        "경보중오경보비율(FDR)": round(fdr, 4) if np.isfinite(fdr) else np.nan,
        f"FDR_CI{pct}_lo": fdr_lo, f"FDR_CI{pct}_hi": fdr_hi,
        "경보수": n_alert,
    }


def select_threshold(y, s, target_recall):
    cands = np.unique(s)
    best = None
    for t in cands:
        tp, fp, fn, tn = confusion_at(y, s, t)
        rec = tp / (tp + fn)
        if rec + 1e-12 < target_recall:
            continue
        fpr = fp / (fp + tn)
        key = (fpr, -t)
        if best is None or key < best[0]:
            best = (key, t)
    if best is None:
        raise RuntimeError("목표 재현율을 만족하는 임계값이 없습니다.")
    return float(best[1])


# ------------------------------------------------------------------
# 3. 이동 블록 부트스트랩 (Moving Block Bootstrap, L=6, B=500, 95% CI)
# ------------------------------------------------------------------
def mbb_sample_indices(n, block_size, rng):
    """시간순 정렬된 시계열에서 길이 L인 겹침 블록을 복원추출하여 크기 n 복원."""
    if n <= block_size:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]


def fit_mbb_predict(X_tr, y_tr, X_ev, n_boot=MBB_N_BOOT, block_size=MBB_BLOCK_SIZE,
                    ci_level=CI_LEVEL, seed=SEED):
    """
    원본 학습셋의 scale_pos_weight를 고정(fixed_spw)한 채 이동 블록 부트스트랩 B회 수행.
    반환: (score_ci_low, score_ci_high, boot_std)
    """
    X_tr_arr = X_tr.to_numpy() if isinstance(X_tr, pd.DataFrame) else np.asarray(X_tr)
    X_ev_arr = X_ev.to_numpy() if isinstance(X_ev, pd.DataFrame) else np.asarray(X_ev)
    y_tr_arr = np.asarray(y_tr)
    n_tr = len(y_tr_arr)

    # [핵심 안전장치] 원본 학습셋의 클래스 가중치를 고정하여 부트스트랩 간 로그오즈 왜곡 방지
    fixed_spw = float((y_tr_arr == 0).sum() / max((y_tr_arr == 1).sum(), 1))
    rng = np.random.default_rng(seed)

    boot_scores = np.empty((n_boot, len(X_ev_arr)), dtype=float)
    b = 0
    attempts = 0
    while b < n_boot:
        attempts += 1
        idx = mbb_sample_indices(n_tr, block_size, rng)
        yb = y_tr_arr[idx]
        if len(np.unique(yb)) < 2:
            if attempts > n_boot * 5:
                raise RuntimeError("부트스트랩 표본에서 단일 클래스만 반복 추출됨")
            continue
        mb = build_model(yb, seed=SEED + b + 1, fixed_spw=fixed_spw)
        mb.fit(X_tr_arr[idx], yb)
        boot_scores[b] = mb.predict_proba(X_ev_arr)[:, 1]
        b += 1

    alpha = 1.0 - ci_level
    ci_lo = np.percentile(boot_scores, 100.0 * (alpha / 2.0), axis=0)
    ci_hi = np.percentile(boot_scores, 100.0 * (1.0 - alpha / 2.0), axis=0)
    ci_sd = np.std(boot_scores, axis=0, ddof=1)
    return ci_lo, ci_hi, ci_sd


def rolling_oof_mbb(dev, features, oof_base, thr, n_boot=MBB_N_BOOT,
                    block_size=MBB_BLOCK_SIZE, ci_level=CI_LEVEL):
    """2020~2023 확장형 시간순 검증 각 연도에 대해 이동 블록 부트스트랩 95% 원점수 구간 산출."""
    parts = []
    print(f"[MBB 시작] 2020~2023 검증 구간 이동 블록 부트스트랩 (L={block_size}, B={n_boot}, {int(ci_level*100)}% CI)")
    for yr_idx, yr in enumerate(VALIDATION_YEARS):
        ev = dev["year"] == yr
        if ev.sum() == 0:
            continue
        tr = dev["target_date"] < dev.loc[ev, "cutoff_date"].min()
        print(f"  -> {yr}년 검증({int(ev.sum())}건) | 학습 {int(tr.sum())}건으로 {n_boot}회 학습 중...")
        lo, hi, sd = fit_mbb_predict(
            dev.loc[tr, features], dev.loc[tr, TARGET].to_numpy(), dev.loc[ev, features],
            n_boot=n_boot, block_size=block_size, ci_level=ci_level, seed=SEED + yr_idx * 1000
        )
        part = oof_base[oof_base["year"] == yr].copy().reset_index(drop=True)
        part["score_ci_lo"] = np.round(lo, 6)
        part["score_ci_hi"] = np.round(hi, 6)
        part["score_boot_sd"] = np.round(sd, 6)
        part["ci_width"] = np.round(hi - lo, 6)
        part["alert_point"] = (part["score"] >= thr).astype(int)

        # 단일 임계값(thr) 대비 95% 원점수 불확실성 구간 위치에 따른 3구역 분류
        zones = []
        for l_val, u_val in zip(lo, hi):
            if u_val < thr:
                zones.append("1_안전(U<thr)")
            elif l_val > thr:
                zones.append("3_확정위험(L>thr)")
            else:
                zones.append("2_경계_채수권장(thr∈CI)")
        part["ci_zone"] = zones
        parts.append(part)

    oof_mbb = pd.concat(parts, ignore_index=True)

    # 구역별 실제 위험/정상 건수 및 윌슨 스코어 95% 오경보율(FDR) 요약
    zone_rows = []
    pct = int(round(ci_level * 100))
    for z_name in ["1_안전(U<thr)", "2_경계_채수권장(thr∈CI)", "3_확정위험(L>thr)", "합계_경계+확정(U>=thr)"]:
        if z_name.startswith("합계"):
            sub = oof_mbb[oof_mbb["score_ci_hi"] >= thr]
        else:
            sub = oof_mbb[oof_mbb["ci_zone"] == z_name]
        n_tot = len(sub)
        n_pos = int((sub[TARGET] == 1).sum())
        n_neg = int((sub[TARGET] == 0).sum())
        fdr = n_neg / n_tot if n_tot > 0 else np.nan
        ppv = n_pos / n_tot if n_tot > 0 else np.nan
        fdr_lo, fdr_hi = wilson_ci(n_neg, n_tot, ci_level)
        zone_rows.append({
            "CI_구역": z_name,
            "총건수": n_tot,
            "실제위험(TP/FN)": n_pos,
            "실제정상(FP/TN)": n_neg,
            "위험적중률(PPV)": round(ppv, 4) if np.isfinite(ppv) else np.nan,
            "실제오경보율(FDR)": round(fdr, 4) if np.isfinite(fdr) else np.nan,
            f"FDR_Wilson_CI{pct}_lo": fdr_lo,
            f"FDR_Wilson_CI{pct}_hi": fdr_hi,
            "평균_CI폭": round(float(sub["ci_width"].mean()), 4) if n_tot > 0 else np.nan,
        })
    zone_tbl = pd.DataFrame(zone_rows)
    return oof_mbb, zone_tbl


# ------------------------------------------------------------------
# 4. 그래프 공통 설정 · 성능 및 신뢰구간 시각화
# ------------------------------------------------------------------
def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


C_FINAL, C_B6, C_GREY = "#2b6cb0", "#a0aec0", "#718096"


def plot_mbb_ci_zones(outdir, oof_mbb, thr, block_size=MBB_BLOCK_SIZE, n_boot=MBB_N_BOOT, ci_level=CI_LEVEL):
    """2020~2023 검증 예측의 원점수 및 이동 블록 부트스트랩 95% 구간 시각화."""
    plt = setup_plot()
    pct = int(round(ci_level * 100))
    df_p = oof_mbb.sort_values("score").reset_index(drop=True)
    x = np.arange(len(df_p))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(
        x, df_p["score_ci_lo"], df_p["score_ci_hi"], color="#90cdf4", alpha=0.45,
        label=f"이동 블록 부트스트랩 {pct}% 원점수 구간 (L={block_size}, B={n_boot})"
    )
    pos = df_p[TARGET] == 1
    neg = ~pos
    ax.scatter(x[neg], df_p.loc[neg, "score"], s=14, color="#2b6cb0", alpha=0.85, label="실제 정상(0)")
    ax.scatter(x[pos], df_p.loc[pos, "score"], s=18, color="#c53030", marker="^", alpha=0.9, label="실제 위험(1)")
    ax.axhline(thr, color="#dd6b20", lw=1.6, ls="--", label=f"확정 임계값 ({thr:.3f})")

    ax.set_xlabel(f"2020~2023 검증 표본 (원점수 오름차순 정렬, n={len(df_p)})")
    ax.set_ylabel("모델 원점수 (보정 미적용 Score)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"개별 예측 원점수 및 이동 블록 부트스트랩 {pct}% 불확실성 구간 (2020~2023 검증)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "perf_6_mbb_ci_zones.png"), dpi=200)
    plt.close(fig)


def performance_plots(outdir, oof, b6_oof, thr, comp_tbl, st_tbl):
    """모든 그래프는 2020~2023 시간순 검증 예측만 사용한다(2024~2025 미사용)."""
    from sklearn.metrics import precision_recall_curve
    plt = setup_plot()
    O = lambda f: os.path.join(outdir, f)
    years = sorted(oof["year"].unique())

    # (P1) 연도별 AP
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(years)); w = 0.38
    ap_f = [yearly_ap(oof)[y] for y in years]
    ax.bar(x - w / 2, ap_f, w, color=C_FINAL, label="최종 7변수")
    cols = [f"AP_{y}" for y in years]
    if all(c in st_tbl.columns for c in cols):
        lo = np.minimum(st_tbl[cols].min().to_numpy(), ap_f)
        hi = np.maximum(st_tbl[cols].max().to_numpy(), ap_f)
        ax.errorbar(x - w / 2, ap_f, yerr=[np.array(ap_f) - lo, hi - np.array(ap_f)],
                    fmt="none", ecolor="black", capsize=3, lw=1, label="시드별 범위(42 + 0~9)")
    if b6_oof is not None:
        ap_b = [yearly_ap(b6_oof)[y] for y in years]
        ax.bar(x + w / 2, ap_b, w, color=C_B6, label="B6 원안 8변수")
    prev = [oof.loc[oof.year == y, TARGET].mean() for y in years]
    ax.scatter(x, prev, marker="_", s=600, color="#c53030", zorder=3, label="위험 비율(무작위 AP 기준선)")
    for i, v in enumerate(ap_f):
        ax.text(i - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years]); ax.set_ylim(0, 1.08)
    ax.set_ylabel("AP (PR-AUC)"); ax.set_title("연도별 순위 성능 (확장형 시간순 검증)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(O("perf_1_yearly_ap.png"), dpi=200); plt.close(fig)

    # (P2) PR Curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y, s = oof[TARGET].to_numpy(), oof["score"].to_numpy()
    pr, rc, _ = precision_recall_curve(y, s)
    ax.plot(rc, pr, color=C_FINAL, lw=2, label=f"최종 7변수 (AP {average_precision_score(y, s):.3f})")
    if b6_oof is not None:
        yb, sb = b6_oof[TARGET].to_numpy(), b6_oof["score"].to_numpy()
        prb, rcb, _ = precision_recall_curve(yb, sb)
        ax.plot(rcb, prb, color=C_B6, lw=2, ls="--", label=f"B6 원안 (AP {average_precision_score(yb, sb):.3f})")
    m = metrics_row(y, s, thr)
    ax.scatter([m["Recall"]], [m["Precision"]], color="#c53030", zorder=4, s=50,
               label=f"선택 임계값 {thr:.3f}")
    ax.axhline(y.mean(), color=C_GREY, lw=0.8, ls=":")
    ax.set_xlabel("재현율 (위험 탐지율)"); ax.set_ylabel("정밀도 (경보 적중률)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title("정밀도–재현율 곡선 (2020~2023 검증 통합)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(O("perf_2_pr_curve.png"), dpi=200); plt.close(fig)

    # (P3) 임계값에 따른 재현율·FPR 교환관계
    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = np.linspace(0.02, 0.98, 97)
    rec = [metrics_row(y, s, t)["Recall"] for t in grid]
    fpr = [metrics_row(y, s, t)["FPR"] for t in grid]
    ax.plot(grid, rec, color=C_FINAL, lw=2, label="재현율")
    ax.plot(grid, fpr, color="#dd6b20", lw=2, label="FPR (정상일 중 경보 비율)")
    ax.axhline(TARGET_RECALL, color=C_FINAL, ls=":", lw=1)
    ax.axvline(thr, color="#c53030", lw=1.2)
    ax.text(thr, 1.03, f" 선택 {thr:.3f}\n 재현율 {m['Recall']:.2f} · FPR {m['FPR']:.2f}",
            fontsize=8, va="bottom", color="#c53030")
    ax.set_xlabel("임계값 (원점수)"); ax.set_ylabel("비율"); ax.set_ylim(0, 1.15)
    ax.set_title(f"임계값 선택: 재현율 ≥ {TARGET_RECALL} 중 FPR 최소")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(O("perf_3_threshold_tradeoff.png"), dpi=200); plt.close(fig)

    # (P4) 선택 임계값에서 연도별 TP·FN·FP
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [metrics_row(g[TARGET].to_numpy(), g["score"].to_numpy(), thr) for _, g in oof.groupby("year")]
    tp = np.array([r["TP"] for r in rows]); fn = np.array([r["FN"] for r in rows]); fp = np.array([r["FP"] for r in rows])
    x = np.arange(len(years)); w = 0.38
    ax.bar(x - w / 2, tp, w, color=C_FINAL, label="탐지(TP)")
    ax.bar(x - w / 2, fn, w, bottom=tp, color="#90cdf4", label="미탐(FN)")
    ax.bar(x + w / 2, fp, w, color="#dd6b20", label="오경보(FP)")
    for i in range(len(years)):
        ax.text(i - w / 2, tp[i] + fn[i] + 0.4, f"{tp[i]}/{tp[i]+fn[i]}", ha="center", fontsize=8)
        ax.text(i + w / 2, fp[i] + 0.4, str(fp[i]), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years]); ax.set_ylabel("건수")
    ax.set_ylim(0, max((tp + fn).max(), fp.max()) * 1.15)
    ax.set_title(f"연도별 경보 결과 (임계값 {thr:.3f}, 왼쪽 막대 = 실제 위험일)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(O("perf_4_confusion_by_year.png"), dpi=200); plt.close(fig)

    # (P5) 변수 제거 실험: 평균 AP 변화
    ab = comp_tbl[comp_tbl["model"].str.startswith(("−", "+"))].copy()
    if len(ab):
        ab = ab.sort_values("ΔAP_mean(최종 대비)")
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(ab) + 1.2))
        colors = ["#c53030" if v < 0 else C_GREY for v in ab["ΔAP_mean(최종 대비)"]]
        ax.barh(ab["model"], ab["ΔAP_mean(최종 대비)"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        for i, v in enumerate(ab["ΔAP_mean(최종 대비)"]):
            ax.text(v, i, f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_xlabel("평균 AP 변화 (최종 모델 대비, 음수 = 그 변수가 성능에 기여)")
        ax.set_title("변수 제거 실험 (설명용, 모델 선택에 미사용)")
        lim = max(0.02, ab["ΔAP_mean(최종 대비)"].abs().max() * 1.4)
        ax.set_xlim(-lim, lim)
        fig.tight_layout(); fig.savefig(O("perf_5_ablation.png"), dpi=200); plt.close(fig)


# ------------------------------------------------------------------
# 5. SHAP (XGBoost 내장 TreeSHAP)
# ------------------------------------------------------------------
def shap_outputs(model, X, outdir, features, toc_col):
    plt = setup_plot()

    dm = xgb.DMatrix(X, missing=np.nan, feature_names=features)
    booster = model.get_booster()
    contrib = booster.predict(dm, pred_contribs=True)
    shap_vals = contrib[:, :-1]
    inter = booster.predict(dm, pred_interactions=True)

    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(outdir, "shap_importance.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(shap_vals, columns=features).to_csv(
        os.path.join(outdir, "shap_values_dev.csv"), index=False, encoding="utf-8-sig")

    p = len(features)
    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            rows.append({"pair": f"{features[i]} × {features[j]}",
                         "mean_abs_interaction": float(np.abs(inter[:, i, j] * 2).mean())})
    pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).to_csv(
        os.path.join(outdir, "shap_interactions.csv"), index=False, encoding="utf-8-sig")

    # (a) 중요도 막대
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#3b6ea5")
    ax.set_xlabel("평균 |SHAP| (로그오즈)")
    ax.set_title("변수별 평균 기여 크기 (2016~2023 학습 모델)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "shap_bar.png"), dpi=200); plt.close(fig)

    # (b) 분포(beeswarm 형태)
    order = imp["feature"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for k, f in enumerate(order[::-1]):
        j = features.index(f)
        v = X[f].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        c = np.where(np.isfinite(c), c, 0.5)
        ax.scatter(shap_vals[:, j], k + rng.uniform(-0.3, 0.3, len(v)), c=c, cmap="coolwarm", s=8, alpha=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP 값 (+: 위험 쪽, −: 정상 쪽)")
    ax.set_title("SHAP 분포 (색: 변수값 낮음 파랑 → 높음 빨강)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "shap_beeswarm.png"), dpi=200); plt.close(fig)

    # (c) 변수별 의존도
    for f in features:
        j = features.index(f)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.scatter(X[f], shap_vals[:, j], s=10, alpha=0.7, color="#3b6ea5")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f); ax.set_ylabel("SHAP")
        ax.set_title(f"{f} 의존도")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"shap_dep_{f}.png"), dpi=200); plt.close(fig)

    # (d) 가뭄 메커니즘: TOC 효과가 가뭄 조건에서 커지는가
    j_toc = features.index(toc_col)
    for dvar in DROUGHT_FEATURES:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sc = ax.scatter(X[toc_col], shap_vals[:, j_toc], c=X[dvar], cmap="viridis", s=12, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=dvar)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(toc_col); ax.set_ylabel(f"{toc_col}의 SHAP")
        ax.set_title(f"{toc_col} 기여 × {dvar}")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"shap_toc_x_{dvar}.png"), dpi=200); plt.close(fig)

    # 가뭄 조건별 TOC SHAP 요약 (SPI 결측 방어 처리 포함)
    valid_spi = X["SPI30d_gamma"].notna()
    dry = valid_spi & (X["SPI30d_gamma"] <= -1.0)
    wet = valid_spi & (X["SPI30d_gamma"] > -1.0)
    summ = pd.DataFrame({
        "조건": ["SPI30 ≤ −1 (가뭄)", "SPI30 > −1 (비가뭄)"],
        "표본수": [int(dry.sum()), int(wet.sum())],
        f"{toc_col}_평균SHAP": [
            float(np.nanmean(shap_vals[dry, j_toc])) if dry.any() else np.nan,
            float(np.nanmean(shap_vals[wet, j_toc])) if wet.any() else np.nan,
        ],
        f"{toc_col}_평균|SHAP|": [
            float(np.nanmean(np.abs(shap_vals[dry, j_toc]))) if dry.any() else np.nan,
            float(np.nanmean(np.abs(shap_vals[wet, j_toc]))) if wet.any() else np.nan,
        ],
    })
    summ.to_csv(os.path.join(outdir, "shap_toc_by_drought.csv"), index=False, encoding="utf-8-sig")
    return imp


# ------------------------------------------------------------------
# 6. 웹 시연 / 신규 입력용 예측 함수 (원점수 + MBB 95% 구간 + 윌슨 오경보율 안내)
# ------------------------------------------------------------------
def predict_new_with_ci(model_path, config_path, dev_df, X_new,
                        n_boot=200, block_size=MBB_BLOCK_SIZE, ci_level=CI_LEVEL):
    """
    신규 관측치 X_new에 대해:
      - 점추정 원점수(score) 및 이진 경보(alert)
      - 이동 블록 부트스트랩 95% 원점수 불확실성 구간(score_ci_lo, score_ci_hi) 및 구역 판정(ci_zone)
      - 과거 검증 기준 시스템 오경보율 95% 윌슨 구간(system_expected_FDR_note)
      - 변수별 SHAP 기여도 반환
    """
    cfg = json.load(open(config_path, encoding="utf-8"))
    feats = cfg["features"]
    thr = cfg["threshold"]

    m = XGBClassifier()
    m.load_model(model_path)
    X = X_new[feats]
    s = m.predict_proba(X)[:, 1]

    lo, hi, sd = fit_mbb_predict(
        dev_df[feats], dev_df[TARGET].to_numpy(), X,
        n_boot=n_boot, block_size=block_size, ci_level=ci_level, seed=SEED
    )

    zones = [
        "1_안전(U<thr)" if u < thr else ("3_확정위험(L>thr)" if l > thr else "2_경계_채수권장(thr∈CI)")
        for l, u in zip(lo, hi)
    ]

    contrib = m.get_booster().predict(
        xgb.DMatrix(X, missing=np.nan, feature_names=feats), pred_contribs=True
    )[:, :-1]
    out = pd.DataFrame(contrib, columns=[f"SHAP_{f}" for f in feats], index=X.index)

    fdr_info = cfg.get("system_wilson_ci", {})
    pct = int(round(fdr_info.get("level", ci_level) * 100))
    fdr_str = (
        f"{fdr_info.get('FDR', np.nan)*100:.1f}% "
        f"({pct}% CI: {fdr_info.get(f'FDR_CI{pct}_lo', np.nan)*100:.1f}~"
        f"{fdr_info.get(f'FDR_CI{pct}_hi', np.nan)*100:.1f}%)"
    )

    out.insert(0, "system_expected_FDR_note", fdr_str)
    out.insert(0, "ci_zone", zones)
    out.insert(0, "score_boot_sd", np.round(sd, 4))
    out.insert(0, "score_ci_hi", np.round(hi, 4))
    out.insert(0, "score_ci_lo", np.round(lo, 4))
    out.insert(0, "alert", (s >= thr).astype(int))
    out.insert(0, "score", np.round(s, 4))
    return out


# ------------------------------------------------------------------
# 7. 메인 파이프라인 (주피터 노트북 & CLI 공용)
# ------------------------------------------------------------------
def run_pipeline(input_path="공통데이터.xlsx", output_dir="final_out_ci",
                 n_boot=MBB_N_BOOT, block_size=MBB_BLOCK_SIZE,
                 ci_level=CI_LEVEL, refit_all=False):
    os.makedirs(output_dir, exist_ok=True)
    O = lambda f: os.path.join(output_dir, f)

    full, dev = load_data(input_path)
    toc_avail = check_columns(dev, TOC_CANDIDATES)

    # (1) 결측·상관 점검
    cols = BASE_FEATURES + toc_avail
    pd.DataFrame({"유효값": dev[cols].notna().sum(), "전체": len(dev)}).to_csv(
        O("01_missing.csv"), encoding="utf-8-sig")
    corr = dev[cols].corr(method="spearman")
    corr.round(3).to_csv(O("01_spearman.csv"), encoding="utf-8-sig")
    high = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(cols) for b in cols[i + 1:]
            if abs(corr.loc[a, b]) >= 0.85 and not (a in TOC_CANDIDATES and b in TOC_CANDIDATES)]
    print("[상관] |ρ| ≥ 0.85 쌍:", high if high else "없음")

    # (2) TOC 표현 선택
    toc_rows, toc_oof = [], {}
    for t in toc_avail:
        oof = rolling_oof(dev, BASE_FEATURES + [t])
        toc_oof[t] = oof
        r = summarize(t, oof); r["유효값"] = int(dev[t].notna().sum())
        toc_rows.append(r)
    toc_tbl = pd.DataFrame(toc_rows)
    toc_tbl.to_csv(O("02_toc_selection.csv"), index=False, encoding="utf-8-sig")

    best_ap = toc_tbl["AP_mean"].max()
    default = TOC_CANDIDATES[0]
    def_ap = float(toc_tbl.loc[toc_tbl["model"] == default, "AP_mean"].iloc[0])
    toc_col = default if def_ap >= best_ap - TOC_RULE_TOLERANCE else         toc_tbl.sort_values("AP_mean", ascending=False)["model"].iloc[0]
    print(toc_tbl.to_string(index=False))
    print(f"[TOC 선택] {toc_col}  (규칙: 기본값이 최고 평균 AP와 {TOC_RULE_TOLERANCE} 이내면 기본값 유지)")

    FEATURES = BASE_FEATURES + [toc_col]
    oof = toc_oof[toc_col]
    oof.to_csv(O("03_oof_predictions.csv"), index=False, encoding="utf-8-sig")

    # (3) B6 비교 및 제거 실험
    b6 = ["TOC_last", "AirTemp_mean_7d", "Flow_mean_3d", "Rain_sum_7d",
          "SPI30d_gamma", "SPI7d_gamma", "Dry_days_lt1mm", "TOC_age_actual"]
    comp = [summarize(f"최종 7변수 ({toc_col})", oof)]
    b6_oof = rolling_oof(dev, b6) if all(c in dev.columns for c in b6) else None
    if b6_oof is not None:
        comp.append(summarize("B6 원안 8변수", b6_oof))

    ablations = {f"− {f}": [x for x in FEATURES if x != f] for f in FEATURES}
    ablations["− 가뭄 2종(SPI30·건조일수)"] = [x for x in FEATURES if x not in DROUGHT_FEATURES]
    if "DOY_cos" in dev.columns:
        ablations["+ DOY_cos (참고)"] = FEATURES + ["DOY_cos"]
    for name, f in ablations.items():
        comp.append(summarize(name, rolling_oof(dev, f)))

    comp_tbl = pd.DataFrame(comp)
    comp_tbl["ΔAP_mean(최종 대비)"] = (comp_tbl["AP_mean"] - comp_tbl.loc[0, "AP_mean"]).round(4)
    comp_tbl.to_csv(O("04_comparison_ablation.csv"), index=False, encoding="utf-8-sig")
    print(comp_tbl.to_string(index=False))

    # (4) 시드 안정성
    st_tbl = pd.DataFrame([summarize(f"seed={s}", rolling_oof(dev, FEATURES, seed=s)) for s in STABILITY_SEEDS])
    st_tbl.to_csv(O("05_seed_stability.csv"), index=False, encoding="utf-8-sig")
    print(f"[시드 안정성] 평균 AP {st_tbl['AP_mean'].mean():.4f} ± {st_tbl['AP_mean'].std(ddof=1):.4f}")

    # (5) 임계값 선정 + [모듈 ①] 윌슨 스코어 신뢰구간 산출
    y, s = oof[TARGET].to_numpy(), oof["score"].to_numpy()
    thr = select_threshold(y, s, TARGET_RECALL)

    sens = []
    for r in SENSITIVITY_RECALLS:
        t = select_threshold(y, s, r)
        sens.append({"목표재현율": r, **metrics_row(y, s, t, ci_level=ci_level)})
    for k in COST_RATIOS:
        best = min(np.unique(s), key=lambda t: (k * confusion_at(y, s, t)[2] + confusion_at(y, s, t)[1], -t))
        sens.append({"목표재현율": f"손실 {k}×FN+FP 최소(참고)", **metrics_row(y, s, best, ci_level=ci_level)})
    pd.DataFrame(sens).to_csv(O("06_threshold_sensitivity.csv"), index=False, encoding="utf-8-sig")

    overall_m = metrics_row(y, s, thr, ci_level=ci_level)
    per_year = [{"연도": "2020~2023 통합", **overall_m}]
    for yr, g in oof.groupby("year"):
        per_year.append({"연도": int(yr), **metrics_row(g[TARGET].to_numpy(), g["score"].to_numpy(), thr, ci_level=ci_level)})
    py_tbl = pd.DataFrame(per_year)
    py_tbl.to_csv(O("06_threshold_by_year.csv"), index=False, encoding="utf-8-sig")

    print(f"\n=== [모듈 ①] 확정 임계값({thr:.4f}) 및 윌슨 스코어 {int(ci_level*100)}% 신뢰구간 ===")
    print(py_tbl.to_string(index=False))

    # (6) [모듈 ②] 2020~2023 검증 구간 이동 블록 부트스트랩 원점수 구간 산출
    oof_mbb, zone_tbl = rolling_oof_mbb(
        dev, FEATURES, oof, thr, n_boot=n_boot, block_size=block_size, ci_level=ci_level
    )
    oof_mbb.to_csv(O("07_oof_mbb_ci_predictions.csv"), index=False, encoding="utf-8-sig")
    zone_tbl.to_csv(O("07_mbb_zone_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"\n=== [모듈 ②] 이동 블록 부트스트랩(L={block_size}, B={n_boot}, {int(ci_level*100)}% CI) 구역별 요약 ===")
    print(zone_tbl.to_string(index=False))

    # (7) 성능 및 신뢰구간 그래프 저장
    performance_plots(output_dir, oof, b6_oof, thr, comp_tbl, st_tbl)
    plot_mbb_ci_zones(output_dir, oof_mbb, thr, block_size=block_size, n_boot=n_boot, ci_level=ci_level)

    # (8) 최종 모델 학습(2016~2023) · SHAP · 설정 저장
    ydev = dev[TARGET].to_numpy()
    final = build_model(ydev).fit(dev[FEATURES], ydev)
    final.save_model(O("final_model_2016_2023.json"))
    shap_outputs(final, dev[FEATURES], output_dir, FEATURES, toc_col)

    pct = int(round(ci_level * 100))
    cfg = {
        "features": FEATURES,
        "toc_column": toc_col,
        "threshold": thr,
        "threshold_rule": f"2020~2023 시간순 검증 예측에서 Recall ≥ {TARGET_RECALL} 중 FPR 최소",
        "xgb_params": XGB_PARAMS,
        "scale_pos_weight": "원본 학습셋 정상 수 / 위험 수 (부트스트랩 내부 고정)",
        "train_period": "2016~2023",
        "sealed_period": f"{SEALED_FROM_YEAR}~ (미사용)",
        "score_note": "원점수는 보정하지 않았으므로 발생 확률로 표기하지 않는다",
        "system_wilson_ci": {
            "level": ci_level,
            "Recall": overall_m["Recall"],
            f"Recall_CI{pct}_lo": overall_m[f"Recall_CI{pct}_lo"],
            f"Recall_CI{pct}_hi": overall_m[f"Recall_CI{pct}_hi"],
            "FPR": overall_m["FPR"],
            f"FPR_CI{pct}_lo": overall_m[f"FPR_CI{pct}_lo"],
            f"FPR_CI{pct}_hi": overall_m[f"FPR_CI{pct}_hi"],
            "FDR": overall_m["경보중오경보비율(FDR)"],
            f"FDR_CI{pct}_lo": overall_m[f"FDR_CI{pct}_lo"],
            f"FDR_CI{pct}_hi": overall_m[f"FDR_CI{pct}_hi"],
        },
        "mbb_ci_config": {
            "method": "Moving Block Bootstrap (percentile CI, fixed scale_pos_weight)",
            "block_size_L": block_size,
            "n_boot_B": n_boot,
            "ci_level": ci_level,
            "interpretation_note": "보정되지 않은 모델 원점수(Score)의 표집 불확실성 구간이며 발생확률 구간과 구분한다."
        },
    }

    if refit_all:
        yall = full[TARGET].to_numpy()
        op = build_model(yall).fit(full[FEATURES], yall)
        op.save_model(O("operational_model_2016_2025.json"))
        cfg["operational_model"] = "2016~2025 전체 재학습 (설정·임계값 고정, 성능 미계산)"
        print("[운영 모델] 2016~2025 전체 재학습 완료 — 성능 지표는 계산하지 않음")

    json.dump(cfg, open(O("config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[완료] 모든 결과 CSV, 모델, 그래프가 '{output_dir}' 폴더에 저장되었습니다.")
    return py_tbl, zone_tbl, oof_mbb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="공통데이터.xlsx")
    ap.add_argument("--output", default="final_out_ci")
    ap.add_argument("--n-boot", type=int, default=MBB_N_BOOT)
    ap.add_argument("--block-size", type=int, default=MBB_BLOCK_SIZE)
    ap.add_argument("--ci-level", type=float, default=CI_LEVEL)
    ap.add_argument("--refit-all", action="store_true")
    args, _ = ap.parse_known_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        n_boot=args.n_boot,
        block_size=args.block_size,
        ci_level=args.ci_level,
        refit_all=args.refit_all,
    )
