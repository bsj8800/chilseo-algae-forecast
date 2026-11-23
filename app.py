# -*- coding: utf-8 -*-
"""
캡스톤디자인 2조 · 낙동강 칠서 녹조(유해남조류) 3일 예보 — 연구 시연용 웹 앱 (Streamlit)

사용 흐름
  1) 보고 싶은 날짜를 고른다.
  2) 공통 엑셀의 '일별 원자료'·'강수와 SPI' 시트에서 그 날짜의 입력값을 자동으로 만든다.
     (예측 기준일 = 대상일 3일 전 10:30, 입력 = 그 전날까지의 자료)
  3) 날짜에 맞는 모델로 예측하고 결과·신뢰도·근거·지도·추이를 보여준다.

기간 구분 (교수님 피드백 반영)
  - 2020~2023 : 검증 — 그 해 이전 자료로만 학습한 모델로 예측
  - 2024~2025 : 참고 — 개발 중 열람한 기간이라 성능 평가에 쓰지 않음
  - 2026.1~9  : 독립 테스트 — 고정한 최종 모델(2016~2023 학습)로 처음 본 자료를 예측

원칙
  - 세포 수가 아니라 '1,000 cells/mL 이상 여부'를 예측한다.
  - 점수는 보정하지 않은 '위험 신호'이며 확률(%)로 표시하지 않는다.
  - 신뢰도: 확정 알고리즘(final_model_with_CI.py)의 이동 블록 부트스트랩 95% 구간이
    경보 기준을 걸치지 않으면 '신뢰도 높음', 걸치면 '신뢰도 낮음(경계)'. 경보 판정 자체는 이진 유지.
  - 2024~2025 실측은 기본으로 숨기고, 켜더라도 참고용임을 밝힌다.
  - 공식 조류경보가 아니라 연구 시연임을 화면에 밝힌다.

실행: streamlit run app.py  (또는 VS Code ▶)
같은 폴더: app.py, final_original.py, test.py, common_data_13cols.xlsx, (test.py 결과) frozen_model.json · frozen_config.json · test_predictions.csv · test_metrics.csv
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb
from xgboost import XGBClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

if __name__ == "__main__":   # VS Code ▶ 로 실행해도 streamlit 으로 다시 띄우고 브라우저를 직접 연다
    try:
        from streamlit import runtime
        if not runtime.exists():
            import socket
            import threading
            import webbrowser

            def free_port(start=8501):
                for p in range(start, start + 20):
                    with socket.socket() as sk:
                        if sk.connect_ex(("127.0.0.1", p)) != 0:
                            return p
                return start

            port = free_port()
            url = f"http://localhost:{port}"
            print(f"\n  앱 주소: {url}  (브라우저가 안 열리면 이 주소를 직접 입력하세요)\n", flush=True)
            threading.Timer(4, lambda: webbrowser.open(url)).start()
            sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
                                      "--server.port", str(port), "--server.headless", "true",
                                      "--browser.gatherUsageStats", "false",
                                      "--theme.base", "light", "--theme.primaryColor", "#3182F6",
                                      "--theme.backgroundColor", "#F4F5F7", "--theme.textColor", "#191F28"]))
    except ImportError:
        pass

# 확정 원본 알고리즘(final_original.py)의 함수를 그대로 불러 쓴다.
import importlib
fm = None
for _name in ["final_original", "final_model_with_CI", "final_model"]:
    try:
        fm = importlib.import_module(_name)
        break
    except ImportError:
        continue
if fm is None:
    raise SystemExit("final_original.py 가 같은 폴더에 없어요.")
HAS_MBB = hasattr(fm, "fit_mbb_predict")

# test.py 가 만든 고정 모델·설정 (같은 폴더)
MODEL_FILE, CONFIG_FILE = "frozen_model.json", "frozen_config.json"
TEST_PRED, TEST_METRICS = "test_predictions.csv", "test_metrics.csv"
CACHE_DIR = "web_cache"                                   # 웹 전용 계산 결과(신뢰도 범위)
GRAPH_DIR = next((d for d in ["final_out_ci", "final_out"] if os.path.isdir(d)), None)  # 원본 실행 그래프(있으면)
REF_FROM, TEST_FROM = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")
TEST_SPLIT = "test_2026"

# 색 (밝은 배경 · 파랑 강조 · 위험은 빨강)
BLUE, RED, INK, SUB, LINE, BG = "#3182F6", "#F04452", "#191F28", "#8B95A1", "#E5E8EB", "#F2F4F6"

# 지도 지점. 창원은 기상청 공개 좌표(35°10′N, 128°34′E), 나머지는 대략적 위치 → 실제 측정소 좌표로 바꿔 쓰세요.
STATIONS = {
    # 칠서: 조류경보제 운영 매뉴얼(2025.8) 공식 좌표 128°28'55.9"E 35°22'55.4"N (칠서취수구 50m)
    "칠서": dict(lat=35.38206, lon=128.48219, tag="예측 지점", what="유해남조류", icon="🎯", exact=True, pos="br"),
    # 남지: 수질측정망 남지 지점 = 남지철교(창녕 남지읍 남지리 ~ 함안 칠서면 계내리), 칠서와 약 2km.
    #       공개 위경도를 찾지 못해 추정값 → 지도를 클릭해 나온 좌표로 바꾸세요.
    "남지": dict(lat=35.3855, lon=128.4950, tag="수질", what="수온 · TOC", icon="💧", exact=False, pos="top"),
    # 청암리: 낙동강홍수통제소 자동유량관측소 '창녕군(청암리)', 주소 창원시 의창구 북면 외산리 761-15.
    #         공개 위경도를 찾지 못해 추정값 → 지도를 클릭해 나온 좌표로 바꾸세요.
    "청암리": dict(lat=35.4000, lon=128.6000, tag="유량", what="3일 평균 유량", icon="🌊", exact=False, pos="right"),
    # 창원: 기상청 ASOS 155 창원 35.17019N 128.57281E
    "창원": dict(lat=35.17019, lon=128.57281, tag="기상", what="일사량 · 강수 · 가뭄지수", icon="🌤️", exact=True, pos="bottom"),
}

INFO = {  # 변수: (이름, 단위, 지점, 아이콘)
    "Temp_last": ("수온", "°C", "남지", "🌡️"),
    "Solar_mean_7d": ("일사량", "MJ/m²", "창원", "☀️"),
    "DOY_sin": ("계절", "", "", "📅"),
    "Flow_mean_3d": ("유량", "m³/s", "청암리", "🌊"),
    "SPI30d_gamma": ("30일 강수 지수", "", "창원", "🌧️"),
    "Dry_days_lt1mm": ("비 안 온 날", "일", "창원", "🏜️"),
    "TOC_last": ("유기물(TOC)", "mg/L", "남지", "🧪"),
    "TOC_state_mean_7d": ("유기물(TOC)", "mg/L", "남지", "🧪"),
}

st.set_page_config(page_title="칠서 녹조 3일 예보", page_icon="🌊", layout="wide")
plt = fm.setup_plot()

# ------------------------------------------------------------------
# 스타일
# ------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, [class*="css"], .stMarkdown, .stText, button, input, label {{
  font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif !important; color: {INK};
  font-feature-settings: "tnum";
}}
.stApp {{ background: #F4F5F7; }}
section[data-testid="stSidebar"] {{ background: #fff; }}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
.sub {{ color: {SUB}; font-size: 14px; font-weight: 500; }}
.h1 {{ font-size: 30px; font-weight: 800; letter-spacing: -0.8px; margin: 4px 0 18px; }}
.h2 {{ font-size: 19px; font-weight: 700; letter-spacing: -0.4px; margin: 30px 0 12px; }}
.card {{ background: #fff; border-radius: 24px; padding: 22px 24px; margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04); }}
.label {{ font-size: 13px; font-weight: 600; color: {SUB}; letter-spacing: .2px; }}
.ctitle {{ font-size: 17px; font-weight: 700; letter-spacing: -0.3px; margin: 2px 0 6px; }}
.chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 5px 10px; border-radius: 999px;
        font-size: 12.5px; font-weight: 700; }}
.hero .big {{ font-size: 27px; font-weight: 800; letter-spacing: -0.7px; line-height: 1.3; margin: 14px 0 6px; }}
.hero .desc {{ font-size: 14px; color: #4E5968; }}
.gauge {{ position: relative; height: 8px; background: {BG}; border-radius: 999px; margin: 20px 0 8px; }}
.gauge .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }}
.gauge .band {{ position: absolute; top: -6px; bottom: -6px; border-radius: 6px; opacity: .16; }}
.cline {{ font-size: 13px; color: #4E5968; margin-top: 10px; line-height: 1.5; }}
.obsline {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 12px;
           padding: 10px 14px; border-radius: 14px; background: #F7F8FA; font-size: 13.5px; color: #4E5968; }}
.obsline b {{ color: {INK}; }}
.gauge .mark {{ position: absolute; top: -5px; width: 3px; height: 18px; border-radius: 2px; background: {INK}; }}
.gscale {{ display: flex; justify-content: space-between; font-size: 12.5px; color: {SUB}; }}
.foot {{ display: flex; gap: 10px; align-items: flex-start; background: #F7F8FA; border-radius: 16px;
        padding: 12px 14px; margin-top: 18px; font-size: 13px; color: #4E5968; line-height: 1.5; }}
.foot b {{ color: {INK}; }}
.row {{ display: flex; align-items: center; gap: 14px; padding: 13px 0; }}
.row + .row {{ border-top: 1px solid #F2F4F6; }}
.ico {{ width: 42px; height: 42px; border-radius: 14px; display: flex; align-items: center;
        justify-content: center; font-size: 20px; flex: none; background: {BG}; }}
.row .t {{ font-size: 15.5px; font-weight: 600; }}
.row .s {{ font-size: 13px; color: {SUB}; margin-top: 2px; }}
.row .r {{ margin-left: auto; text-align: right; }}
.row .v {{ font-size: 16px; font-weight: 700; }}
.grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.gridn {{ display: grid; gap: 12px; }}
.spread {{ flex: 1; display: flex; flex-direction: column; justify-content: space-evenly; }}
.spread .row + .row {{ border-top: 1px solid #F2F4F6; }}
.topbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 26px; }}
.brand {{ display: flex; align-items: center; gap: 10px; font-size: 17px; font-weight: 800; letter-spacing: -0.3px; }}
.brand .logo {{ width: 34px; height: 34px; border-radius: 11px; background: {BLUE}; display: inline-flex;
               align-items: center; justify-content: center; font-size: 18px; }}
.topmeta {{ font-size: 12.5px; color: {SUB}; background: #fff; padding: 7px 12px; border-radius: 999px;
           box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.st-key-datebar {{ background: #fff; border-radius: 22px; padding: 14px 20px 18px;
                  box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04); margin-bottom: 8px; }}
.st-key-datebar label p {{ font-size: 13px !important; color: {SUB} !important; font-weight: 600 !important; }}
.footer {{ margin-top: 40px; padding: 22px 0 10px; border-top: 1px solid #E5E8EB; font-size: 12.5px;
          color: {SUB}; line-height: 1.8; }}
.footer b {{ color: #4E5968; }}
.eqh {{ height: 410px; box-sizing: border-box; display: flex; flex-direction: column; }}
.eqimg {{ flex: 1; min-height: 0; display: flex; align-items: center; margin-top: 6px; }}
.eqimg img {{ width: 100%; max-height: 100%; object-fit: contain; }}
.mini {{ background: #fff; border-radius: 20px; padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04); }}
.mini .k {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #4E5968; font-weight: 600; }}
.mini .k i {{ font-style: normal; width: 28px; height: 28px; border-radius: 10px; background: {BG};
             display: inline-flex; align-items: center; justify-content: center; font-size: 15px; }}
.mini .v {{ font-size: 22px; font-weight: 800; letter-spacing: -0.4px; margin-top: 10px; }}
.mini .n {{ font-size: 12px; color: {SUB}; margin-top: 3px; }}
.note {{ font-size: 12.5px; color: {SUB}; line-height: 1.6; margin-top: 10px; }}
iframe {{ border-radius: 24px; box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.05); }}
div[data-baseweb="tab-list"] {{ gap: 8px; }}
button[data-baseweb="tab"] {{ font-size: 15px; font-weight: 700; }}
div[data-baseweb="select"] > div {{ border-radius: 12px; background: #fff; }}
.stButton button {{ border-radius: 12px; font-weight: 600; background: #fff; }}
</style>
""", unsafe_allow_html=True)


def html(s):
    st.markdown("\n".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


# ------------------------------------------------------------------
# 자료 · 모델
# ------------------------------------------------------------------
def find_excel():
    """최종 공통데이터(common_data_13cols.xlsx)를 우선, 없으면 이름에 2026이 든 파일, 그다음 최근 파일."""
    files = [f for f in glob.glob("*.xlsx") if not os.path.basename(f).startswith("~$")
             and f != "uploaded_common.xlsx"]
    files.sort(key=lambda f: (os.path.basename(f) == "common_data_13cols.xlsx", "2026" in os.path.basename(f),
                              os.path.getmtime(f)), reverse=True)
    return files[0] if files else None


@st.cache_resource
def load_model(_mtime):
    cfg = json.load(open(CONFIG_FILE, encoding="utf-8"))
    m = XGBClassifier()
    m.load_model(MODEL_FILE)
    return cfg, m


@st.cache_data
def load_raw(xlsx):
    daily = pd.read_excel(xlsx, sheet_name="일별 원자료", parse_dates=["Date"]).set_index("Date").sort_index()
    spi = pd.read_excel(xlsx, sheet_name="강수와 SPI", parse_dates=["Date"]).set_index("Date").sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    full, dev = fm.load_data(xlsx)
    return daily, spi, full, dev


def fold_train_mask(d, yr):
    """yr년 검증용 학습 자료: 그 해 첫 입력 마감일보다 앞선 목표일 (알고리즘 코드의 시간순 검증과 동일)."""
    ev = d["year"] == yr
    return (d["target_date"] < d.loc[ev, "cutoff_date"].min()) if ev.any() else None


@st.cache_resource
def fold_models(xlsx, feats):
    """2020~2023: 그 해 이전 자료로만 학습한 모델."""
    _, d = fm.load_data(xlsx)
    out = {}
    for yr in fm.VALIDATION_YEARS:
        tr = fold_train_mask(d, yr)
        if tr is None:
            continue
        y = d.loc[tr, fm.TARGET].to_numpy()
        out[yr] = fm.build_model(y).fit(d.loc[tr, feats], y)
    return out


def validation_oof():
    """2020~2023 시간순 검증 예측 (원본 rolling_oof 와 같은 학습자료·시드)."""
    parts = []
    for yr, mdl in FOLDS.items():
        g = dev[dev["year"] == yr]
        if len(g):
            parts.append(g[["target_date", fm.TARGET]].assign(score=mdl.predict_proba(g[FEATS])[:, 1]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["target_date", fm.TARGET, "score"])


def pick_model(target):
    """날짜에 맞는 모델과 기간 구분을 돌려준다."""
    y = target.year
    if y in FOLDS:
        return FOLDS[y], "oof"          # 그 해 이전 자료로만 학습 → 검증
    if y < min(FOLDS, default=2020):
        return model, "insample"        # 최종 모델이 이미 학습한 날짜 (화면에서 선택 불가)
    if target < TEST_FROM:
        return model, "reference"       # 2024~2025: 개발 중 열람한 참고 기간
    return model, "test"                # 2026~: 고정 모델로 처음 보는 독립 테스트


PERIOD_NOTE = {
    "oof": ("✅", "검증 기간 · {y}년", "{y}년 이전 자료로만 학습한 모델이 예측했어요. 실제 성능에 가까운 결과예요."),
    "insample": ("⚠️", "학습에 쓰인 날짜예요", "모델이 이미 본 자료라 성능 판단에는 쓰지 마세요."),
    "reference": ("ℹ️", "참고 기간 · 2024~2025", "개발 중 이미 열람한 기간이라 성능 평가에는 쓰지 않아요. 결과는 참고만 해 주세요."),
    "test": ("🧪", "독립 테스트 · 2026", "모델을 고정한 뒤 처음 본 자료예요. 실제 성능을 가장 정직하게 보여주는 기간이에요."),
}
PERIOD_BADGE = {"oof": ("검증", "#E8F7EE", "#1A7F45"), "reference": ("참고", "#F2F4F6", "#6B7684"),
                "test": ("독립 테스트", "#F3EEFF", "#6B3FD4"), "insample": ("학습", "#FFF4E5", "#B26A00")}


# ------------------------------------------------------------------
# 신뢰도: 이동 블록 부트스트랩 95% 원점수 구간 (확정 알고리즘 fit_mbb_predict 그대로 사용)
#   - 2020~2023: 연도별 검증 학습자료, 시드 SEED + 연도순번×1000 → 알고리즘 결과표(07_*)와 같은 값
#   - 2024~    : 2016~2023 전체 학습자료, 시드 SEED
#   - 모든 날짜를 한 번에 계산해 결과 폴더에 저장 (처음 한 번만 수 분)
# ------------------------------------------------------------------
def ci_settings():
    return dict(B=int(cfg.get("n_boot", getattr(fm, "MBB_N_BOOT", 500))),
                L=int(cfg.get("block_size", getattr(fm, "MBB_BLOCK_SIZE", 6))),
                level=float(getattr(fm, "CI_LEVEL", 0.95)))


def load_or_build_ci():
    if not HAS_MBB:
        return None
    cs = ci_settings()
    key = dict(cs, features=FEATS, first=str(first_day.date()), last=str(last_day.date()),
               excel=os.path.basename(xlsx), excel_mtime=int(os.path.getmtime(xlsx)),
               model_sha256=cfg.get("model_sha256"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    p_csv, p_key = os.path.join(CACHE_DIR, "web_ci_cache.csv"), os.path.join(CACHE_DIR, "web_ci_cache.json")
    if os.path.exists(p_csv) and os.path.exists(p_key):
        try:
            if json.load(open(p_key, encoding="utf-8")) == key:
                c = pd.read_csv(p_csv, parse_dates=["date"]).set_index("date")
                return c
        except Exception:
            pass
    jobs = []
    for k, yr in enumerate(fm.VALIDATION_YEARS):
        tr = fold_train_mask(dev, yr)
        if tr is None:
            continue
        days = pd.date_range(max(first_day, pd.Timestamp(f"{yr}-01-01")), min(last_day, pd.Timestamp(f"{yr}-12-31")))
        jobs.append((f"{yr}년 검증", tr, fm.SEED + k * 1000, days))
    # 2024년 이후: test.py 와 같은 설정(2016~2023 전체, seed=SEED) → test_predictions.csv 의 구간과 같은 값
    jobs.append(("2024년 이후", pd.Series(True, index=dev.index), fm.SEED,
                 pd.date_range(max(first_day, REF_FROM), last_day)))
    html('<div class="h1">신뢰도 계산을 준비하고 있어요</div>'
         f'<div class="sub">이동 블록 부트스트랩(L={cs["L"]}, B={cs["B"]})으로 날짜마다 95% 구간을 계산해요. '
         '처음 한 번만 걸리고(수 분), 결과는 저장해 두었다가 다시 써요.</div>')
    bar = st.progress(0.0)
    parts = []
    for i, (name, tr, seed, days) in enumerate(jobs):
        bar.progress(i / len(jobs), text=f"{name} 계산 중… ({i + 1}/{len(jobs)})")
        X = pd.DataFrame([build_inputs(daily, spi, d)[0] for d in days])[FEATS].astype(float)
        lo, hi, _ = fm.fit_mbb_predict(dev.loc[tr, FEATS], dev.loc[tr, fm.TARGET].to_numpy(), X,
                                       n_boot=cs["B"], block_size=cs["L"], ci_level=cs["level"], seed=seed)
        parts.append(pd.DataFrame({"date": days, "lo": lo, "hi": hi}))
    bar.progress(1.0, text="완료")
    c = pd.concat(parts).set_index("date")
    c.to_csv(p_csv, encoding="utf-8-sig")
    json.dump(key, open(p_key, "w", encoding="utf-8"), ensure_ascii=False)
    st.rerun()


def ci_of(d):
    if CI is None or d not in CI.index:
        return None
    r = CI.loc[d]
    return float(r["lo"]), float(r["hi"])


def confidence(risk, band):
    """경보 판정(이진)은 그대로 두고, 95% 구간이 경보 기준을 걸치지 않으면 '신뢰도 높음'."""
    if band is None:
        return None
    lo, hi = band
    return (lo > THR) if risk else (hi < THR)


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def predict_dates(dates):
    """여러 날짜를 각 날짜에 맞는 모델로 예측."""
    rows = [build_inputs(daily, spi, d)[0] for d in dates]
    out = np.zeros(len(dates))
    kinds = [pick_model(d) for d in dates]
    for m in {id(k[0]): k[0] for k in kinds}.values():
        idx = [i for i, k in enumerate(kinds) if k[0] is m]
        out[idx] = predict(m, FEATS, [rows[i] for i in idx])[0]
    return out


def last_obs(series, cutoff, max_age=7):
    s = series.loc[cutoff - pd.Timedelta(days=max_age):cutoff].dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (np.nan, None)


def build_inputs(daily, spi, target):
    """목표일 T의 입력을 원자료에서 만든다(공통 엑셀 모델 입력과 553건 대조 검증한 규칙)."""
    cutoff, issue = target - pd.Timedelta(days=4), target - pd.Timedelta(days=3)
    x, meta = {}, {}
    x["Temp_last"], meta["Temp_last"] = last_obs(daily["Temp"], cutoff)
    x["TOC_last"], meta["TOC_last"] = last_obs(daily["TOC"], cutoff)
    state = daily["TOC"].ffill(limit=7).loc[cutoff - pd.Timedelta(days=6):cutoff]
    x["TOC_state_mean_7d"] = float(state.mean()) if state.notna().sum() == 7 else np.nan
    sol = daily["Solar"].loc[cutoff - pd.Timedelta(days=6):cutoff]
    x["Solar_mean_7d"] = float(sol.mean()) if sol.notna().sum() == 7 else np.nan
    flo = daily["Flow"].loc[cutoff - pd.Timedelta(days=2):cutoff]
    x["Flow_mean_3d"] = float(flo.mean()) if flo.notna().sum() == 3 else np.nan
    x["SPI30d_gamma"] = float(spi["SPI30d_gamma"].get(cutoff, np.nan))
    x["Dry_days_lt1mm"] = float(spi["Dry_days_lt1mm"].get(cutoff, np.nan))
    x["DOY_sin"] = float(np.sin(2 * np.pi * issue.dayofyear / 365.2425))
    return x, meta, cutoff, issue


def predict(model, feats, rows):
    X = pd.DataFrame(rows)[feats].astype(float)
    s = model.predict_proba(X)[:, 1]
    c = model.get_booster().predict(xgb.DMatrix(X, missing=np.nan, feature_names=feats), pred_contribs=True)[:, :-1]
    return s, c


def kdate(d, year=False):
    return (f"{d.year}년 " if year else "") + f"{d.month}월 {d.day}일"


def fmt(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "자료 없음"
    t = f"{v:,.0f}" if (abs(v) >= 100 or unit == "일") else f"{v:.1f}"
    return f"{t} {unit}".strip()


LEVEL_TEXT = {  # (높을 때, 낮을 때, 평소) — 값의 의미를 쉬운 말로
    "Temp_last": ("물이 평소보다 따뜻해요", "물이 평소보다 차가워요", "평소 수온이에요"),
    "Solar_mean_7d": ("햇빛이 평소보다 강했어요", "햇빛이 평소보다 약했어요", "평소 햇빛이에요"),
    "Flow_mean_3d": ("강물이 평소보다 많이 흘러요", "강물이 평소보다 적게 흘러요", "평소 유량이에요"),
    "SPI30d_gamma": ("최근 30일 비가 평소보다 많았어요", "최근 30일 평소보다 가물었어요", "평소 강수량이에요"),
    "Dry_days_lt1mm": ("비 없는 날이 길게 이어졌어요", "최근에 비가 왔어요", "평소 수준이에요"),
    "TOC_last": ("유기물이 평소보다 많아요", "유기물이 평소보다 적어요", "평소 수준의 유기물이에요"),
    "TOC_state_mean_7d": ("유기물이 평소보다 많아요", "유기물이 평소보다 적어요", "평소 수준의 유기물이에요"),
}


def level_word(f, v, dev):
    if np.isnan(v) or f not in dev.columns:
        return ""
    lo, hi = dev[f].quantile([0.3, 0.7])
    hi_t, lo_t, mid_t = LEVEL_TEXT.get(f, ("평소보다 높아요", "평소보다 낮아요", "평소 수준이에요"))
    return hi_t if v > hi else lo_t if v < lo else mid_t


def render_map(x, risk, target):
    """한글 지명이 나오는 OpenStreetMap 위에 지점 카드를 올린 지도 (Leaflet)."""
    hub = STATIONS["칠서"]
    col = RED if risk else BLUE
    status = "위험 기준 초과 예상" if risk else "위험 기준 미만 예상"
    marks = []
    for name, stn in STATIONS.items():
        vals = [f"{INFO[f][0]} {fmt(x[f], INFO[f][1])}" for f in FEATS if f in INFO and INFO[f][2] == name]
        marks.append(dict(name=name, lat=stn["lat"], lon=stn["lon"], tag=stn["tag"], icon=stn["icon"],
                          val=" · ".join(vals), hub=(name == "칠서"), approx=not stn["exact"], pos=stn.get("pos", "top")))
    data = json.dumps(dict(marks=marks, hub=[hub["lat"], hub["lon"]], col=col, status=status,
                           day=f"{target.month}월 {target.day}일"), ensure_ascii=False)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body,#map{{margin:0;height:100%;font-family:Pretendard,'Malgun Gothic',sans-serif;}}
 #map{{border-radius:20px;overflow:hidden;background:#eef1f4;}}
 .leaflet-tile-pane{{filter:grayscale(.55) saturate(.7) brightness(1.06) contrast(.92);}}
 .leaflet-container a{{color:#8B95A1}}
 .pin{{position:relative;transform:translate(-50%,-50%);}}
 .dot{{width:16px;height:16px;border-radius:50%;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.25);}}
 .hubdot{{width:22px;height:22px;}}
 .pulse{{position:absolute;left:50%;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
   animation:pulse 1.8s ease-out infinite;}}
 @keyframes pulse{{0%{{transform:scale(1);opacity:.6}}100%{{transform:scale(3.4);opacity:0}}}}
 .card{{position:absolute;white-space:nowrap;background:#fff;
   border-radius:14px;padding:8px 12px;box-shadow:0 6px 20px rgba(0,0,0,.12);font-size:13px;color:#191F28;}}
 .card.top{{left:50%;bottom:24px;transform:translateX(-50%);}}
 .card.bottom{{left:50%;top:24px;transform:translateX(-50%);}}
 .card.left{{right:24px;top:50%;transform:translateY(-50%);}}
 .card.right{{left:24px;top:50%;transform:translateY(-50%);}}
 .card.br{{left:20px;top:18px;}}
 .card b{{font-size:14px}} .card .s{{color:#8B95A1;font-size:12px;margin-top:2px}}
 .hubcard{{padding:10px 14px;border:2px solid currentColor;}} .hubcard .st{{font-weight:700;font-size:14px;margin-top:2px}}
 .legend{{position:absolute;left:14px;bottom:14px;z-index:999;background:#fff;border-radius:14px;padding:10px 14px;
   box-shadow:0 4px 16px rgba(0,0,0,.10);font-size:12px;color:#4E5968;line-height:1.9}}
 .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
</style></head><body><div id="map"></div>
<div class="legend"><b style="color:#191F28;font-size:13px">어떻게 예측하나요?</b><br>
<i style="background:{col}"></i>칠서 · 녹조를 예측하는 곳<br>
<i style="background:#4E5968"></i>관측 지점 · 예측에 쓴 자료<br>
<span style="color:#8B95A1">점선 = 자료가 모이는 방향</span></div>
<script>
const D = {data};
const map = L.map('map', {{zoomControl:false, attributionControl:true}});
L.control.zoom({{position:'bottomright'}}).addTo(map);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:18, attribution:'© OpenStreetMap'}}).addTo(map);
const pts = [];
D.marks.forEach(m => {{
  pts.push([m.lat, m.lon]);
  if (!m.hub) L.polyline([[m.lat, m.lon], D.hub], {{color:'#8B95A1', weight:2, dashArray:'6 6', opacity:.9}}).addTo(map);
}});
D.marks.forEach(m => {{
  let h;
  if (m.hub) {{
    h = `<div class="pin"><div class="pulse" style="background:${{D.col}}"></div>
         <div class="dot hubdot" style="background:${{D.col}}"></div>
         <div class="card hubcard ${{m.pos}}" style="color:${{D.col}}"><b style="color:#191F28">🎯 칠서 · ${{D.day}}</b><div class="st" style="color:${{D.col}}">${{D.status}}</div></div></div>`;
  }} else {{
    h = `<div class="pin"><div class="dot" style="background:#4E5968"></div>
         <div class="card ${{m.pos}}"><b>${{m.icon}} ${{m.name}} · ${{m.tag}}</b><div class="s">${{m.val || ''}}${{m.approx ? ' · 추정 위치' : ''}}</div></div></div>`;
  }}
  L.marker([m.lat, m.lon], {{icon: L.divIcon({{html:h, className:'', iconSize:[0,0]}}), zIndexOffset: m.hub ? 1000 : 0}}).addTo(map);
}});
map.fitBounds(pts, {{paddingTopLeft:[120,120], paddingBottomRight:[300,110]}});
map.on('click', e => L.popup({{closeButton:false}}).setLatLng(e.latlng)
  .setContent(`<span style="font-size:12px">위도 ${{e.latlng.lat.toFixed(5)}}<br>경도 ${{e.latlng.lng.toFixed(5)}}</span>`).openOn(map));
</script></body></html>"""


def fig_b64(fig):
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style_axes(ax):
    for k in ["top", "right", "left"]:
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=SUB, labelsize=9, length=0)
    ax.grid(axis="y", color=BG, lw=1)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------
# 준비 (엑셀 · 학습 결과)
# ------------------------------------------------------------------
xlsx = find_excel()
with st.sidebar:
    st.markdown("**설정**")
    up = st.file_uploader("다른 공통 엑셀 사용", type=["xlsx"])
    if up is not None:
        xlsx = "uploaded_common.xlsx"
        with open(xlsx, "wb") as f:
            f.write(up.getbuffer())
    show_recent_obs = st.toggle("2024~2025 참고 기간 실측도 보기", value=False,
                                help="2024~2025년은 개발 중 열람한 기간이라 성능 평가에 쓰지 않습니다. "
                                     "검증(2020~2023)과 독립 테스트(2026) 기간은 실측을 항상 함께 보여줍니다.")

if xlsx is None:
    st.warning("공통 엑셀이 없어요. 왼쪽 사이드바에서 올려 주세요.")
    st.stop()

if not (os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE)):
    html('<div class="h1">처음 한 번, test.py 를 실행해 주세요</div>'
         '<div class="sub">test.py 가 2016~2023 자료로 모델을 학습·고정하고 2026 테스트를 수행해 '
         'frozen_model.json · frozen_config.json · test_predictions.csv 를 만들어요. 부트스트랩 500회 때문에 수 분 걸려요.</div>')
    if not os.path.exists("test.py"):
        st.error("test.py 가 같은 폴더에 없어요.")
        st.stop()
    if st.button("test.py 실행", type="primary", width="stretch"):
        with st.spinner("학습·테스트하고 있어요..."):
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            res = subprocess.run([sys.executable, "test.py"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        with open("학습_로그.txt", "w", encoding="utf-8") as f:
            f.write(res.stdout + "\n" + res.stderr)
        if res.returncode == 0:
            st.rerun()
        st.error("학습 중 오류가 났어요. 아래 내용을 캡처해 주세요.")
        st.code(res.stderr[-3000:])
    st.stop()

cfg, model = load_model(os.path.getmtime(MODEL_FILE))
FEATS, THR = cfg["features"], float(cfg["threshold"])
daily, spi, full, dev = load_raw(xlsx)
FOLDS = fold_models(xlsx, list(FEATS))
SHOW_FROM = pd.Timestamp("2020-01-01")   # 2016~2019는 최종 모델의 학습 결과라 선택 불가
first_day = max(daily.index.min() + pd.Timedelta(days=12), SHOW_FROM)
last_day = daily.index.max()
CI = load_or_build_ci()

# ------------------------------------------------------------------
# 머리말 (제목 + 날짜 선택)
# ------------------------------------------------------------------
PRESETS = {  # 대표 사례 (공통 엑셀 실측 기준)
    "[테스트] 2026년 8월 · 최고 농도": "2026-08-10",
    "[테스트] 2026년 6월 · 녹조 시작": "2026-06-01",
    "[테스트] 2026년 3월 · 평상시": "2026-03-16",
    "[검증] 2022년 7월 · 녹조 대발생": "2022-07-28",
    "[검증] 2023년 11월 · 늦가을 녹조": "2023-11-27",
    "[검증] 2022년 3월 · 평상시": "2022-03-14",
}
lo_d, hi_d = first_day.date(), last_day.date()
import calendar
import datetime as _dt


def sync_ymd():
    d = st.session_state.target_date
    st.session_state.sel_y, st.session_state.sel_m, st.session_state.sel_d = d.year, d.month, d.day


def set_date(d):
    st.session_state.target_date = min(max(d, lo_d), hi_d)
    sync_ymd()


if "target_date" not in st.session_state:
    set_date(min(pd.Timestamp("2026-08-10"), last_day).date())


def on_ymd():
    y, m = st.session_state.sel_y, st.session_state.sel_m
    d = min(st.session_state.sel_d, calendar.monthrange(y, m)[1])
    set_date(_dt.date(y, m, d))


def shift_day(n):
    set_date(st.session_state.target_date + _dt.timedelta(days=n))


def set_preset():
    v = st.session_state.get("preset")
    if v in PRESETS:
        set_date(pd.Timestamp(PRESETS[v]).date())


html(f"""
<div class="topbar">
  <div class="brand"><span class="logo">🌊</span><span>칠서 녹조 예보</span></div>
  <div class="topmeta">캡스톤디자인 2조 · 검증 2020~2023 · 독립 테스트 2026.1~9 · 공식 조류경보 아님</div>
</div>
<div class="h1">3일 뒤 녹조, 미리 알아봐요</div>
<div class="sub" style="margin:-8px 0 18px;">날짜를 고르면 그 날 낙동강 칠서 지점의 녹조 위험을 3일 전 자료로 예측해 보여드려요</div>
""")
with st.container(key="datebar"):
    cy, cm, cd, cp, cn, cs = st.columns([0.8, 0.6, 0.6, 0.55, 0.55, 1.5], vertical_alignment="bottom")
    with cy:
        st.selectbox("연도", list(range(lo_d.year, hi_d.year + 1)), key="sel_y", on_change=on_ymd,
                     format_func=lambda v: f"{v}년")
    with cm:
        st.selectbox("월", list(range(1, 13)), key="sel_m", on_change=on_ymd, format_func=lambda v: f"{v}월")
    with cd:
        n_days = calendar.monthrange(st.session_state.sel_y, st.session_state.sel_m)[1]
        if st.session_state.sel_d > n_days:
            st.session_state.sel_d = n_days
        st.selectbox("일", list(range(1, n_days + 1)), key="sel_d", on_change=on_ymd, format_func=lambda v: f"{v}일")
    with cp:
        st.button("◀ 전날", on_click=shift_day, args=(-1,), width="stretch")
    with cn:
        st.button("다음 날 ▶", on_click=shift_day, args=(1,), width="stretch")
    with cs:
        st.selectbox("대표 사례로 바로 가기", ["선택하세요"] + list(PRESETS), key="preset", on_change=set_preset)
target = pd.Timestamp(st.session_state.target_date)

tab_main, tab_info = st.tabs(["예보", "예측 방법"])

# ==================================================================
# 예보: 왼쪽 큰 지도 · 오른쪽 정보 카드
# ==================================================================
with tab_main:
    x, meta, cutoff, issue = build_inputs(daily, spi, target)
    used_model, kind = pick_model(target)
    s, sv = predict(used_model, FEATS, [x])
    score, contrib = float(s[0]), sv[0]
    risk = score >= THR
    p_ico, p_title, p_desc = PERIOD_NOTE[kind]
    p_title, p_desc = p_title.format(y=target.year), p_desc.format(y=target.year)
    bg, fg, accent = ("#FFEEF0", "#B4232F", RED) if risk else ("#E8F3FF", "#1B64DA", BLUE)
    band = ci_of(target)
    conf = confidence(risk, band)
    obs_visible = kind != "reference" or show_recent_obs
    obs_val = daily["Cyano"].get(target, np.nan) if "Cyano" in daily.columns else np.nan

    # ===== 1줄: 지도 | 결과 + 판단 근거 (높이를 맞춤) =====
    col_map, col_info = st.columns([1.55, 1], gap="large")
    with col_map:
        components.html(render_map(x, risk, target), height=770)

    with col_info:
        headline = "녹조 위험 기준을<br>넘을 것 같아요" if risk else "녹조 위험 기준을<br>넘지 않을 것 같아요"
        chip_bg = "#FFEEF0" if risk else "#E8F3FF"
        status = "위험 기준 초과 예상" if risk else "위험 기준 미만 예상"
        b_txt, b_bg, b_fg = PERIOD_BADGE[kind]
        badges = f'<span class="chip" style="background:{b_bg}; color:{b_fg};">{b_txt}</span>'
        band_div, band_txt, conf_line = "", "", ""
        if band is not None:
            lo_, hi_ = band
            band_div = (f'<div class="band" style="left:{lo_*100:.1f}%; width:{max(hi_-lo_, 0.006)*100:.1f}%;'
                        f' background:{accent};"></div>')
            band_txt = f' <span style="color:{SUB}">· 95% 범위 {lo_:.2f}~{hi_:.2f}</span>'
            if conf:
                c_chip = f'<span class="chip" style="background:{BG}; color:{INK};">신뢰도 높음</span>'
                c_msg = ("95% 범위 전체가 경보 기준보다 높아요. 믿을 만한 위험 신호예요." if risk else
                         "95% 범위 전체가 경보 기준보다 낮아요. 안정적인 정상 판단이에요.")
            else:
                c_chip = '<span class="chip" style="background:#FFF4E5; color:#B26A00;">신뢰도 낮음 · 판정 경계</span>'
                c_msg = ("경보는 냈지만 95% 범위가 기준선에 걸쳐 있어요. 현장 채수로 확인을 권해요." if risk else
                         "정상으로 판단했지만 95% 범위가 기준선에 걸쳐 있어요. 가까운 날짜도 확인해 주세요.")
            badges += " " + c_chip
            conf_line = f'<div class="cline">{c_msg}</div>'
        result_line = ""
        if obs_visible and not np.isnan(obs_val):
            real = obs_val >= 1000
            res_txt, res_col = {(True, True): ("위험 탐지 ✅", RED), (False, False): ("정상 적중 ✅", BLUE),
                                (False, True): ("미탐 ❌ (위험을 놓침)", INK),
                                (True, False): ("오경보 ⚠️ (실제는 정상)", "#B26A00")}[(bool(risk), bool(real))]
            result_line = (f'<div class="obsline"><span>실제 측정 <b>{obs_val:,.0f}</b> cells/mL</span>'
                           f'<b style="color:{res_col}">{res_txt}</b></div>')
        elif kind == "reference" and not np.isnan(obs_val):
            result_line = '<div class="obsline"><span>실측은 참고 기간이라 숨겼어요 (왼쪽 설정에서 켤 수 있어요)</span></div>'
        html(f"""
        <div class="card hero">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="label">{kdate(target, True)} · 칠서</span>
            <span class="chip" style="background:{chip_bg}; color:{accent};">● {status}</span>
          </div>
          <div class="big" style="color:{fg};">{headline}</div>
          <div class="desc">3일 뒤 유해남조류가 1,000 cells/mL 이상일지 예측했어요</div>
          <div style="margin-top:12px; display:flex; gap:6px; flex-wrap:wrap;">{badges}</div>
          <div class="gauge">{band_div}<div class="fill" style="width:{score*100:.0f}%; background:{accent};"></div>
            <div class="mark" style="left:calc({THR*100:.1f}% - 1px);"></div></div>
          <div class="gscale"><span>위험 신호 <b style="color:{INK}">{score:.2f}</b>{band_txt}</span><span>경보 기준 {THR:.2f}</span></div>
          {conf_line}{result_line}
          <div class="foot"><span style="font-size:16px;">{p_ico}</span>
            <div><b>{p_title}</b><br>{p_desc}</div></div>
        </div>
        """)
        order = np.argsort(-np.abs(contrib))
        rows = []
        for i in order[:3]:
            f = FEATS[i]
            name, unit, stn, ico = INFO.get(f, (f, "", "", "•"))
            up_ = contrib[i] > 0
            tint = "#FFF0F1" if up_ else "#EEF5FF"
            chip = (f'<span class="chip" style="background:#FFEEF0;color:{RED};">위험 높임 ↑</span>' if up_ else
                    f'<span class="chip" style="background:#E8F3FF;color:{BLUE};">위험 낮춤 ↓</span>')
            if f == "DOY_sin":
                val, sub = f"{issue.month}월", ("녹조가 잘 생기는 시기예요" if up_ else "녹조가 덜 생기는 시기예요")
            else:
                val, sub = fmt(x[f], unit), (level_word(f, x[f], dev) or "최근 자료 없음")
            rows.append(f'<div class="row"><div class="ico" style="background:{tint}">{ico}</div>'
                        f'<div><div class="t">{name} <span style="font-weight:700">{val}</span></div>'
                        f'<div class="s">{sub}</div></div><div class="r">{chip}</div></div>')
        html('<div class="card" style="margin-bottom:0;"><div style="display:flex;justify-content:space-between;'
             'align-items:baseline;"><div class="ctitle">이렇게 판단했어요</div>'
             '<span class="label">영향이 큰 순서</span></div>' + "".join(rows) + "</div>")
    html('<div class="note">칠서·창원은 공식 좌표, 남지·청암리는 주소 기준 추정 위치예요. 지도를 누르면 그 자리의 좌표가 보여요 · '
         f'{kdate(issue)} 오전 10:30 예측 · {kdate(cutoff)}까지의 관측 자료 사용 · '
         '위험 신호는 확률이 아닌 모델 점수예요 · 공식 조류경보가 아니에요</div>')

    # ===== 2줄: 가져온 관측값 (한 줄) =====
    html('<div class="h2">가져온 관측값</div>')
    cells = []
    for f in FEATS:
        name, unit, stn, ico = INFO.get(f, (f, "", "", "•"))
        if f == "DOY_sin":
            v, n = f"{issue.month}월", "예측일 기준"
        else:
            v = fmt(x[f], unit)
            n = stn
            if meta.get(f) is not None:
                n += f" · {meta[f]:%m/%d} 측정"
            elif f in meta:
                n += " · 7일 내 측정 없음"
        cells.append(f'<div class="mini"><div class="k"><i>{ico}</i>{name}</div><div class="v">{v}</div>'
                     f'<div class="n">{n}</div></div>')
    html(f'<div class="gridn" style="grid-template-columns: repeat({len(cells)}, 1fr);">' + "".join(cells) + "</div>")
    if any(np.isnan(x[f]) for f in FEATS):
        html('<div class="note">⚠️ 비어 있는 값은 임의로 채우지 않고 결측으로 계산했어요.</div>')

    # ===== 3줄: 30일 흐름 | 시민 안내 =====
    html('<div style="height:18px"></div>')
    col_chart, col_tip = st.columns([1.55, 1], gap="large")
    with col_chart:
        days = pd.date_range(max(first_day, target - pd.Timedelta(days=29)), target)
        s30 = predict_dates(days)
        fig, ax = plt.subplots(figsize=(8.4, 3.45))
        y = np.asarray(s30, float)
        bands = [ci_of(d) for d in days]
        if all(b is not None for b in bands):   # 95% 신뢰 범위 (부트스트랩)
            ax.fill_between(days, [b[0] for b in bands], [b[1] for b in bands], color="#8B95A1", alpha=.13,
                            lw=0, zorder=0.5, step=None)
        # 선 아래 면: 경보 기준 위 = 빨강, 아래 = 파랑
        ax.fill_between(days, THR, y, where=y >= THR, interpolate=True, color=RED, alpha=.16, lw=0, zorder=1)
        ax.fill_between(days, THR, y, where=y < THR, interpolate=True, color=BLUE, alpha=.12, lw=0, zorder=1)
        ax.plot(days, y, color=INK, lw=2.6, zorder=3, solid_capstyle="round")
        ax.axhline(THR, color=RED, lw=1.3, ls=(0, (5, 4)), zorder=2)
        ax.text(days[-1] + pd.Timedelta(hours=12), THR, f" 경보 기준\n {THR:.2f}", color=RED, fontsize=10,
                va="center", ha="left", fontweight="bold")
        # 고른 날짜 강조
        ax.scatter([target], [score], s=120, color=accent, zorder=5, edgecolor="white", linewidth=2.5)
        ax.annotate(f"{target.month}/{target.day}  {score:.2f}", (target, score), xytext=(-10, 14 if score < 0.85 else -22),
                    textcoords="offset points", ha="right", fontsize=11, fontweight="bold", color=accent)
        show_obs = obs_visible
        if show_obs and "Cyano" in daily.columns:
            obs = daily["Cyano"].loc[days.min():days.max()].dropna()
            ax.axhspan(-0.13, -0.03, color="#F7F8FA", zorder=0)
            ax.text(days[0] - pd.Timedelta(hours=18), -0.08, "실측", fontsize=9, color=SUB, va="center", ha="right")
            for d_, v_ in obs.items():
                ax.scatter([d_], [-0.08], marker="s", s=55, color=RED if v_ >= 1000 else "#B0B8C1", zorder=4)
        ax.set_ylim(-0.15 if show_obs else 0, 1.05)
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["낮음  0", "0.5", "높음  1"])
        ax.set_xlim(days[0] - pd.Timedelta(hours=12), days[-1] + pd.Timedelta(hours=12))
        style_axes(ax)
        ax.tick_params(colors="#4E5968", labelsize=10.5)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.tight_layout()
        legend = ""
        if show_obs:
            extra = " · 참고 기간 실측(성능 평가 아님)" if kind == "reference" else ""
            legend = (f'<span style="color:{RED}">■</span> 실측 1,000 이상 &nbsp; '
                      f'<span style="color:#B0B8C1">■</span> 실측 미만{extra}')
        html('<div class="card eqh" style="margin-bottom:0;"><div style="display:flex;justify-content:space-between;'
             'align-items:baseline;"><div class="ctitle">최근 30일 흐름</div>'
             f'<span class="label"><span style="color:{RED}">■</span> 경보 기준 초과 &nbsp;<span style="color:{BLUE}">■</span> 기준 미만'
             + (' &nbsp;<span style="color:#C4CAD1">■</span> 95% 범위' if CI is not None else '') + '</span></div>'
             f'<div class="eqimg"><img src="data:image/png;base64,{fig_b64(fig)}"></div>'
             f'<div class="note" style="margin-top:2px;">{legend or "&nbsp;"}</div></div>')

    with col_tip:
        if risk:
            tips = [("🚫", "강물에 닿는 활동은 피해 주세요", "물놀이 · 수상레저 · 맨손 낚시"),
                    ("🐕", "반려동물이 강물을 마시지 않게", "녹조 낀 물가에 가까이 가지 않게"),
                    ("🧼", "닿았다면 깨끗한 물로 씻어 주세요", "가렵거나 따가우면 진료를 받아 보세요")]
        else:
            tips = [("😊", "녹조 걱정은 크지 않아요", "물이 초록빛이면 접촉은 피해 주세요"),
                    ("📅", "여름·가을엔 빨리 바뀌어요", "가까운 날짜도 함께 확인해 보세요"),
                    ("🌡️", "더운 날이 이어지면 다시 확인", "수온이 오르면 녹조가 빨리 늘어요")]
        tips.append(("🔎", "공식 조류경보는 물환경정보시스템", "이 화면은 대학 수업 연구 시연이에요"))
        html('<div class="card eqh" style="margin-bottom:0;"><div class="ctitle">이럴 땐 이렇게 해요</div>'
             '<div class="spread">' + "".join(
            f'<div class="row"><div class="ico">{i}</div><div><div class="t" style="font-size:15px;">{t}</div>'
            f'<div class="s">{d}</div></div></div>' for i, t, d in tips) + "</div></div>")

    html('<div style="height:18px"></div>')
    with st.expander("녹조(유해남조류)가 뭐예요?"):
        html('<div class="note" style="color:#4E5968;">물이 따뜻하고 잘 흐르지 않을 때 남조류가 크게 늘어나 강물이 '
             '초록빛으로 변하는 현상이에요. 일부 남조류는 독소나 냄새 물질을 만들어요. '
             '낙동강 칠서 지점은 창원 지역 취수장과 가까워 조류경보를 운영하는 곳이에요. '
             '이 예보는 3일 뒤 유해남조류가 1 mL에 1,000개 이상일지(조류경보 \'관심\' 기준 농도)를 미리 알려줘요.</div>')

# ==================================================================
# 모델 정보
# ==================================================================
with tab_info:
    names = " · ".join(dict.fromkeys(INFO.get(f, (f,))[0] for f in FEATS))
    ci_row = ""
    if HAS_MBB:
        cs = ci_settings()
        ci_row = (f'<div class="row"><div class="ico">📏</div><div><div class="t">신뢰도 = 95% 범위가 경보 기준을 걸치는지</div>'
                  f'<div class="s">이동 블록 부트스트랩(블록 {cs["L"]}, {cs["B"]}회)으로 위험 신호의 95% 범위를 구해요. '
                  "경보 판정은 그대로 두고, 범위가 기준선을 걸치지 않으면 '신뢰도 높음', 걸치면 '신뢰도 낮음(경계)'이에요"
                  '</div></div></div>')
    html(f"""
    <div class="card">
      <div class="row"><div class="ico">🧠</div><div><div class="t">XGBoost · 입력 {len(FEATS)}개</div>
        <div class="s">{names}</div></div></div>
      <div class="row"><div class="ico">📚</div><div><div class="t">2016~2023년 자료로 학습한 뒤 고정</div>
        <div class="s">모델·입력·경보 기준을 고정한 뒤 2026년 자료를 붙여 평가했어요</div></div></div>
      <div class="row"><div class="ico">🗓️</div><div><div class="t">기간마다 역할이 달라요</div>
        <div class="s"><b>2020~2023 검증</b> 그 해 이전 자료로만 학습한 모델로 예측 ·
          <b>2024~2025 참고</b> 개발 중 열람해 성능 평가에서 제외 ·
          <b>2026.1~9 독립 테스트</b> 고정 모델로 처음 본 자료 예측</div></div></div>
      <div class="row"><div class="ico">🎯</div><div><div class="t">경보 기준 {THR:.3f}</div>
        <div class="s">2020~2023 검증에서 위험일 85% 이상을 잡으면서 오경보가 가장 적은 값</div></div></div>
      {ci_row}
    </div>
    """)

    # ---------- 검증 vs 독립 테스트 성적 ----------
    def score_row(label, y, s_, dates):
        a = s_ >= THR - 1e-9          # CSV 저장 시 소수점 끝자리 오차 보정
        tp, fn = int((a & (y == 1)).sum()), int((~a & (y == 1)).sum())
        fp, tn = int((a & (y == 0)).sum()), int((~a & (y == 0)).sum())
        f = lambda t: "-" if np.isnan(t[0]) else f"{t[0]:.2f} ({t[1]:.2f}~{t[2]:.2f})"
        row = {"기간": label, "건수": len(y), "실제 위험": int(y.sum()), "잡은 위험일": tp, "놓친 위험일": fn,
               "오경보": fp, "탐지율 (95% 범위)": f(wilson(tp, tp + fn)),
               "정상일 중 경보 비율 (95% 범위)": f(wilson(fp, fp + tn)),
               "경보 적중률 (95% 범위)": f(wilson(tp, tp + fp))}
        conf_rows = []
        if CI is not None:
            confs = [confidence(a_, ci_of(pd.Timestamp(d_))) for d_, a_ in zip(dates, a)]
            for lab, sel_risk in [("위험 판정", True), ("정상 판정", False)]:
                for hi_conf in [True, False]:
                    m = np.array([a_ == sel_risk and c_ is not None and c_ == hi_conf
                                  for a_, c_ in zip(a, confs)], bool)
                    n_, k_ = int(m.sum()), int(y[m].sum())
                    conf_rows.append({"기간": label, "판정": lab, "신뢰도": "높음" if hi_conf else "낮음(경계)",
                                      "건수": n_, "실제 위험": k_,
                                      "실제 위험 비율": "-" if n_ == 0 else f"{k_ / n_:.2f}"})
        return row, conf_rows

    rows, crow = [], []
    oo = validation_oof()
    if len(oo):
        r_, c_ = score_row("2020~2023 검증", oo[fm.TARGET].to_numpy().astype(int), oo["score"].to_numpy(),
                           oo["target_date"])
        rows.append(r_); crow += c_
    test = full[full["split"] == TEST_SPLIT] if "split" in full.columns else full.iloc[0:0]
    if len(test):
        ts = predict(model, FEATS, test[FEATS].to_dict("records"))[0]
        ty = test[fm.TARGET].to_numpy().astype(int)
        r_, c_ = score_row("2026 독립 테스트", ty, ts, test["target_date"])
        rows.append(r_); crow += c_
    if rows:
        html('<div class="h2">검증 vs 독립 테스트</div>'
             '<div class="sub" style="margin:-4px 0 12px;">같은 경보 기준으로 채점했어요 · 2024~2025 참고 기간은 채점하지 않아요 · '
             '괄호는 윌슨 95% 범위</div>')
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(test):
            n_miss = int(test[FEATS].isna().any(axis=1).sum())
            html(f'<div class="note">2026 테스트는 {len(test)}건(실제 위험 {int(ty.sum())}건)이라 범위가 넓어요. '
                 '점수 하나보다 검증 범위와 겹치는지를 보세요.'
                 + (f' 입력 일부가 비어 있는 날 {n_miss}건은 채우지 않고 결측 그대로 예측했어요.' if n_miss else '')
                 + '</div>')
    if crow:
        html('<div class="h2">신뢰도별 실제 위험 비율</div>'
             '<div class="sub" style="margin:-4px 0 12px;">신뢰도 높은 위험의 실제 위험 비율이 높고, 신뢰도 높은 정상의 비율이 낮을수록 '
             '신뢰도 구분이 의미 있어요</div>')
        st.dataframe(pd.DataFrame(crow), hide_index=True, width="stretch")

    # ---------- 2026 테스트 날짜별 ----------
    if len(test):
        html('<div class="h2">2026 독립 테스트 · 날짜별 판정</div>')
        td = pd.to_datetime(test["target_date"]).to_numpy()
        fig, ax = plt.subplots(figsize=(11, 3.6))
        bb = [ci_of(pd.Timestamp(d)) for d in td]
        if all(b is not None for b in bb):
            ax.vlines(td, [b[0] for b in bb], [b[1] for b in bb], color="#C4CAD1", lw=3, zorder=1)
        ax.plot(td, ts, color=INK, lw=1.2, alpha=.6, zorder=2)
        ax.axhline(THR, color=RED, lw=1.2, ls=(0, (5, 4)))
        a_ = ts >= THR
        for lab, sel, mk, col in [("탐지", a_ & (ty == 1), "o", RED), ("정상 적중", ~a_ & (ty == 0), "o", BLUE),
                                  ("미탐", ~a_ & (ty == 1), "X", INK), ("오경보", a_ & (ty == 0), "^", "#FF9F0A")]:
            if sel.any():
                ax.scatter(td[sel], ts[sel], marker=mk, s=60, color=col, zorder=4, label=f"{lab} {int(sel.sum())}",
                           edgecolor="white", linewidth=1)
        ax.set_ylim(0, 1.02); ax.set_yticks([0, THR, 1]); ax.set_yticklabels(["0", f"기준 {THR:.2f}", "1"])
        style_axes(ax)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
        ax.legend(loc="upper left", fontsize=9, ncol=4, frameon=False)
        fig.tight_layout()
        html(f'<div class="card"><img style="width:100%" src="data:image/png;base64,{fig_b64(fig)}">'
             '<div class="note">회색 세로줄: 95% 범위 · 점: 위험 신호(3일 전 예측) · 빨간 점선: 경보 기준</div></div>')

    # 2026 계절별 (test.py 결과 파일)
    if os.path.exists(TEST_METRICS):
        tm = pd.read_csv(TEST_METRICS)
        keep = [c for c in ["group", "n", "positive", "negative", "TP", "FN", "FP", "TN", "Recall", "Precision",
                            "F1", "FPR", "AP"] if c in tm.columns]
        html('<div class="h2">2026 테스트 · 계절별 (test.py 결과)</div>'
             '<div class="sub" style="margin:-4px 0 12px;">여름·9월은 실제 위험만 있어서 정상 구분 능력은 겨울·봄 결과로 봐야 해요</div>')
        st.dataframe(tm[keep].rename(columns={"group": "구분", "n": "건수", "positive": "실제 위험",
                                              "negative": "실제 정상", "TP": "탐지", "FN": "미탐",
                                              "FP": "오경보", "TN": "정상 적중"}).round(3),
                     hide_index=True, width="stretch")

    if len(oo):
        html('<div class="h2">연도별 검증 결과</div>'
             '<div class="sub" style="margin:-4px 0 12px;">각 연도를 그 이전 자료로만 학습한 모델로 예측했어요</div>')
        yr_rows = []
        for yr_, g in oo.groupby(oo["target_date"].dt.year):
            yy, aa = g[fm.TARGET].to_numpy().astype(int), g["score"].to_numpy() >= THR - 1e-9
            tp_, fn_ = int((aa & (yy == 1)).sum()), int((~aa & (yy == 1)).sum())
            fp_, tn_ = int((aa & (yy == 0)).sum()), int((~aa & (yy == 0)).sum())
            yr_rows.append({"연도": yr_, "건수": len(g), "잡은 위험일": tp_, "놓친 위험일": fn_, "오경보": fp_,
                            "탐지율": round(tp_ / max(tp_ + fn_, 1), 3),
                            "정상일 중 경보 비율": round(fp_ / max(fp_ + tn_, 1), 3)})
        st.dataframe(pd.DataFrame(yr_rows), hide_index=True, width="stretch")

    html('<div class="h2">해석할 때 주의할 점</div><div class="card"><div class="note" style="margin:0;color:#4E5968;">'
         '· 2026 테스트는 1~9월 중 남조류 수치가 공개된 44건이에요. 10~12월은 포함되지 않아요.<br>'
         '· 입력 일부가 비어 있는 4건은 채우지 않고 XGBoost의 결측 처리로 예측했어요.<br>'
         '· 수질 실제 공개시각 미확인, 2026 유량 잠정값, 2025년 칠서 채수지점 변경을 고려해야 해요. 후향적 새 연도 평가이며 실시간 운영 실증은 아니에요.<br>'
         '· 위험 신호와 95% 범위는 보정된 발생확률이 아니에요. SHAP 근거는 예측 기여이며 인과관계 증명이 아니에요.<br>'
         f'· 경보 기준 {THR:.6f}은 2026 예측 전에 2016~2023 자료로 고정한 값이에요'
         + (f' (이전 실행본 {cfg["previous_run_threshold"]:.6f}과 다르며, 2026 결과로 고른 값이 아니에요)'
            if cfg.get("previous_run_threshold") else '') + '.</div></div>')

    def show(fname, cap):
        if GRAPH_DIR is None:
            return
        fp = os.path.join(GRAPH_DIR, fname)
        if os.path.exists(fp):
            html(f'<div class="h2">{cap}</div>')
            st.image(fp, width="stretch")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        show("perf_1_yearly_ap.png", "최종 모델 vs B6 원안")
        show("perf_3_threshold_tradeoff.png", "경보 기준을 고른 이유")
        show("shap_toc_x_SPI30d_gamma.png", "유기물 영향 × 가뭄")
    with g2:
        show("perf_4_confusion_by_year.png", "연도별 탐지 · 미탐 · 오경보")
        show("shap_bar.png", "어떤 요인이 중요했나")
        show("perf_5_ablation.png", "요인을 하나씩 뺐을 때")
    show("perf_6_mbb_ci_zones.png", "검증 예측의 95% 범위 (이동 블록 부트스트랩)")

html(f"""
<div class="footer">
  <div><b>칠서 녹조 예보</b> · 캡스톤디자인 2조 연구 시연 · 공식 조류경보가 아닙니다</div>
  <div>자료: 공통 엑셀 (일별 원자료 · 강수와 SPI, 2026년 9월까지) · 기상청 창원(155) · 남지 수질 · 창녕군(청암리) 유량 · 조류경보제 칠서 지점</div>
  <div>모델: XGBoost (2016~2023 학습 후 고정) · 검증 2020~2023 · 참고 2024~2025(성능 평가 제외) · 독립 테스트 2026.1~9 · 2026 유량은 검보정 전 잠정자료</div>
</div>
""")
