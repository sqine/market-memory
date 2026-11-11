from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''import sys, yaml
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 200)
CFG = yaml.safe_load(open(ROOT / "configs/base.yaml"))
RES = ROOT / "results"; (RES / "figures").mkdir(parents=True, exist_ok=True)'''

ORIGINAL = r'''ORIG = {
    ("1wk", "S&P 500"): (0.5547, 0.4918), ("1wk", "NASDAQ"): (0.5763, 0.5211), ("1wk", "Apple"): (0.5506, 0.5923),
    ("1wk", "Bitcoin"): (0.6553, 0.6016), ("1wk", "KOSPI"): (0.5495, 0.5254), ("1wk", "Samsung"): (0.5178, 0.5458),
    ("1wk", "Google"): (0.5399, 0.6052), ("1wk", "Intel"): (0.5266, 0.5144), ("1wk", "cocacola"): (0.4994, 0.5222),
    ("1wk", "nvdia"): (0.5611, 0.5790), ("1wk", "tesla"): (0.5990, 0.5901), ("1wk", "sk hynix"): (0.5336, 0.5661),
    ("1wk", "eth"): (0.6655, 0.5291),
    ("1mo", "S&P 500"): (0.6102, 0.6159), ("1mo", "NASDAQ"): (0.6122, 0.5124), ("1mo", "Apple"): (0.5604, 0.5762),
    ("1mo", "Bitcoin"): (0.7886, 0.6156), ("1mo", "KOSPI"): (0.5765, 0.5977), ("1mo", "Samsung"): (0.5198, 0.5465),
    ("1mo", "Google"): (0.5674, 0.5819), ("1mo", "Intel"): (0.5196, 0.5119), ("1mo", "cocacola"): (0.5245, 0.5313),
    ("1mo", "nvdia"): (0.5705, 0.6883), ("1mo", "tesla"): (0.7061, 0.5131), ("1mo", "sk hynix"): (0.5320, 0.5876),
    ("1mo", "eth"): (0.7653, 0.6983),
}'''


def hurst_nb():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md("""# 01. 허스트 지수 (R/S 분석)

- 원본(2025-2 투자공학 과제)은 13개 자산의 월간·주간 로그수익률로 R/S 허스트 지수를 구하고, 수익률을 **한 번** 섞은 값(Scrambled)과 비교했다.
- 여기서는 같은 R/S 계산을 그대로 재현한 뒤, 섞기를 **500번** 반복해 귀무분포(기억 없음)를 만들고 p값으로 판단한다.
- 데이터: Yahoo Finance 종가(원본 과제에서 받은 CSV), 2024-12까지."""),
        code(SETUP + "\n" + r'''from market_memory.hurst import hurst_test, log_returns, rs_hurst
H = CFG["hurst"]
def load(name, unit):
    return pd.read_csv(ROOT / H["data_dir"] / f"{name}_{unit}.csv")["Close"]'''),
        md("## 1. 원본 값 재현 (Unscrambled)"),
        code(ORIGINAL + "\n" + r'''rep = pd.DataFrame([{"주기": u, "자산": a, "원본 H": h, "재계산 H": rs_hurst(log_returns(load(a, u))), "원본 Scrambled(1회)": s}
                    for (u, a), (h, s) in ORIG.items()])
rep["차이"] = rep["재계산 H"] - rep["원본 H"]
print("최대 |차이|:", rep["차이"].abs().max().round(5))
rep.round(4)'''),
        md("## 2. 섞기 500회 귀무분포와 비교"),
        code(r'''rows = []
for u in H["units"]:
    for a in H["assets"]:
        r = hurst_test(load(a, u), n_shuffles=H["n_shuffles"], seed=H["seed"])
        rows.append({"주기": u, "자산": a, **{k: v for k, v in r.items()}})
tab = pd.DataFrame(rows)
tab["유의(5%)"] = tab["p_value"] < 0.05
tab.to_csv(RES / "hurst_table.csv", index=False)
tab.round(4)'''),
        md("## 3. 원본의 1회 Scrambled 값은 귀무분포 안 어디에 있었나"),
        code(r'''chk = rep.merge(tab[["주기", "자산", "null_mean", "null_lo", "null_hi"]], on=["주기", "자산"])
chk["1회 값이 95% 구간 안"] = chk["원본 Scrambled(1회)"].between(chk["null_lo"], chk["null_hi"])
print("원본 1회 Scrambled 값 중 95% 구간 안:", int(chk["1회 값이 95% 구간 안"].sum()), "/", len(chk))
print("원본에서 'Unscrambled > Scrambled' 였던 경우:", int((chk["원본 H"] > chk["원본 Scrambled(1회)"]).sum()), "/", len(chk))
chk[["주기", "자산", "원본 H", "원본 Scrambled(1회)", "null_mean", "null_lo", "null_hi"]].round(4)'''),
        md("## 4. 요약표"),
        code(r'''piv = tab.pivot(index="자산", columns="주기", values=["H", "null_mean", "p_value"]).reindex(H["assets"])
lines = ["| 자산 | " + " | ".join(f"{u} H (귀무 평균, p)" for u in H["units"]) + " |", "|---|" + "---|" * len(H["units"])]
for a in H["assets"]:
    cells = []
    for u in H["units"]:
        h, m, p = piv.loc[a, ("H", u)], piv.loc[a, ("null_mean", u)], piv.loc[a, ("p_value", u)]
        cells.append(f"{h:.3f} ({m:.3f}, {p:.3f}){' *' if p < 0.05 else ''}")
    lines.append(f"| {a} | " + " | ".join(cells) + " |")
md_t = "\n".join(lines)
(RES / "hurst_summary.md").write_text(md_t + "\n")
print(md_t)
print()
print(tab.groupby("주기")[["H", "null_mean"]].mean().round(4))
print(tab.groupby("주기")["유의(5%)"].sum())'''),
        md("## 5. 그림"),
        code(r'''fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, u in zip(axes, H["units"]):
    d = tab[tab["주기"] == u].set_index("자산").reindex(H["assets"])
    y = np.arange(len(d))
    ax.hlines(y, d["null_lo"], d["null_hi"], color="lightgray", lw=6, label="섞기 500회 95% 구간")
    ax.scatter(d["null_mean"], y, color="gray", s=12, label="섞기 평균")
    ax.scatter(d["H"], y, color=np.where(d["p_value"] < 0.05, "crimson", "steelblue"), zorder=3, label="실제 H (빨강=유의)")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.index); ax.set_title(f"{u} 로그수익률"); ax.grid(alpha=.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(RES / "figures/hurst_null.png", dpi=150); plt.show()'''),
    ]


def chaos_nb():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md("""# 02. 상관차원과 리아프노프 지수 (금, Gold)

- 원본(2025-2 투자공학 과제): 금 월간 가격을 CPI와 시간 추세로 회귀해 추세를 뺀 값(Detrended)으로 상관차원과 리아프노프 지수를 구했다. 비교 대상은 한 번 섞은 계열이었다.
- 섞기는 시계열의 자기상관까지 없앤다. 그래서 '섞은 것과 다르다'는 카오스가 아니라 단순한 자기상관만으로도 나올 수 있다.
- 여기서는 **위상 무작위화 대리계열(phase-randomized surrogate)** 을 추가한다. 이 계열은 자기상관(파워 스펙트럼)은 같고 비선형 구조만 없앤 것이다. 실제 계열이 이것과도 다를 때만 비선형 결정론의 증거로 본다.
- 대리계열 19개 → 단측 순위 검정 최소 p = 0.05."""),
        code(SETUP + "\n" + r'''from market_memory.chaos import (correlation_dimension, detrend_with_cpi, lyapunov_rosenstein,
                                  phase_surrogate, shuffle_surrogate, zscore)
C = CFG["chaos"]
gold = pd.read_csv(ROOT / C["gold_file"], parse_dates=["Date"]).set_index("Date")["Price"].dropna()
cpi = pd.read_csv(ROOT / C["cpi_file"], parse_dates=["DATE"]).set_index("DATE")["CPIAUCSL"]
monthly = gold.resample("MS").first().dropna()
detr = detrend_with_cpi(monthly, cpi)
daily_ret = np.log(gold / gold.shift(1)).dropna().iloc[-C["daily_window"]:]
SERIES = {"월간 추세 제거 가격": zscore(detr.values), f"일간 로그수익률(최근 {C['daily_window']}일)": zscore(daily_ret.values)}
print(f"월간: {detr.index.min().date()} ~ {detr.index.max().date()}, N={len(detr)}")
print(f"일간: {daily_ret.index.min().date()} ~ {daily_ret.index.max().date()}, N={len(daily_ret)}")'''),
        md("## 1. 상관차원 D2(m): 실제 vs 섞기 vs 위상 무작위화"),
        code(r'''radii = np.logspace(-2, 0.7, 40)
m_values = list(range(C["m_range"][0], C["m_range"][1] + 1))
rng = np.random.default_rng(C["seed"])
d2 = {}
for name, x in SERIES.items():
    sh = [shuffle_surrogate(x, rng) for _ in range(C["n_surrogates"])]
    ph = [phase_surrogate(x, rng) for _ in range(C["n_surrogates"])]
    rows = []
    for m in m_values:
        obs = correlation_dimension(x, m, radii, C["tau"])
        s_sh = np.array([correlation_dimension(s, m, radii, C["tau"]) for s in sh])
        s_ph = np.array([correlation_dimension(s, m, radii, C["tau"]) for s in ph])
        rows.append({"m": m, "실제": obs, "섞기 평균": np.nanmean(s_sh), "섞기 최소": np.nanmin(s_sh),
                     "위상 평균": np.nanmean(s_ph), "위상 최소": np.nanmin(s_ph),
                     "p(섞기보다 작음)": (np.sum(s_sh <= obs) + 1) / (len(s_sh) + 1),
                     "p(위상보다 작음)": (np.sum(s_ph <= obs) + 1) / (len(s_ph) + 1)})
    d2[name] = pd.DataFrame(rows).set_index("m")
    d2[name].to_csv(RES / f"d2_{'monthly' if '월간' in name else 'daily'}.csv")
    print(name); display(d2[name].round(3))'''),
        md("## 2. 최대 리아프노프 지수 (Rosenstein)"),
        code(r'''L = C["lyap"]
stat = lambda x: lyapunov_rosenstein(x, m=L["m"], tau=L["tau"], max_t=L["max_t"], fit_range=tuple(L["fit_range"]))["lyap"]
rng = np.random.default_rng(C["seed"] + 1)
ly_rows, curves = [], {}
for name, x in SERIES.items():
    obs = lyapunov_rosenstein(x, m=L["m"], tau=L["tau"], max_t=L["max_t"], fit_range=tuple(L["fit_range"]))
    curves[name] = obs["log_div"]
    for kind, gen in [("섞기", shuffle_surrogate), ("위상", phase_surrogate)]:
        s = np.array([stat(gen(x, rng)) for _ in range(C["n_surrogates"])])
        ly_rows.append({"계열": name, "대리계열": kind, "실제 λ": obs["lyap"], "대리 평균": s.mean(), "대리 최소": s.min(),
                        "대리 최대": s.max(), "p(대리보다 큼)": (np.sum(s >= obs["lyap"]) + 1) / (len(s) + 1),
                        "p(대리보다 작음)": (np.sum(s <= obs["lyap"]) + 1) / (len(s) + 1)})
ly = pd.DataFrame(ly_rows)
ly.to_csv(RES / "lyapunov.csv", index=False)
ly.round(4)'''),
        md("## 3. 요약표"),
        code(r'''lines = ["| 계열 | N | D2(m=10) 실제 | 섞기 평균 | 위상 평균 | p(위상보다 작음) | λ 실제 | λ 위상 평균 | p(λ가 위상보다 큼) |", "|---|---|---|---|---|---|---|---|---|"]
for name, x in SERIES.items():
    r = d2[name].loc[m_values[-1]]
    l = ly[(ly["계열"] == name) & (ly["대리계열"] == "위상")].iloc[0]
    lines.append(f"| {name} | {len(x)} | {r['실제']:.2f} | {r['섞기 평균']:.2f} | {r['위상 평균']:.2f} | {r['p(위상보다 작음)']:.2f} | "
                 f"{l['실제 λ']:.4f} | {l['대리 평균']:.4f} | {l['p(대리보다 큼)']:.2f} |")
md_t = "\n".join(lines)
(RES / "chaos_summary.md").write_text(md_t + "\n")
print(md_t)'''),
        md("## 4. 그림"),
        code(r'''fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, t) in zip(axes[:2], d2.items()):
    ax.plot(t.index, t["실제"], "o-", color="crimson", label="실제")
    ax.plot(t.index, t["섞기 평균"], "s--", color="gray", label="섞기 평균")
    ax.plot(t.index, t["위상 평균"], "^--", color="steelblue", label="위상 무작위화 평균")
    ax.plot(t.index, t.index, ":", color="k", lw=1, label="D2 = m")
    ax.set_title(f"상관차원 D2(m) · {name}"); ax.set_xlabel("임베딩 차원 m"); ax.legend(fontsize=8); ax.grid(alpha=.3)
for name, c in curves.items():
    axes[2].plot(np.arange(len(c)), c, label=name)
lo, hi = CFG["chaos"]["lyap"]["fit_range"]
axes[2].axvspan(lo, hi, color="gold", alpha=.2, label="기울기 적합 구간")
axes[2].set_title("Rosenstein 평균 로그 발산"); axes[2].set_xlabel("t (스텝)"); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(RES / "figures/chaos_overview.png", dpi=150); plt.show()'''),
    ]


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for name, cells in [("01_hurst.ipynb", hurst_nb()), ("02_chaos.ipynb", chaos_nb())]:
        nb = nbf.v4.new_notebook(); nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, ROOT / "notebooks" / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
