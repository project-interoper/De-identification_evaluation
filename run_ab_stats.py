# -*- coding: utf-8 -*-
"""
run_ab_stats.py: Program A vs Program B 비식별화 성능의 통계 비교 재현 스크립트.

무엇을 계산하나:
  문서별 TP/FP/FN(strict, micro)에서
  (1) 전체 코퍼스 micro P/R/F1 (A, B 각각)
  (2) 부트스트랩 95% 신뢰구간 (F1 차이 B - A)
  (3) Wilcoxon 부호순위 검정 (문서별 P, R, F1의 A vs B)
  (4) McNemar 검정 (문서 단위 '놓친 PII 0건' 안전성 A vs B)

데이터 출처(각 벤더 폴더의 prelim_scores '문서별' 시트):
  주의: '문서별' 시트의 A_/B_ 열은 벤더가 아니라 '같은 벤더의 방식A(Strict)/방식B(Lenient) 채점'이다.
        따라서 Program A = vendor-A file, A_ columns (Strict); Program B = vendor-B file, A_ columns (Strict).
        두 파일을 doc_id로 정렬해 짝지어(paired) 비교한다.

실행:
python run_ab_stats.py --a_xlsx "vendorA_admission/prelim_scores.xlsx" --b_xlsx "vendorB_admission/prelim_scores.xlsx" --record 입원 --out stats_AB_입원.json
python run_ab_stats.py --a_xlsx "vendorA_discharge/prelim_scores.xlsx" --b_xlsx "vendorB_discharge/prelim_scores.xlsx" --record 퇴원 --out stats_AB_퇴원.json
"""
import argparse, json
import numpy as np
import openpyxl
from scipy import stats

def to_num(x):
    try: return float(x)
    except: return None

def load_strict_perdoc(path):
    """'문서별' 시트에서 방식A(Strict) 열(A_TP/A_FP/A_FN)만 doc_id별로 로드."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["문서별"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]; idx = {h: i for i, h in enumerate(hdr)}
    need = ["doc_id", "A_TP", "A_FP", "A_FN", "gold_span수", "놓친PII_0(A)"]
    for n in need[:4]:
        if n not in idx: raise KeyError(f"'{n}' 열이 없음: {path}")
    out = {}
    for r in rows[1:]:
        did = r[idx["doc_id"]]
        if did is None: continue
        tp = to_num(r[idx["A_TP"]]); fp = to_num(r[idx["A_FP"]]); fn = to_num(r[idx["A_FN"]])
        if tp is None or fp is None or fn is None: continue
        gold = to_num(r[idx["gold_span수"]]) if "gold_span수" in idx else (tp + fn)
        miss0 = r[idx["놓친PII_0(A)"]] if "놓친PII_0(A)" in idx else None
        out[str(did)] = dict(TP=tp, FP=fp, FN=fn, gold=gold,
                             miss0=(1 if str(miss0).strip() in ("예", "Y", "1", "True") else 0))
    return out

def micro(lst):
    TP = sum(x["TP"] for x in lst); FP = sum(x["FP"] for x in lst); FN = sum(x["FN"] for x in lst)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(P=round(P, 4), R=round(R, 4), F1=round(F, 4), TP=int(TP), FP=int(FP), FN=int(FN))

def doc_prf(d):
    P = d["TP"] / (d["TP"] + d["FP"]) if d["TP"] + d["FP"] else (1.0 if d["gold"] == 0 else 0.0)
    R = d["TP"] / (d["TP"] + d["FN"]) if d["TP"] + d["FN"] else (1.0 if d["gold"] == 0 else 0.0)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F

def bootstrap_ci_f1diff(A, B, ids, n_boot=5000, seed=42):
    """문서 재표집 기반 micro-F1 차이(B - A)의 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(ids); idxarr = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(idxarr, size=n, replace=True)
        a = micro([A[ids[i]] for i in samp]); b = micro([B[ids[i]] for i in samp])
        diffs.append(b["F1"] - a["F1"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(np.mean(diffs)), 4), round(float(lo), 4), round(float(hi), 4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_xlsx", required=True, help="Program A prelim_scores.xlsx")
    ap.add_argument("--b_xlsx", required=True, help="Program B prelim_scores.xlsx")
    ap.add_argument("--record", default="입원")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="stats_AB.json")
    args = ap.parse_args()

    A = load_strict_perdoc(args.a_xlsx)   # Program A의 Strict 문서별
    B = load_strict_perdoc(args.b_xlsx)   # Program B의 Strict 문서별
    ids = sorted(set(A) & set(B))
    print(f"[{args.record}] 공통 문서 {len(ids)}건 (A {len(A)} / B {len(B)})")

    Al = [A[i] for i in ids]; Bl = [B[i] for i in ids]
    mA = micro(Al); mB = micro(Bl)
    print(f"  Program A micro : P={mA['P']} R={mA['R']} F1={mA['F1']}")
    print(f"  Program B micro: P={mB['P']} R={mB['R']} F1={mB['F1']}")

    # 문서별 P/R/F1
    pA = np.array([doc_prf(A[i]) for i in ids]); pB = np.array([doc_prf(B[i]) for i in ids])
    wil = {}
    for k, j in [("P", 0), ("R", 1), ("F1", 2)]:
        try:
            s, p = stats.wilcoxon(pB[:, j], pA[:, j], zero_method="wilcox")
            wil[k] = dict(stat=round(float(s), 2), p=float(p),
                          median_diff=round(float(np.median(pB[:, j] - pA[:, j])), 4))
        except Exception as e:
            wil[k] = dict(error=str(e))
    print(f"  Wilcoxon(F1): p={wil['F1'].get('p'):.2e}" if 'p' in wil['F1'] else "  Wilcoxon(F1): n/a")

    # 부트스트랩 F1 차이 CI
    mean_d, lo, hi = bootstrap_ci_f1diff(A, B, ids, n_boot=args.n_boot, seed=args.seed)
    print(f"  Bootstrap ΔF1(B-A): {mean_d}  95% CI [{lo}, {hi}]")

    # McNemar (문서 안전성: 놓친 PII 0건 여부 A vs B): 불일치쌍만 사용
    b01 = sum(1 for i in ids if A[i]["miss0"] == 0 and B[i]["miss0"] == 1)  # A실패,B성공
    b10 = sum(1 for i in ids if A[i]["miss0"] == 1 and B[i]["miss0"] == 0)  # A성공,B실패
    if b01 + b10 > 0:
        mc = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)   # 연속성 보정
        mc_p = float(stats.chi2.sf(mc, 1))
    else:
        mc, mc_p = 0.0, 1.0
    safeA = sum(A[i]["miss0"] for i in ids) / len(ids)
    safeB = sum(B[i]["miss0"] for i in ids) / len(ids)
    print(f"  McNemar(놓친PII0): b01={b01} b10={b10} chi2={mc:.3f} p={mc_p:.3g} | 안전율 A={safeA:.3f} B={safeB:.3f}")

    out = dict(record=args.record, n_docs=len(ids),
               micro=dict(A=mA, B=mB, F1_diff=round(mB["F1"] - mA["F1"], 4),
                          P_diff=round(mB["P"] - mA["P"], 4), R_diff=round(mB["R"] - mA["R"], 4)),
               wilcoxon=wil,
               bootstrap_F1diff=dict(mean=mean_d, ci_low=lo, ci_high=hi, n_boot=args.n_boot),
               mcnemar_missedPII0=dict(b01_Afail_Bok=b01, b10_Aok_Bfail=b10, chi2=round(mc, 3), p=mc_p,
                                       safe_rate_A=round(safeA, 4), safe_rate_B=round(safeB, 4)))
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
