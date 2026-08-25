# -*- coding: utf-8 -*-
"""
run_gliner_zeroshot.py — 참조 baseline: GLiNER zero-shot(무학습) de-id.
공정성: 상용 A/B와 동일하게 '학습 없이 적용'. 전체 문서에서 평가(분할 불필요).

임계값 스윕(threshold sweep): 한 번의 추론으로 여러 임계값을 모두 평가한다.
GLiNER 예측에는 각 탐지의 신뢰도(score)가 있으므로, 매우 낮은 기준값으로 한 번
추론해 모든 후보를 모은 뒤, 여러 임계값에서 필터링만 해 P/R/F1 곡선을 만든다.

실행(권장 — 스윕):
  python run_gliner_zeroshot.py --gold_dir data\gold_입원 --record 입원 \
      --sweep 0.3,0.4,0.5,0.6,0.7,0.8,0.9 --out results_gliner_입원.json

실행(단일 임계값만):
  python run_gliner_zeroshot.py --gold_dir data\gold_입원 --record 입원 \
      --threshold 0.5 --out results_gliner_입원.json
필요:
  gold_loader가 표준 포맷 반환. LABEL_PROMPTS(라벨→자연어 프롬프트)를 도메인에 맞게 조정.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_loader import load_gold_set
from deid_eval import score_both, zero_missed_pii_rate

# PHI label -> GLiNER natural-language prompt (zero-shot needs labels phrased as text).
# Predicted label names are mapped back to the PHI category codes.
LABEL_PROMPTS = {
    "person name":"NAME", "patient name":"NAME",
    "date":"DAY", "birth date":"B_DAY",
    "hospital name":"HOSP_NAME", "medical institution":"HOSP_NAME",
    "address":"ADD_RES",
    "patient id":"PAT_ID", "medical record number":"PAT_ID",
    "phone number":"TEL_NUM",
    "resident registration number":"SSN_NUM",
    "organization name":"ETC_NAME", "university":"ETC_NAME",
    "pathology number":"PATH_NUM",
    "rare disease":"RARE_DX", "rare drug":"RARE_DRUG",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold_dir",required=True)
    ap.add_argument("--worklist",default=None)
    ap.add_argument("--record",default="입원")
    ap.add_argument("--model",default="urchade/gliner_multi-v2.1")
    ap.add_argument("--threshold",type=float,default=0.5,help="단일 임계값(스윕 미사용 시)")
    ap.add_argument("--sweep",default=None,help="쉼표구분 임계값 목록. 예: 0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--base_threshold",type=float,default=0.2,
                    help="스윕 시 후보 수집용 최저 기준값(이 값 미만은 아예 안 뽑음). 스윕 최소값보다 낮게.")
    ap.add_argument("--max_len",type=int,default=384)
    ap.add_argument("--out",default="results_gliner.json")
    args=ap.parse_args()

    # 스윕 임계값 결정
    if args.sweep:
        sweep=[float(x) for x in args.sweep.split(",")]
        collect_th=min(args.base_threshold, min(sweep))  # 후보는 가장 낮은 값으로 수집
    else:
        sweep=[args.threshold]
        collect_th=args.threshold

    from gliner import GLiNER
    import torch
    model=GLiNER.from_pretrained(args.model)
    if torch.cuda.is_available(): model=model.to("cuda")
    prompts=list(LABEL_PROMPTS.keys())

    docs=load_gold_set(args.gold_dir,args.worklist)
    print(f"[data] {len(docs)} documents  |  수집 기준값 {collect_th}  |  스윕 {sweep}")

    gold_spans=[]
    # 예측 후보를 (did,s,e,lab,score)로 한 번만 모음
    cand=[]
    for di,(did,d) in enumerate(docs.items()):
        text=d["text"]
        for s,e,l in d["spans"]: gold_spans.append((did,s,e,l))
        step=args.max_len*3  # 대략 문자수
        for base in range(0,len(text),step):
            chunk=text[base:base+step]
            if not chunk.strip(): continue
            ents=model.predict_entities(chunk,prompts,threshold=collect_th)
            for en in ents:
                lab=LABEL_PROMPTS.get(en["label"])
                if not lab: continue
                cand.append((did, base+en["start"], base+en["end"], lab, float(en.get("score",1.0))))
        if (di+1)%100==0: print(f"  ...{di+1}/{len(docs)} 문서 추론")

    print(f"[cand] 후보 탐지 {len(cand)}건 (기준값 {collect_th} 이상)")

    # 각 임계값에서 필터링해 채점
    sweep_results={}
    for th in sweep:
        pred=[(did,s,e,lab) for (did,s,e,lab,sc) in cand if sc>=th]
        res=score_both(gold_spans,pred,per_label=True)
        zr,ndoc,nsafe=zero_missed_pii_rate(gold_spans,pred)
        sweep_results[f"{th:.2f}"]={
            "threshold":th,"n_pred":len(pred),
            "scores":res,"zero_missed_PII":{"rate":zr,"n_docs":ndoc,"n_safe":nsafe}}
        print(f"  th={th:.2f}  n_pred={len(pred):5d}  "
              f"Strict P={res['Strict']['P']:.4f} R={res['Strict']['R']:.4f} F1={res['Strict']['F1']:.4f}  "
              f"zeroPII={zr:.4f}")

    # 최고 F1 임계값
    best_th=max(sweep, key=lambda t: sweep_results[f"{t:.2f}"]["scores"]["Strict"]["F1"])
    best=sweep_results[f"{best_th:.2f}"]

    out={"baseline":"GLiNER-zeroshot","model":args.model,"record":args.record,
         "n_docs":len(docs),"collect_threshold":collect_th,
         "sweep":sweep_results,
         "best_by_strict_F1":{"threshold":best_th,
             "Strict_F1":best["scores"]["Strict"]["F1"],
             "Strict_P":best["scores"]["Strict"]["P"],
             "Strict_R":best["scores"]["Strict"]["R"],
             "zero_missed_PII":best["zero_missed_PII"]["rate"]},
         # 하위호환: 단일 threshold=0.5 결과도 최상위에 노출(있으면)
         "threshold":sweep[0] if len(sweep)==1 else None,
         "scores":sweep_results.get("0.50",sweep_results[f"{sweep[0]:.2f}"])["scores"] if "0.50" in sweep_results or len(sweep)==1 else best["scores"]}
    json.dump(out,open(args.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[BEST] threshold={best_th:.2f}  Strict F1={best['scores']['Strict']['F1']:.4f}  "
          f"(P={best['scores']['Strict']['P']:.4f} R={best['scores']['Strict']['R']:.4f})  "
          f"zero-missed-PII={best['zero_missed_PII']['rate']:.4f}")
    print(f"[saved] {args.out}")

if __name__=="__main__":
    main()
