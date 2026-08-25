# -*- coding: utf-8 -*-
"""
deid_eval.py — De-identification baseline fair-scoring harness
System A/B와 동일한 Strict/Lenient 매칭·micro P/R/F1을 재현한다.

[span 표현] (doc_id, start, end, label)  — end는 파이썬 슬라이스 관례(exclusive)
[Strict]  gold와 pred의 (start,end,label) 3요소 모두 일치 → TP
[Lenient] (start,end)만 일치(label 무시) → TP  ← System B 'span-only'와 동일
micro: 모든 라벨·문서를 합산해 TP/FP/FN → P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)
"""
from collections import defaultdict

def _index(spans, strict=True):
    """spans: list of (doc_id,start,end,label). strict면 label 포함 키."""
    d = defaultdict(int)
    for doc_id, s, e, lab in spans:
        key = (doc_id, s, e, lab) if strict else (doc_id, s, e)
        d[key] += 1
    return d

def score(gold_spans, pred_spans, strict=True, per_label=False):
    """
    gold_spans, pred_spans: list of (doc_id,start,end,label)
    return: dict(P,R,F1,TP,FP,FN) 또는 per_label일 때 라벨별 dict 추가
    멀티셋 매칭(같은 위치 중복 span도 카운트).
    """
    g = _index(gold_spans, strict)
    p = _index(pred_spans, strict)
    TP = FP = FN = 0
    keys = set(g) | set(p)
    # 라벨별 집계(strict 키에서 label 추출; lenient는 gold label로 귀속)
    labstat = defaultdict(lambda: [0,0,0])  # TP,FP,FN
    # gold label 조회용(lenient에서 FP 라벨 귀속 위해 pred label도 필요)
    for k in keys:
        gc, pc = g.get(k,0), p.get(k,0)
        tp = min(gc,pc); fp = max(pc-gc,0); fn = max(gc-pc,0)
        TP += tp; FP += fp; FN += fn
        if per_label:
            lab = k[3] if strict else None
            if lab is not None:
                labstat[lab][0]+=tp; labstat[lab][1]+=fp; labstat[lab][2]+=fn
    def prf(tp,fp,fn):
        P = tp/(tp+fp) if tp+fp else 0.0
        R = tp/(tp+fn) if tp+fn else 0.0
        F = 2*P*R/(P+R) if P+R else 0.0
        return P,R,F
    P,R,F = prf(TP,FP,FN)
    out = {"P":round(P,4),"R":round(R,4),"F1":round(F,4),"TP":TP,"FP":FP,"FN":FN}
    if per_label:
        pl={}
        for lab,(tp,fp,fn) in sorted(labstat.items()):
            p_,r_,f_=prf(tp,fp,fn)
            pl[lab]={"P":round(p_,4),"R":round(r_,4),"F1":round(f_,4),"TP":tp,"FP":fp,"FN":fn}
        out["per_label"]=pl
    return out

def score_both(gold_spans, pred_spans, per_label=True):
    return {"Strict":score(gold_spans,pred_spans,strict=True,per_label=per_label),
            "Lenient":score(gold_spans,pred_spans,strict=False,per_label=False)}

def zero_missed_pii_rate(gold_spans, pred_spans, pii_labels=None):
    """문서단위 zero-missed-PII: 그 문서의 gold span을 모두 잡은(FN=0) 문서 비율.
    pii_labels 지정 시 해당 라벨만 대상."""
    g=defaultdict(set); p=defaultdict(set)
    for doc,s,e,lab in gold_spans:
        if pii_labels is None or lab in pii_labels: g[doc].add((s,e,lab))
    for doc,s,e,lab in pred_spans:
        p[doc].add((s,e,lab))
    docs=set(g)|set(p); safe=0
    for doc in docs:
        missed=[sp for sp in g[doc] if (sp[0],sp[1],sp[2]) not in {(a,b,c) for a,b,c in p[doc]}]
        # lenient 관점 미스: 위치라도 잡으면 안전으로 볼지 결정 가능 — 여기선 strict FN=0
        if not missed: safe+=1
    return round(safe/len(docs),4) if docs else 0.0, len(docs), safe
