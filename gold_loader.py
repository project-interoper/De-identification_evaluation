# -*- coding: utf-8 -*-
"""
gold_loader.py — Label Studio(full export) JSON 전용 로더.
원문이 data.text에 통째로 들어있으므로 엑셀 join 불필요.

Label Studio 항목 구조:
  { "id":.., "annotations":[{"result":[
        {"value":{"start":30,"end":32,"text":"19","labels":["DAY"]}, ...}, ...]}],
    "data":{"No":16915, "text":"...원문..."} }
  · start=포함, end=제외 (파이썬 슬라이스와 동일)
  · 여러 파일(4개)을 gold_dir에 두면 모두 읽어 합침.

표준 출력: {doc_id(str): {"text":str, "spans":[(start,end,label)]}}
"""
import json, os, glob

def parse_labelstudio_file(path):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):           # 단일 항목이 dict로 온 경우 방어
        data = [data]
    out = {}
    for item in data:
        d = item.get("data", {})
        # 문서 ID: data.No 우선, 없으면 item id
        doc_id = d.get("No", d.get("no", item.get("id")))
        doc_id = str(doc_id)
        text = d.get("text", "")
        spans = []
        anns = item.get("annotations", []) or []
        # 보통 annotations[0] 사용(단일 어노테이터). ground_truth가 있으면 그것 우선.
        chosen = None
        for a in anns:
            if a.get("ground_truth"):
                chosen = a; break
        if chosen is None and anns:
            chosen = anns[0]
        if chosen:
            for r in chosen.get("result", []):
                v = r.get("value", {})
                s, e = v.get("start"), v.get("end")
                labs = v.get("labels", [])
                if s is None or e is None or not labs:
                    continue
                spans.append((int(s), int(e), labs[0]))
        out[doc_id] = {"text": text, "spans": spans}
    return out

def load_gold_set(gold_dir, worklist_xlsx=None):
    """
    gold_dir 안의 모든 *.json(Label Studio export)을 로드해 합침.
    같은 doc_id가 여러 파일에 있으면 뒤에 온 파일이 덮어씀(보통 파일별 문서가 겹치지 않음).
    worklist_xlsx: 미사용(호환 위해 시그니처만 유지).
    """
    docs = {}
    files = sorted(glob.glob(os.path.join(gold_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"gold_dir에 *.json 없음: {gold_dir}")
    dup = 0
    for fp in files:
        part = parse_labelstudio_file(fp)
        for did, v in part.items():
            if did in docs:
                dup += 1
                # 이미 있으면 span이 더 많은 쪽 유지(빈 어노테이션 방지)
                if len(v["spans"]) <= len(docs[did]["spans"]):
                    continue
            docs[did] = v
    if dup:
        print(f"[gold_loader] 중복 doc_id {dup}건 발견(더 많은 span 쪽 유지)")
    return docs

# 단독 실행 시 자가 점검
if __name__ == "__main__":
    import sys
    gd = sys.argv[1] if len(sys.argv) > 1 else "."
    docs = load_gold_set(gd)
    n_span = sum(len(d["spans"]) for d in docs.values())
    labs = sorted({l for d in docs.values() for _,_,l in d["spans"]})
    print(f"문서 {len(docs)}개 / 총 span {n_span}개")
    print(f"라벨 종류({len(labs)}): {labs}")
    # 오프셋 정합성: 무작위 문서 몇 개에서 text[s:e]가 span의 표기와 맞는지
    import itertools
    checked=mismatch=0
    for did, d in itertools.islice(docs.items(), 20):
        for s,e,lab in d["spans"][:50]:
            checked += 1
            # value.text가 있으면 그것과 비교(파서에선 안 저장했으니 길이만 확인)
            frag = d["text"][s:e]
            if not frag:  # 빈 조각이면 offset 이상
                mismatch += 1
    print(f"오프셋 점검: {checked}개 확인, 빈 조각 {mismatch}개 (0이어야 정상)")
    # 예시 출력
    did = next(iter(docs)); d = docs[did]
    print(f"\n[예시 문서 {did}] 앞 8개 span:")
    for s,e,lab in d["spans"][:8]:
        print(f"  {lab:10s} text[{s}:{e}] = {d['text'][s:e]!r}")
