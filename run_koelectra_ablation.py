# -*- coding: utf-8 -*-
"""
run_koelectra_ablation.py: KoELECTRA 학습 전후(zero-shot vs fine-tuned) 동일 test 비교.

리뷰어 대응: "무학습과 학습에 서로 다른 모델을 썼다"는 지적을 없애기 위해,
같은 KoELECTRA 체크포인트를 (a) zero-shot 과 (b) fine-tuned 로 '동일 test 문서'에서 비교한다.

기존 run_klue_bert.py 와 같은 유틸(gold_loader, deid_eval)을 그대로 재사용하고,
같은 --seed 42 --test_ratio 0.2 를 주면 KLUE-BERT fine-tuned 와도 동일 test 분할이 재현된다.

실행:
python run_koelectra_ablation.py --gold_dir data\gold_입원 --record 입원 --model Leo97/KoELECTRA-small-v3-modu-ner --epochs 5 --test_ratio 0.2 --seed 42 --out results_koelectra_ablation_입원.json
python run_koelectra_ablation.py --gold_dir data\gold_퇴원 --record 퇴원 --model Leo97/KoELECTRA-small-v3-modu-ner --epochs 5 --test_ratio 0.2 --seed 42 --out results_koelectra_ablation_퇴원.json
"""
import argparse, json, os, sys, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_loader import load_gold_set
from deid_eval import score_both, zero_missed_pii_rate

# KoELECTRA(MODU NER) 태그 -> 우리 PHI 라벨 매핑 (기존 zero-shot 결과와 동일)
MODU_TO_PHI = {"PS": "NAME", "OG": "HOSP_NAME", "LC": "ADD_RES", "DT": "DAY"}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_windows(text, spans, tok, max_len, stride, l2i, train=True):
    enc = tok(text, return_offsets_mapping=True, truncation=True,
              max_length=max_len, stride=stride,
              return_overflowing_tokens=True, padding=False)
    spans_sorted = sorted(spans, key=lambda x:(x[0],x[1]))
    out=[]
    for w in range(len(enc["input_ids"])):
        offsets=enc["offset_mapping"][w]
        bio=["O"]*len(offsets)
        for s,e,lab in spans_sorted:
            started=False
            for i,(a,b) in enumerate(offsets):
                if a==b: continue
                if b<=s or a>=e: continue
                bio[i]=("B-" if not started else "I-")+lab; started=True
        item={"input_ids":enc["input_ids"][w],"attention_mask":enc["attention_mask"][w],
              "labels":[l2i.get(x,0) for x in bio]}
        if not train: item["offsets"]=offsets
        out.append(item)
    return out

def windows_for_infer(text, tok, max_len, stride):
    """예측 전용: gold 없이 창만 생성 (offsets 포함)."""
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len,
              stride=stride, return_overflowing_tokens=True, padding=False)
    return [{"input_ids":enc["input_ids"][w],"attention_mask":enc["attention_mask"][w],
             "offsets":enc["offset_mapping"][w]} for w in range(len(enc["input_ids"]))]

def merge_spans(doc_id, w_offsets, w_tags):
    raw=[]
    for offsets,tags in zip(w_offsets,w_tags):
        cur=None
        for (a,b),t in zip(offsets,tags):
            if a==b:
                if cur: raw.append(cur); cur=None
                continue
            if t.startswith("B-"):
                if cur: raw.append(cur)
                cur=[a,b,t[2:]]
            elif t.startswith("I-") and cur and t[2:]==cur[2]:
                cur[1]=b
            else:
                if cur: raw.append(cur); cur=None
        if cur: raw.append(cur); cur=None
    uniq=set((s,e,l) for s,e,l in raw)
    return [(doc_id,s,e,l) for s,e,l in uniq]

def predict(model, i2l, tok, docs, test_ids, max_len, stride, dev, torch):
    """fine-tuned 모델로 test 문서 예측 -> pred_spans (우리 라벨 공간)."""
    model.eval(); preds=[]
    for did in test_ids:
        wins=windows_for_infer(docs[did]["text"], tok, max_len, stride)
        w_off=[]; w_tag=[]
        for it in wins:
            ii=torch.tensor([it["input_ids"]]).to(dev)
            am=torch.tensor([it["attention_mask"]]).to(dev)
            with torch.no_grad():
                logits=model(input_ids=ii,attention_mask=am).logits[0]
            tags=[i2l[int(x)] for x in logits.argmax(-1).cpu()]
            w_off.append(it["offsets"]); w_tag.append(tags)
        preds+=merge_spans(did,w_off,w_tag)
    return preds

def predict_zeroshot(tok, docs, test_ids, max_len, stride, dev, torch, model_name):
    """원본 KoELECTRA NER 헤드 그대로 로드 -> MODU 태그 예측 -> PHI 매핑."""
    from transformers import AutoModelForTokenClassification
    zmodel=AutoModelForTokenClassification.from_pretrained(model_name).to(dev)
    zmodel.eval()
    zid2label=zmodel.config.id2label   # MODU 태그 (예: B-PS, I-LC ...)
    preds=[]
    for did in test_ids:
        wins=windows_for_infer(docs[did]["text"], tok, max_len, stride)
        w_off=[]; w_tag=[]
        for it in wins:
            ii=torch.tensor([it["input_ids"]]).to(dev)
            am=torch.tensor([it["attention_mask"]]).to(dev)
            with torch.no_grad():
                logits=zmodel(input_ids=ii,attention_mask=am).logits[0]
            raw=[zid2label[int(x)] for x in logits.argmax(-1).cpu()]
            mapped=[]
            for t in raw:
                if t in ("O","",None): mapped.append("O"); continue
                pref,_,tag=t.partition("-")
                phi=MODU_TO_PHI.get(tag)
                mapped.append(f"{pref}-{phi}" if phi and pref in ("B","I") else "O")
            w_off.append(it["offsets"]); w_tag.append(mapped)
        preds+=merge_spans(did,w_off,w_tag)
    return preds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold_dir",required=True)
    ap.add_argument("--record",default="입원")
    ap.add_argument("--model",default="Leo97/KoELECTRA-small-v3-modu-ner")
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--max_len",type=int,default=512)
    ap.add_argument("--stride",type=int,default=128)
    ap.add_argument("--test_ratio",type=float,default=0.2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="results_koelectra_ablation.json")
    args=ap.parse_args()
    set_seed(args.seed)

    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              TrainingArguments, Trainer, DataCollatorForTokenClassification)
    from datasets import Dataset

    docs=load_gold_set(args.gold_dir); doc_ids=sorted(docs.keys())
    print(f"[data] {len(doc_ids)} documents")

    labset=set()
    for d in docs.values():
        for _,_,lab in d["spans"]: labset.add(lab)
    labels_list=["O"]+[f"{p}-{l}" for l in sorted(labset) for p in ("B","I")]
    l2i={l:i for i,l in enumerate(labels_list)}; i2l={i:l for l,i in l2i.items()}

    tok=AutoTokenizer.from_pretrained(args.model)
    rng=random.Random(args.seed); ids=doc_ids[:]; rng.shuffle(ids)
    n_test=int(len(ids)*args.test_ratio)
    test_ids=sorted(set(ids[:n_test])); train_ids=[d for d in ids if d not in set(test_ids)]
    print(f"[split] train {len(train_ids)} / test {len(test_ids)} (seed={args.seed}, ratio={args.test_ratio})")

    dev="cuda" if torch.cuda.is_available() else "cpu"
    gold_spans=[]
    for did in test_ids:
        for s,e,l in docs[did]["spans"]: gold_spans.append((did,s,e,l))

    # ---------- (A) ZERO-SHOT (같은 test 문서) ----------
    print("[zero-shot] KoELECTRA 원본 NER 헤드로 예측...")
    zs_pred=predict_zeroshot(tok, docs, test_ids, args.max_len, args.stride, dev, torch, args.model)
    zs=score_both(gold_spans, zs_pred, per_label=True)
    zs_zr,_,_=zero_missed_pii_rate(gold_spans, zs_pred)
    print(f"  zero-shot Strict F1={zs['Strict']['F1']}  P={zs['Strict']['P']}  R={zs['Strict']['R']}")

    # ---------- (B) FINE-TUNED (동일 train/test) ----------
    # O-collapse 방지: 'O'(index 0) 라벨 가중치를 낮춘 가중 손실 Trainer
    import torch.nn as nn
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels=inputs.pop("labels")
            outputs=model(**inputs)
            logits=outputs.logits
            w=torch.ones(logits.size(-1), device=logits.device); w[0]=0.1  # 'O' 비중 축소
            lf=nn.CrossEntropyLoss(weight=w, ignore_index=-100)
            loss=lf(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    print("[fine-tune] 학습 시작...")
    train_rows={"input_ids":[],"attention_mask":[],"labels":[]}
    for did in train_ids:
        for it in make_windows(docs[did]["text"],docs[did]["spans"],tok,args.max_len,args.stride,l2i,train=True):
            train_rows["input_ids"].append(it["input_ids"])
            train_rows["attention_mask"].append(it["attention_mask"])
            train_rows["labels"].append(it["labels"])
    train_ds=Dataset.from_dict(train_rows)

    model=AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels_list), id2label=i2l, label2id=l2i,
        ignore_mismatched_sizes=True)   # ★ KoELECTRA NER 헤드(15라벨)와 우리 19라벨 불일치 허용
    coll=DataCollatorForTokenClassification(tok)
    targs=TrainingArguments(output_dir="./_koelectra_ckpt",num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,learning_rate=args.lr,
        logging_steps=50,save_strategy="no",report_to=[],seed=args.seed,fp16=torch.cuda.is_available())
    try:
        trainer=WeightedTrainer(model=model,args=targs,train_dataset=train_ds,data_collator=coll,processing_class=tok)
    except TypeError:
        trainer=WeightedTrainer(model=model,args=targs,train_dataset=train_ds,data_collator=coll,tokenizer=tok)
    trainer.train()

    ft_pred=predict(model, i2l, tok, docs, test_ids, args.max_len, args.stride, dev, torch)
    print(f"[diag] fine-tuned가 예측한 span 수: {len(ft_pred)} (0이면 O-collapse = 학습 실패)")
    if len(ft_pred)==0:
        print("[warn] 예측 span이 0입니다. --lr 을 1e-4로 올리거나 --epochs 를 15로 늘려 재실행하세요.")
    ft=score_both(gold_spans, ft_pred, per_label=True)
    ft_zr,ndoc,nsafe=zero_missed_pii_rate(gold_spans, ft_pred)
    print(f"  fine-tuned Strict F1={ft['Strict']['F1']}  P={ft['Strict']['P']}  R={ft['Strict']['R']}")

    out={"model":args.model,"record":args.record,"seed":args.seed,"test_ratio":args.test_ratio,
         "n_train":len(train_ids),"n_test":len(test_ids),"test_doc_ids":test_ids,
         "zero_shot":{"scores":zs,"zero_missed_PII":zs_zr},
         "fine_tuned":{"scores":ft,"zero_missed_PII":{"rate":ft_zr,"n_docs":ndoc,"n_safe":nsafe}},
         "delta_Strict_F1":round(ft["Strict"]["F1"]-zs["Strict"]["F1"],4)}
    json.dump(out,open(args.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[RESULT] {args.record}  zero-shot F1={zs['Strict']['F1']}  ->  fine-tuned F1={ft['Strict']['F1']}  (Δ={out['delta_Strict_F1']})")
    print(f"[saved] {args.out}")

if __name__=="__main__":
    main()
