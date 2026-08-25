# -*- coding: utf-8 -*-
"""
run_klue_bert.py — 주력 baseline: KLUE-BERT fine-tune de-id NER (긴 문서 sliding-window 지원).
공정성: 문서 단위 train/test 분할 → test에서만 평가. (★상용 A/B도 같은 test 문서로 재채점)

v2 변경: 문서가 max_len 토큰을 넘으면 stride로 겹치는 창(window)을 만들어 전부 학습/예측.
         (이전 버전은 512에서 잘려 긴 문서의 뒷부분 gold를 통째로 놓쳤음)

실행:
  python run_klue_bert.py --gold_dir GOLD_DIR --record 입원 \
      --model klue/bert-base --epochs 5 --test_ratio 0.2 --seed 42 --out results_klue_입원.json
옵션:
  --max_len 512 --stride 128   (창 크기/겹침. 긴 문서면 그대로 두면 됨)
"""
import argparse, json, os, sys, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_loader import load_gold_set
from deid_eval import score_both, zero_missed_pii_rate

def set_seed(s):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_windows(text, spans, tok, max_len, stride, l2i, train=True):
    """
    긴 문서를 sliding window로 나눠 (input_ids, attention_mask, labels[, offset, win_start]) 목록 반환.
    return_overflowing_tokens로 창을 만들고, 각 창의 offset_mapping으로 char-span→BIO.
    """
    enc = tok(text, return_offsets_mapping=True, truncation=True,
              max_length=max_len, stride=stride,
              return_overflowing_tokens=True, padding=False)
    spans_sorted = sorted(spans, key=lambda x:(x[0],x[1]))
    out=[]
    n_win=len(enc["input_ids"])
    for w in range(n_win):
        offsets=enc["offset_mapping"][w]
        bio=["O"]*len(offsets)
        for s,e,lab in spans_sorted:
            started=False
            for i,(a,b) in enumerate(offsets):
                if a==b: continue
                if b<=s or a>=e: continue
                bio[i]=("B-" if not started else "I-")+lab
                started=True
        item={"input_ids":enc["input_ids"][w],
              "attention_mask":enc["attention_mask"][w],
              "labels":[l2i.get(x,0) for x in bio]}
        if not train:
            item["offsets"]=offsets
        out.append(item)
    return out

def bio_to_spans_from_windows(doc_id, windows_offsets, windows_tags):
    """여러 창의 (offsets,tags)를 합쳐 char-span 추출 후 중복 제거."""
    raw=[]
    for offsets,tags in zip(windows_offsets,windows_tags):
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
    # 겹치는 창에서 같은 span이 중복되므로 set으로 정리
    uniq=set((s,e,l) for s,e,l in raw)
    return [(doc_id,s,e,l) for s,e,l in uniq]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold_dir",required=True)
    ap.add_argument("--record",default="입원")
    ap.add_argument("--model",default="klue/bert-base")
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--lr",type=float,default=3e-5)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--max_len",type=int,default=512)
    ap.add_argument("--stride",type=int,default=128)
    ap.add_argument("--test_ratio",type=float,default=0.2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="results_klue.json")
    args=ap.parse_args()
    set_seed(args.seed)

    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              TrainingArguments, Trainer, DataCollatorForTokenClassification)
    from datasets import Dataset

    docs=load_gold_set(args.gold_dir)
    doc_ids=sorted(docs.keys())
    print(f"[data] {len(doc_ids)} documents loaded")

    labset=set()
    for d in docs.values():
        for _,_,lab in d["spans"]: labset.add(lab)
    labels_list=["O"]+[f"{p}-{l}" for l in sorted(labset) for p in ("B","I")]
    l2i={l:i for i,l in enumerate(labels_list)}; i2l={i:l for l,i in l2i.items()}
    print(f"[labels] {len(labset)} entity types: {sorted(labset)}")

    tok=AutoTokenizer.from_pretrained(args.model)

    rng=random.Random(args.seed); ids=doc_ids[:]; rng.shuffle(ids)
    n_test=int(len(ids)*args.test_ratio)
    test_ids=set(ids[:n_test]); train_ids=[d for d in ids if d not in test_ids]
    print(f"[split] train {len(train_ids)} / test {len(test_ids)} docs")

    # 학습 데이터: 모든 문서를 창으로 펼침
    train_rows={"input_ids":[],"attention_mask":[],"labels":[]}
    n_win=0
    for did in train_ids:
        for it in make_windows(docs[did]["text"],docs[did]["spans"],tok,args.max_len,args.stride,l2i,train=True):
            train_rows["input_ids"].append(it["input_ids"])
            train_rows["attention_mask"].append(it["attention_mask"])
            train_rows["labels"].append(it["labels"]); n_win+=1
    train_ds=Dataset.from_dict(train_rows)
    print(f"[train] {len(train_ids)} docs -> {n_win} windows (max_len={args.max_len}, stride={args.stride})")

    model=AutoModelForTokenClassification.from_pretrained(
        args.model,num_labels=len(labels_list),id2label=i2l,label2id=l2i)
    coll=DataCollatorForTokenClassification(tok)
    targs=TrainingArguments(output_dir="./_klue_ckpt",num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,learning_rate=args.lr,
        logging_steps=50,save_strategy="no",report_to=[],seed=args.seed,fp16=torch.cuda.is_available())
    try:
        trainer=Trainer(model=model,args=targs,train_dataset=train_ds,data_collator=coll,processing_class=tok)
    except TypeError:
        trainer=Trainer(model=model,args=targs,train_dataset=train_ds,data_collator=coll,tokenizer=tok)
    trainer.train()

    # 예측: test 문서를 창으로 나눠 각 창 예측 후 span 병합
    model.eval(); dev=next(model.parameters()).device
    gold_spans=[]; pred_spans=[]
    for did in test_ids:
        text=docs[did]["text"]
        for s,e,l in docs[did]["spans"]: gold_spans.append((did,s,e,l))
        wins=make_windows(text,docs[did]["spans"],tok,args.max_len,args.stride,l2i,train=False)
        w_offsets=[]; w_tags=[]
        for it in wins:
            ii=torch.tensor([it["input_ids"]]).to(dev)
            am=torch.tensor([it["attention_mask"]]).to(dev)
            with torch.no_grad():
                logits=model(input_ids=ii,attention_mask=am).logits[0]
            tags=[i2l[int(x)] for x in logits.argmax(-1).cpu()]
            w_offsets.append(it["offsets"]); w_tags.append(tags)
        pred_spans+=bio_to_spans_from_windows(did,w_offsets,w_tags)

    res=score_both(gold_spans,pred_spans,per_label=True)
    zr,ndoc,nsafe=zero_missed_pii_rate(gold_spans,pred_spans)
    out={"baseline":"KLUE-BERT-finetune","model":args.model,"record":args.record,
         "seed":args.seed,"test_ratio":args.test_ratio,"max_len":args.max_len,"stride":args.stride,
         "n_train":len(train_ids),"n_test":len(test_ids),
         "test_doc_ids":sorted(test_ids),
         "scores":res,"zero_missed_PII":{"rate":zr,"n_docs":ndoc,"n_safe":nsafe}}
    json.dump(out,open(args.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n[RESULT] Strict F1={res['Strict']['F1']}  P={res['Strict']['P']}  R={res['Strict']['R']}  Lenient F1={res['Lenient']['F1']}  zero-missed-PII={zr}")
    print(f"[saved] {args.out}")

if __name__=="__main__":
    main()
