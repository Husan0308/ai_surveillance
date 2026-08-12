#!/usr/bin/env python3
"""Operator-only Global-ID ground-truth scorer.

Input is JSON Lines. Each line identifies a physically observed consenting test
subject and the UI metadata visible at that moment:
{"subject":"PERSON_A","camera_id":"CAM-01","local_track_id":"...","global_id":"UNK 9"}

This tool never writes to or influences production identity state.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict


def evaluate(records):
    subject_ids=defaultdict(set);global_subjects=defaultdict(set);examples=[]
    for raw in records:
        subject=str(raw.get("subject") or "").strip();global_id=str(raw.get("global_id") or "").strip()
        if not subject or not global_id:continue
        subject_ids[subject].add(global_id);global_subjects[global_id].add(subject);examples.append(dict(raw))
    false_splits={key:sorted(value) for key,value in subject_ids.items() if len(value)>1}
    false_merges={key:sorted(value) for key,value in global_subjects.items() if len(value)>1}
    transitions=sum(max(0,len({item.get("camera_id") for item in examples if item.get("subject")==subject})-1) for subject in subject_ids)
    return {"observations":len(examples),"subjects":len(subject_ids),"camera_transitions":transitions,
            "same_person_reuse_accuracy":None if not subject_ids else sum(len(ids)==1 for ids in subject_ids.values())/len(subject_ids),
            "false_splits":false_splits,"false_merges":false_merges,"pass":not false_splits and not false_merges}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",help="JSONL file; stdin when omitted");args=parser.parse_args()
    stream=open(args.input,encoding="utf-8") if args.input else sys.stdin
    try:records=[json.loads(line) for line in stream if line.strip()]
    finally:
        if args.input:stream.close()
    print(json.dumps(evaluate(records),indent=2,sort_keys=True))


if __name__=="__main__":main()
