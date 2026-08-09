#!/usr/bin/env python3
"""Evaluate labelled live face-attempt CSV data without changing runtime thresholds."""
import argparse,csv

def main():
 p=argparse.ArgumentParser();p.add_argument("csv_file");p.add_argument("--threshold",type=float,required=True);args=p.parse_args();rows=list(csv.DictReader(open(args.csv_file,newline="",encoding="utf-8")))
 required={"expected_match","similarity"}
 if not rows or not required.issubset(rows[0]):raise SystemExit("CSV requires expected_match and similarity columns")
 genuine=[float(r["similarity"]) for r in rows if r["expected_match"].strip().lower() in ("1","true","yes")];impostor=[float(r["similarity"]) for r in rows if r["expected_match"].strip().lower() not in ("1","true","yes")]
 tp=sum(x>=args.threshold for x in genuine);fn=len(genuine)-tp;fp=sum(x>=args.threshold for x in impostor);tn=len(impostor)-fp
 print(f"threshold={args.threshold:.4f} genuine={len(genuine)} impostor={len(impostor)}")
 print(f"TAR_recall={tp/max(len(genuine),1):.4f} false_accepts={fp} false_rejects={fn} true_rejects={tn}")
 print("PROVISIONAL" if len(genuine)<50 or len(impostor)<50 else "DATASET_SIZE_ACCEPTABLE_FOR_PROJECT CALIBRATION_ONLY")
if __name__=="__main__":main()
