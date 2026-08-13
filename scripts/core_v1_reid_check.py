#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=3.0) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:8001')
    parser.add_argument('--seconds',type=int,default=60)
    parser.add_argument('--interval',type=float,default=5.0)
    args=parser.parse_args()
    deadline=time.monotonic()+max(1,args.seconds)
    while time.monotonic()<deadline:
        health=get_json(args.base.rstrip('/')+'/health')
        reid=health.get('reid') or {}
        detector=health.get('detector') or {}
        histories=health.get('frame_history') or {}
        print(json.dumps({
            'reid_ready':reid.get('ready'),
            'reid_error':reid.get('last_error'),
            'embedded':reid.get('embedded'),
            'embed_rate':reid.get('embed_rate'),
            'reid_batch_ms':reid.get('last_batch_ms'),
            'queue':reid.get('queue_depth'),
            'frame_misses':reid.get('frame_misses'),
            'identity':reid.get('identity'),
            'detector_batch':detector.get('batch_ms'),
            'detector_finish':detector.get('finish_age_ms'),
            'histories':histories,
        },sort_keys=True))
        time.sleep(max(.2,args.interval))

if __name__=='__main__':main()
