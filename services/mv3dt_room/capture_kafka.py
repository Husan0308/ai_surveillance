#!/usr/bin/env python3
import json, signal, time
from kafka import KafkaConsumer, TopicPartition
from google.protobuf.json_format import MessageToDict
from schema_pb2 import Frame

running=True
def stop(*_):
    global running; running=False
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
c=KafkaConsumer(bootstrap_servers='localhost:9092',enable_auto_commit=False,consumer_timeout_ms=1000)
tp=TopicPartition('mv3dt',0); c.assign([tp]); start=c.end_offsets([tp])[tp]; c.seek(tp,start)
print(json.dumps({'event':'capture_start','offset':start,'host_time_ns':time.time_ns()}),flush=True)
count=0
while running:
    for _,records in c.poll(timeout_ms=250,max_records=1000).items():
        for msg in records:
            f=Frame(); f.ParseFromString(msg.value)
            print(json.dumps({'event':'message','offset':msg.offset,'broker_timestamp_ms':msg.timestamp,'host_time_ns':time.time_ns(),'frame':MessageToDict(f)},separators=(',',':')),flush=True)
            count+=1
print(json.dumps({'event':'capture_stop','count':count,'host_time_ns':time.time_ns()}),flush=True)
c.close()
