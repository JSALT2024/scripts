#!/usr/bin/env python3
import json
import sys

data = json.load(sys.stdin)
for vid in data.keys():
    for clip in data[vid]["clip_order"]:
        transl = data[vid][clip]["translation"]
        print(vid,clip,transl, sep="\t")
