import csv
import json
import sys

# by Dominik
# Út 16. července 2024, 20:40:52 CEST

# Parse .tsv data with keywords into json
# It doesn't include the paraphrases. Then another script 

def parse_keywords(is_ok, keywords):
    if is_ok == "OK":
        keywords = eval(keywords).split(", ") 
    else:
        keywords = []
    return keywords


def parse_line_yt(line):
    beg_t, end_t, vid, begframe, endframe, fps, transl, is_ok, keywords = line.split("\t")
    begframe = int(begframe)
    endframe = int(endframe)
    clip_name = "%06d-%06d" % (begframe, endframe)

    return vid, clip_name, transl, parse_keywords(is_ok, keywords)

def parse_line_h2s(line):
    # expects tab-separated columns with the following header:
    #    VIDEO_ID        VIDEO_NAME      SENTENCE_ID     SENTENCE_NAME   START_REALIGNED END_REALIGNED   SENTENCE        STATE   KEYWORDS  
    vid, _, clip_name, _, _, _, transl, is_ok, keywords = line.split("\t")
    return vid, clip_name, transl, parse_keywords(is_ok, keywords)

if len(sys.argv) < 2:
    print(f"Usage: python3 %s [h or y] < in.tsv > out.json" % sys.argv[0],file=sys.stderr)
    print(f"h ... expects how2sign format",file=sys.stderr)
    print(f"y ... youtube-asl format",file=sys.stderr)
    sys.exit(1)

if sys.argv[1][0] == "y":
    print("INFO: parsing Youtube-ASL",file=sys.stderr)
    parse_line = parse_line_yt
else:
    print("INFO: parsing How2Sign",file=sys.stderr)
    parse_line = parse_line_h2s

data = {}  # creating data from scratch. If only adding paraphrases, first load this dict from file...

for line in sys.stdin:

    vid, clip_name, transl, keywords = parse_line(line)

    if vid not in data:
        data[vid] = {
                "clip_order":[]  # skip this in adding paraphrases
        }

    data[vid]["clip_order"].append(clip_name)  # skip this in adding paraphrases

    # in adding paraphrases, change this to add the paraphrases
    data[vid][clip_name] = {
            "translation":transl,
            "keywords":keywords,
            }

json.dump(data, sys.stdout, indent=4)
