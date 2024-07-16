import json
import sys
import argparse
import pandas as pd


def parse_keywords(is_ok, keywords):
    if is_ok == "OK":
        keywords = keywords.split(", ")
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
    # expects tab-separated columns with the following header: VIDEO_ID        VIDEO_NAME      SENTENCE_ID
    # SENTENCE_NAME   START_REALIGNED END_REALIGNED   SENTENCE        STATE   KEYWORDS
    return line[0], line[3], line[6], parse_keywords(line[7], line[8])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_tsv', type=str, help="The input tsv file")
    parser.add_argument('-input_csv', type=str, help="The input csv file")
    parser.add_argument('-output', type=str, help="The output file")
    parser.add_argument('-dataset', type=str, help="The format of the input file")
    args = parser.parse_args()

    if args.dataset == "y":
        print("INFO: parsing Youtube-ASL", file=sys.stderr)
        parse_line = parse_line_yt
    else:
        print("INFO: parsing How2Sign", file=sys.stderr)
        parse_line = parse_line_h2s
    data = {}  # creating data from scratch. If only adding paraphrases, first load this dict from file...

    df_tsv = pd.read_csv(args.input_tsv, sep="\t", header=None)
    df_csv = pd.read_csv(args.input_csv, sep="\t")
    for (index, tsv_line),(index2, csv_line) in zip(df_tsv.iterrows(), df_csv.iterrows()):

        vid, clip_name, transl, keywords = parse_line(tsv_line)

        if vid not in data:
            data[vid] = {
                "clip_order": []  # skip this in adding paraphrases
            }

        data[vid]["clip_order"].append(clip_name)  # skip this in adding paraphrases

        # in adding paraphrases, change this to add the paraphrases
        data[vid][clip_name] = {
            "translation": transl,
            "keywords": keywords,
            "paraphrases": [csv_line["GPT_Rephrase1"], csv_line["GPT_Rephrase2"], csv_line["GPT_Rephrase3"],
                            csv_line["GPT_Rephrase4"], csv_line["GPT_Rephrase5"]]
        }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)
