import json

with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.dev.json') as f:
    data = json.load(f)

for video in data:
    clip_order = data[video]["clip_order"]
    starting_value = 0
    for clip in clip_order:
        clip_time = clip.split(".")[1]
        start, finish = clip_time.split("-")
        start = int(start)
        finish = int(finish)
        if start < starting_value:
            print(video)
            print(clip_order)
        starting_value = start
