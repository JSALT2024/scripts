import pandas as pd

original_file = pd.read_csv(r'C:\Work\JSALT\scripts\data\train.filtered3.beg_dur_id_frames_fps_text.norm.filter-lanid.tsv', sep='\t', header=None)
rephrased_file = pd.read_csv(r'C:\Work\JSALT\scripts\data\rephrased_0-595423_fixed3.csv')

print()
