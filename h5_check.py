import h5py


with h5py.File('./h5py/pose.h5','r') as fr:
    keys = list(fr.keys())

    for key in keys:
        print(key)
        for sub_key in fr[key].keys():
            print(sub_key)
            for i in range(0, len(fr[key][sub_key])):
                print(fr[key][sub_key][i])
            print("\n")
