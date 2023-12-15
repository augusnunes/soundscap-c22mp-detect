import p_tqdm
from tqdm import tqdm 
from soundscape_c22mp import detect_anomalies
from glob import glob
import numpy as np
import pickle

if __name__=="__main__":
    dirs = [
        "./data/",
    ]

    files = []

    for d in dirs:
        files.extend(glob(d + "*.wav"))

    files = sorted(files)

    batches = 1
    files = np.array_split(files, batches)

    for i, f in tqdm(enumerate(files)):
        outputs = p_tqdm.p_map(detect_anomalies, f, num_cpus=8)
        results = dict(zip(f.tolist(), outputs))

        with open("results_{}.pkl".format(i), "wb") as f:
            pickle.dump(results, f)

    