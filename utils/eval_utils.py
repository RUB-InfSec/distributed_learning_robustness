import os

import numpy as np
import orjson as json


def get_results_from_data(data):
    results = []
    for exp in data.keys():
        for distr in data[exp].keys():
            for r in data[exp][distr].keys():
                for N in data[exp][distr][r].keys():
                    res = get_results(data[exp][distr][r][N])
                    eps = int(exp.split("_")[-1])
                    expname = exp.replace(f"_eps_{eps}", "")
                    line = [expname.replace("uniform", ""), distr, int(r), int(N), int(eps)]
                    if expname == "master_configs":
                        line[0] = "Central"
                    for vote in res.keys():
                        l = line.copy()
                        l.append(vote)
                        for v in res[vote]:
                            l.append(v)
                        results.append(l)
    return results


def get_raw_data(dir_path):
    data = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as f:
            expname = filename.replace("results_", "").replace(".json", "")
            l = f.readline()
            data[expname] = json.loads(l)
    return data


def get_results(data):
    results = {'soft': [], 'hard': [], 'weighted': []}
    benignIDX = None
    for dataset in data.keys():
        svotes = None
        hvotes = None
        mvotes = None
        for n in data[dataset].keys():
            labels = np.asarray(data[dataset][n]["labels"])
            labels = np.asarray(labels)
            labels = labels.reshape(-1)

            x = np.asarray(data[dataset][n]["acc_out"])
            x = x.reshape(-1, 10)
            if (np.max(x) > 1).any() or (np.min(x) < 0).any():
                x = np.exp(x - np.max(x, axis=1)[:, None])
                x = x / x.sum(axis=1)[:, None]

            # Soft votes:
            if svotes is None:
                svotes = x
            else:
                svotes += x
            # Hard votes:
            idx = np.argmax(x, axis=1)
            xh = np.zeros_like(x)
            xh[np.arange(labels.size), idx] = 1
            if hvotes is None:
                hvotes = xh
            else:
                hvotes += xh

            # Weighted votes:
            xw = np.zeros_like(x)
            xw[np.arange(labels.size), idx] = x[np.arange(labels.size), idx]
            if mvotes is None:
                mvotes = xw
            else:
                mvotes += xw

        svotes = np.argmax(svotes, axis=1) == labels
        hvotes = np.argmax(hvotes, axis=1) == labels
        mvotes = np.argmax(mvotes, axis=1) == labels
        if dataset == "benign":
            benignIDX = [svotes, hvotes, mvotes]
            results['soft'].append(svotes.sum() / svotes.size)
            results['hard'].append(hvotes.sum() / hvotes.size)
            results['weighted'].append(mvotes.sum() / mvotes.size)
        else:
            results['soft'].append(1 - svotes[benignIDX[0]].sum() / benignIDX[0].sum())
            results['hard'].append(1 - hvotes[benignIDX[0]].sum() / benignIDX[0].sum())
            results['weighted'].append(1 - mvotes[benignIDX[0]].sum() / benignIDX[0].sum())
    return results
