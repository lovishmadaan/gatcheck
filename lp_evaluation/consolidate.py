import numpy as np
import sys
from os import listdir
from os.path import isfile, join

dataset = sys.argv[1]
path = 'results/' + dataset + '/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

results = []
for file in onlyfiles:
    parameters = file[ : -4].split('_')[3 : ]
    # print(parameters)
    # exit()
    l = int(parameters[0][len('layer') : ])
    approx = int(parameters[1][len('approximate') : ])
    h = int(parameters[2][len('h') : ])
    ina = int(parameters[3][len('ina') : ])
    outa = int(parameters[4][len('outa') : ])
    act = str(parameters[5][len('act') : ])

    with open(path + file, 'r') as fr:
        scores = fr.read().strip()
        mean, stdev = tuple(map(float, scores.split(', ')))
        mean=float("{0:.3f}".format(mean))
        stdev=float("{0:.3f}".format(stdev))
        results.append([mean, stdev, l, approx, h, ina, outa,act])
    fr.close()

results = np.array(results)
results = np.flip(results[results[:, 0].argsort()], axis=0)
print(results[0])

np.savetxt('consolidated_pairwise_' + str(dataset) + '.csv', results, delimiter=',', fmt='%s', header='mean, std, layers, approx, hidden_dim, attn_heads_hidden, attn_heads_output, act', comments='')