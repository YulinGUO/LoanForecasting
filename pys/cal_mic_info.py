"""
calculate the mic of two result

"""


import pandas as pd
import numpy as np
from minepy import MINE

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

fs = ['gbdt_submit_withall_median_20171208_190111',
'subByBaggingLgb20171210_104415Final200',
'subByXgb20171209_005051Final',
'subByCat20171209_000840Final',
'rf_submit_withall_pln_median_20171209_115716',
'subByLgb20171208_224118Final']

res = []
res.append(pd.read_csv(OUTPUT_PATH+'/gbdt_submit_withall_median_20171208_190111.csv',header=None,names=['uid','target']).target.values)
res.append(pd.read_csv(OUTPUT_PATH+'/subByBaggingLgb20171210_104415Final200.csv',header=None,names=['uid','target']).target.values)
res.append(pd.read_csv(OUTPUT_PATH+'/subByXgb20171209_005051Final.csv',header=None,names=['uid','target']).target.values)
res.append(pd.read_csv(OUTPUT_PATH+'/subByCat20171209_000840Final.csv',header=None,names=['uid','target']).target.values)
res.append(pd.read_csv(OUTPUT_PATH+'/rf_submit_withall_pln_median_20171209_115716.csv',header=None,names=['uid','target']).target.values)
res.append(pd.read_csv(OUTPUT_PATH+'/subByLgb20171208_224118Final.csv',header=None,names=['uid','target']).target.values)

cc = len(res)
cm = []
for i in range(cc):
    tmp = []
    for j in range(cc):
        m = MINE()
        m.compute_score(res[i], res[j])
        tmp.append(m.mic())
    cm.append(tmp)


import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cc)
    plt.xticks(tick_marks, fs, rotation=45)
    plt.yticks(tick_marks, fs)
    plt.tight_layout()

plot_confusion_matrix(cm, title='mic')
plt.show()

