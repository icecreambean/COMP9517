# Script to analyse print output of 'hogsvm_testbed'
import re

fh = open('z507_hogsvm_jupyter_results_28March2020_v2.txt')

# assume most-to-all strings are unique
re_newTest = re.compile(r'^####\[TEST ([0-9]+)/') # g1
re_orientations = re.compile(r'Number of orientations: ([0-9]+)') # g1
re_cellPixels = re.compile(r'Cell pixel shape: \(([0-9]+), [0-9]+\)') # g1
re_blockSize = re.compile(r'Number of cells per block: \(([0-9]+), [0-9]+\)') # g1

cur_test_index = None

# key: '{orientations}_{cell pixels}_{block size}'
# vals: 'feature_size', 'auc', 'accuracy', 'precision', 'recall',
#       'training_accuracy'
dr = {}


for line in fh:
    line = line.strip()
    # synchronise to test case (if not already)
    m = re_newTest.match(line)
    if m:
        cur_test_index = int(m.group(1))
        dr[cur_test_index] = {} # init
    if cur_test_index == None:
        continue
    # 
        

fh.close()