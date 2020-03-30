# Script to analyse print output of 'hogsvm_testbed'
# Since our testbed now generates a CSV file, we just want to scrape the
# original printed output we saved (MANUALLY) for the following missing parameters 
# we forgot to save... (Jupyter code could do with further refactoring)
#  * (initial) HOG processing and SVM training time
#  * pickle file training size (using a directory search)

# Super hacky code to scrape text files...

import re
import csv
from collections import Counter
import os

# NOTE: see main() for hardcoded input file paths
CSV_TIMING_FN = 'hogsvm_result_all_performance.csv'
CSV_FALSE_IMAGE_FN = 'hogsvm_result_all_false_images.csv'

# assume most-to-all strings are unique
# (assume positive integers)
re_newTest = re.compile(r'^####\[TEST ([0-9]+)/') # g1
re_orientations = re.compile(r'Number of orientations: ([0-9]+)') # g1
re_cellPixels = re.compile(r'Cell pixel shape: \(([0-9]+), [0-9]+\)') # g1
re_blockSize = re.compile(r'Number of cells per block: \(([0-9]+), [0-9]+\)') # g1
# (assume positive float)
rfloat = r'(\d+(\.\d*)?)'
re_hogTime = re.compile(rfloat + r's runtime \(conversion to HOG\)') # g1
re_trainingTime = re.compile(rfloat + r's runtime \(SGD training\)') # g1; not actually SGD

re_newState = re.compile(r'^\[([\w\s]+)\]:') # g1
parameter_state = 'Current parameter sweep'
training_state = 'Training SVM model'
false_img_state = 'Falsely detected images'

re_falseImg = re.compile(r'(F[PN]):\s+(.*\.pnm)') # g1,2
rfile_basename = r'_ori\(([0-9]+)\)_cellpix\(([0-9]+)\)_blksze\(([0-9]+)\)_blknrm\(L2-Hys\)'
re_genresultFile = re.compile('hogsvm_result' + rfile_basename + '.txt')
re_modelFile = re.compile('hogsvm_model' + rfile_basename + '.pickle')

# key: 'Orientations', 'Cells per Block', 'Pixels per Cell', 
#      'HOG Processing Time (s)', 'Training Time (s)'
okey = 'Orientations'
bkey = 'Cells per Block'
ckey = 'Pixels per Cell'
hkey = 'HOG Processing Time (s)'
tkey = 'Training Time (s)'

####################################################################
def main():
    # load print file output (since only these files contain the runtime lines)
    if False:
        fp1 = 'z507_hogsvm_jupyter_results_28March2020.txt'
        fp2 = 'z507_hogsvm_jupyter_results_30March2020_cellpixel_additional.txt'
        entries, __ = read_printfile(fp1)
        entries_2, __ = read_printfile(fp2)
        entries += entries_2
        with open(CSV_TIMING_FN, 'w', newline='') as csv_fh:
            csv_writer = csv.writer(csv_fh)
            csv_headers = [bkey, ckey, okey, hkey, tkey]
            csv_writer.writerow(csv_headers)
            for d in entries:
                row = [d[k] for k in csv_headers]
                csv_writer.writerow(row)
        print('Length of entries:', len(entries))

    # scrape generated log files for false images
    # (must have cached results from running Jupyter to work)
    if True:
        allfiles = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
        genresult_files = [f for f in allfiles if re_genresultFile.search(f)]
        print('#generated result files:', len(genresult_files))
        # ** get false images
        false_imgs_total = Counter()
        for f in genresult_files:
            __, false_imgs = read_printfile(f, 1)
            false_imgs_total += false_imgs
        with open(CSV_FALSE_IMAGE_FN, 'w', newline='') as csv_fh:
            csv_writer = csv.writer(csv_fh)
            csv_headers = ['Error Type', 'Image', 'Error Occurrences']
            csv_writer.writerow(csv_headers)
            for k in false_imgs_total:
                error_type, img_fp = k.split('~') # hack
                row = [error_type, img_fp, false_imgs_total[k]]
                csv_writer.writerow(row)

    # (Determine the filesize within read_printfile() instead for convenience...)
    # model_files = [f for f in allfiles if re_modelFile.search(f)]
    # print('#model pickle files:', len(model_files))

####################################################################
# Reads file storing Jupyter print output for specific information (that was not 
# automatically saved...) regarding training time
def read_printfile(fp, cur_test_index=None):
    with open(fp) as fh:
        file_lines = fh.readlines()

    # cur_test_index = None   # can preset `cur_test_index` to fake this to run
    cur_state = None

    # to generate CSV performance params
    entry = {} 
    entries = []
    # to track false image statistical occurrences (in test data)
    # key: 
    false_imgs = Counter()

    for line in file_lines:
        line = line.strip()
        # synchronise to test case (if not already)
        m = re_newTest.search(line)
        if m:
            # save old record
            if len(entry) > 0:
                entries.append(entry)
            # set up new record
            cur_test_index = int(m.group(1))
            # if cur_test_index == 2: # DEBUG only
            #     break
            entry['Test ID'] = cur_test_index
            entry = {} # init
            cur_state = None
        if cur_test_index == None:
            continue
        # synchronise to state
        m = re_newState.search(line)
        if m:
            cur_state = m.group(1)

        # get parameter from required state
        if cur_state == parameter_state: # parameters
            # orientations
            m = re_orientations.search(line)
            if m:
                entry[okey] = int(m.group(1))
                continue
            # pixels per cell
            m = re_cellPixels.search(line)
            if m:
                entry[ckey] = int(m.group(1))
            # cells per block
            m = re_blockSize.search(line)
            if m:
                entry[bkey] = int(m.group(1))
        elif cur_state == training_state: # training
            # hog time
            m = re_hogTime.search(line)
            if m:
                entry[hkey] = float(m.group(1))
                continue
            # training time
            m = re_trainingTime.search(line)
            if m:
                entry[tkey] = float(m.group(1))
                continue
        elif cur_state == false_img_state: # false images (listed)
            m = re_falseImg.search(line)
            if m: # hack
                false_imgs[m.group(1)+'~'+m.group(2)] += 1
                continue

    return entries, false_imgs

######################################################################
if __name__ == '__main__':
    main()
