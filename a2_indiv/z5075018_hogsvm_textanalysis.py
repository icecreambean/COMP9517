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
#re_testingTime = re.compile(rfloat + r's prediction time \(')

re_newState = re.compile(r'^\[([\w\s]+)\]:') # g1
parameter_state = 'Current parameter sweep'
training_state = 'Training SVM model'
#testing_state = 'Classifier statistics (on test data)'
false_img_state = 'Falsely detected images'

re_falseImg = re.compile(r'(F[PN]):\s+(.*\.pnm)') # g1,2
rfile_basename = r'_ori\(([0-9]+)\)_cellpix\(([0-9]+)\)_blksze\(([0-9]+)\)_blknrm\(L2-Hys\)'
re_genresultFile = re.compile('hogsvm_result' + rfile_basename + '.txt')
#re_modelFile = re.compile('hogsvm_model' + rfile_basename + '.pickle')
def get_model_fn(orientations, cellpixels, blocksize):
    base = '_ori({})_cellpix({})_blksze({})_blknrm(L2-Hys)'.format(orientations, cellpixels, blocksize)
    return 'hogsvm_model' + base + '.pickle'


# key: 'Orientations', 'Cells per Block', 'Pixels per Cell', 
#      'HOG Processing Time (s)', 'Training Time (s)'
okey = 'Orientations'
bkey = 'Cells per Block'
ckey = 'Pixels per Cell'
hkey = 'HOG Processing Time (s)'
tkey = 'Training Time (s)'
skey = 'Training Pickle Size (bytes)'

####################################################################
def main():
    # load print file output (since only these files contain the runtime lines)
    if True:
        fp_list = [
            'z507_hogsvm_jupyter_results_28March2020.txt',
            'z507_hogsvm_jupyter_results_30March2020_cellpixel_additional.txt',
            'z507_hogsvm_jupyter_results_30March2020_orientation_additional.txt',
            'z507_hogsvm_jupyter_results_30March2020_all224test.txt',
            'z507_hogsvm_jupyter_results_1Apr2020.txt',
            'z507_hogsvm_jupyter_results_1Apr2020_14-15-16.txt'
        ]
        entries = []
        for fpi in fp_list:
            subentries, __ = read_printfile(fpi, include_filesize=True)
            entries += subentries
        with open(CSV_TIMING_FN, 'w', newline='') as csv_fh:
            csv_writer = csv.writer(csv_fh)
            csv_headers = [bkey, ckey, okey, hkey, tkey, skey]
            csv_writer.writerow(csv_headers)
            count = 0
            for d in entries:
                # super hacky: ignore row if expected headers not in row
                # (handles the issue of duplicate output, assuming no pickle file deletions)
                if hkey not in d or tkey not in d:
                    continue
                row = [d[k] for k in csv_headers]
                csv_writer.writerow(row)
                count += 1
        print('Length of entries: {},   Actual entries written to csv: {}'.format(len(entries), count))

    # scrape generated log files for false images
    # (must have cached results from running Jupyter to work)
    if True:
        allfiles = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
        genresult_files = [f for f in allfiles if re_genresultFile.search(f)]
        print('#generated result files:', len(genresult_files))
        # ** get false images
        false_imgs_total = Counter()
        for f in genresult_files:
            __, false_imgs = read_printfile(f, cur_test_index=1)
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
def read_printfile(fp, cur_test_index=None, include_filesize=False):
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
    termination_case = lambda x: x >= len(file_lines)

    #for line in file_lines:
    i = -1
    while True:
        i += 1 # update loop
        if not termination_case(i):
            line = file_lines[i]
            line = line.strip()
        else:
            line = '' # (hack to save code duplication)
        # synchronise to test case (if not already)
        m = re_newTest.search(line)
        if m or termination_case(i):
            # save old record
            if len(entry) > 0:
                # hack: include filesize of pickle model
                if include_filesize:
                    train_pickle_fn = get_model_fn(entry[okey], entry[ckey], entry[bkey])
                    entry[skey] = os.path.getsize(train_pickle_fn)
                # (save old record)
                entries.append(entry)
                # termination case
                if termination_case(i):
                    break
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
        # elif cur_state == testing_state:
        #     # testing time
        #     m = re_testingTime.search(line)
        #     if m:
        #         entry[testkey] = float(m.group)
        #         continue
        elif cur_state == false_img_state: # false images (listed)
            m = re_falseImg.search(line)
            if m: # hack
                false_imgs[m.group(1)+'~'+m.group(2)] += 1
                continue

    return entries, false_imgs

######################################################################
if __name__ == '__main__':
    main()
