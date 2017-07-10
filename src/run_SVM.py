#!/usr/bin/env python

# SVM+PCA for Temperature Data
from progressbar import *
import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
from math import floor
import pickle
import optparse
import unittest
import random
import os, sys
import util

from sklearn import decomposition
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data_path = "../varied_level1"
delta_e_path = "../69_unprocessed_mats"
instances = 500
exp = 20
num_folds = 3
total_time = [2.0]
MAX_TIME = max(total_time)

if not os.path.exists(delta_e_path):
    os.makedirs(delta_e_path)
for i in total_time:
    if not os.path.exists(os.path.join(delta_e_path,'%0.2f' % i)):
        os.makedirs(os.path.join(delta_e_path,'%0.2f' % i))

def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()

def create_binary_dataset(fmat1, fmat2, label1, label2, t):
    data = {'data':[], 'target':[]}

    for fvec in fmat1:
        temp, slope = fvec
        length = len(temp)
        temp = temp[:int(t*length/MAX_TIME)]
        slope = slope[:int(t*length/MAX_TIME)]
        slope[-1] = 0
        data['data'].append(temp + slope)
        data['target'].append(str(label1))

    for fvec in fmat2:
        temp, slope = fvec
        length = len(temp)
        temp = temp[:int(t*length/MAX_TIME)]
        slope = slope[:int(t*length/MAX_TIME)]
        slope[-1] = 0
        data['data'].append(temp + slope)
        data['target'].append(str(label2))

    return data

def run_pca(dataset):
    # pca = decomposition.PCA(n_components=10)
    pca = decomposition.PCA()
    pca.fit(dataset['data'])
    reduced_mat = pca.transform(dataset['data'])
    dataset['data'] = reduced_mat
    return dataset


def run_crossvalidation_new(data_dict, folds):
    # data_dict = run_pca(data_dict)
    skf = StratifiedKFold(data_dict['target'], n_folds=folds)
    svc = svm.SVC(kernel='linear')
    #svc = svm.SVC(kernel='rbf')
    #svc = svm.SVC(kernel='poly', degree=3)
    scores = cross_val_score(svc, data_dict['data'], data_dict['target'], cv=skf, scoring='f1_weighted')
    # print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2)
    return scores.mean()

def transform_erfc(vec, t_m0=25):
    out = []
    for trial in vec:
        temp, slope = trial
        while True:
            time.sleep(10)
        temp = [(t - t_m0) / (temp[0] - t_m0) for t in temp]
        slope = [s / (temp[0] - t_m0) for s in slope]
        out += [(temp, slope)]
    return out

if __name__ == '__main__':

    datatags = {}
    for f in os.listdir(data_path):
        if f.endswith(".pkl"):
            trial, t_sens, t_amb, noise = f[:-4].split('_')
            key = (t_sens, t_amb, noise)

            if key in datatags:
                datatags[key].append(f)
            else:
                datatags[key] = [f]

    for t in total_time:
        print 'Iterating through time %.2f' % t
        # matrices = {}
        for indk, k in enumerate(datatags):
            if not k == ('308.15', '298.15', '0.05'):
                continue
            print 'Iterating through model %s' % str(k)
            # temp_data = {'data':[], 'target':[]}
            trials = []

            print "Loading Data"
            widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
            pbar = ProgressBar(widgets=widgets, maxval=len(datatags[k])).start()
            for i, fname in enumerate(datatags[k]):
                dataVec = util.load_pickle(os.path.join(data_path, fname))
                trials.append(dataVec)

                pbar.update(i)
            pbar.finish()
            print

            delta_e_results = {}
            instances = len(trials[0].keys())

            for ins1 in range(instances-1):
                for ins2 in range(ins1+1,instances):
                    vec1 = [dVec[dVec.keys()[ins1]] for dVec in trials]
                    vec2 = [dVec[dVec.keys()[ins2]] for dVec in trials]
                    # vec1 = transform_erfc(vec1)
                    # vec2 = transform_erfc(vec2)

                    data_dic = create_binary_dataset(vec1, vec2, ins1, ins2, t)

                    score = run_crossvalidation_new(data_dic, num_folds)

                    if not dVec.keys()[ins1] in delta_e_results.keys():
                        delta_e_results[dVec.keys()[ins1]] = {}

                    delta_e_results[dVec.keys()[ins1]][dVec.keys()[ins2]] = score
                    report = 'T %s K %s Ins1 %s Ins2 %s S %.2f' % (t, indk, str(ins1).zfill(3), str(ins2).zfill(3), score)
                    disp_to_term(report)

            util.save_pickle(delta_e_results, os.path.join(delta_e_path, '%.2f'%t, '%s_%s_%s.pkl'%k))