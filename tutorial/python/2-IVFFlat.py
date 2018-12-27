# -*- coding: utf-8 -*-

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import json as js
import time
import math

corpus_obj = js.loads(open('./data/kmQuery.json.20181218').read())['data']
kb = {}               # key is knowledge id, values is knowledge text list, including standard and similarity questions
kb_stan = {}
for item in corpus_obj:
    eid = item['eId'].encode("utf-8")
    kb[eid] = []
    kb[eid].append(item['query'].encode('utf-8'))
    kb_stan[eid] = item['query'].encode('utf-8')
    for sim in item['similarQueries']:
        kb[item['eId']].append(sim['query'].encode('utf-8'))

kb_query_list = []    # key is index id, value is revised query of knowledge
kb_dict = {}          # key is index id, value is knowledge id
kb_reverse_dict = {}  # key is knowledge id, value is index id list
index_kb = []         # train set
with open(sys.argv[1], 'r') as f:
    for idx, line in enumerate(f):
        json_obj = js.loads(line.strip('\r\n'))
        index_kb.append([float(item) for item in json_obj['embeding']])
        kid = json_obj['kid'].encode('utf-8')
        kb_dict[idx] = kid
        if kid not in kb_reverse_dict:
            kb_reverse_dict[kid] = []
        kb_reverse_dict[kid].append(idx)
        kb_query_list.append(json_obj['r_query'].encode('utf-8'))
    f.close()
index_kb = np.asarray(index_kb, dtype=np.float32)

revised_query_list = []    # revised query
query_list = []            # prediction data
label_kid_list = []        # knowledge id list
label = []                 # index id list
query_size = 0
with open(sys.argv[2], 'r') as f:
    for idx, line in enumerate(f):
        json_obj = js.loads(line.strip('\r\n'))
        '''
        kid = json_obj['kid'].encode('utf-8')
        if kid not in kb_reverse_dict:
            continue
        query_size += 1
        label_kid_list.append(kid)
        label.append(kb_reverse_dict[kid])
        '''
        revised_query_list.append(json_obj['r_query'].encode('utf-8'))
        query_list.append([float(item) for item in json_obj['embeding']])
    f.close()
query_list = np.asarray(query_list, dtype=np.float32)

dim_size = index_kb.shape[1]            # dimension
train_size = index_kb.shape[0]           # database size
recall_size = int(sys.argv[3])
index_probe = int(sys.argv[4])

import faiss

def FlatL2Index():
    index = faiss.IndexFlatL2(dim_size)
    assert index.is_trained
    index.add(index_kb)

    file = open('./data.20181219/flagL2.dat', 'w')
    start_time = time.time()
    D, I = index.search(query_list[:1], recall_size)
    print("FlatL2 index %s seconds" % (time.time() - start_time))
    D, I = index.search(query_list, recall_size)     # actual search
    correct_counter = 0
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
        else:
            file.write(revised_query_list[idx] + '\t' + '####'.join(kb[label_kid_list[idx]]) + '\t' + '####'.join([kb_query_list[int(item)] for item in recall_list]) + '\n')
    print("FlatL2 precision is %f" % (float(correct_counter) / float(query_size)))
    file.close()
#FlatL2Index()

def FlatL2Index_kb():
    index = faiss.IndexFlatL2(dim_size)
    assert index.is_trained
    index.add(index_kb)

    file = open('./data.20181219/flagL2.dat', 'w')
    start_time = time.time()
    D, I = index.search(query_list[:1], recall_size)
    print("FlatL2 index %s seconds" % (time.time() - start_time))
    D, I = index.search(query_list, recall_size)     # actual search

    with open('./data.20181219/jiayi.csv', 'w') as f:
        for idx, recall_list in enumerate(I):
            kb_str_list = []
            for kb_idx in recall_list:
                if kb_stan[kb_dict[int(kb_idx)]] not in kb_str_list:
                    kb_str_list.append(kb_stan[kb_dict[int(kb_idx)]])
                if len(kb_str_list) >= 12:
                    break
            f.write(revised_query_list[idx] + '\t' + ('####').join(kb_str_list) + '\n')
        f.close()

    '''
    correct_counter = 0
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
        else:
            file.write(revised_query_list[idx] + '\t' + '####'.join(kb[label_kid_list[idx]]) + '\t' + '####'.join([kb_query_list[int(item)] for item in recall_list]) + '\n')
    print("FlatL2 precision is %f" % (float(correct_counter) / float(query_size)))
    file.close()
    '''
FlatL2Index_kb()

def IVFFlatIndex():
    nlist = int(math.sqrt(train_size))    # number of clusters
    quantizer = faiss.IndexFlatL2(dim_size)  # store cluster center-ids
    index = faiss.IndexIVFFlat(quantizer, dim_size, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search, or faiss.METRIC_INNER_PRODUCT

    assert not index.is_trained
    index.train(index_kb)
    assert index.is_trained
    index.add(index_kb)                  # add may be a bit slower as well

    D, I = index.search(query_list, recall_size)     # actual search

    correct_counter = 0
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
    print("IVFFlat index probe 1 precision is %f" % (float(correct_counter) / float(query_size)))
    index.nprobe = index_probe               # default nprobe is 1, try a few more
    start_time = time.time()
    D, I = index.search(query_list[:1], recall_size)
    print("IVFFlat index %s seconds" % (time.time() - start_time))
    D, I = index.search(query_list, recall_size)
    correct_counter = 0
    file = open('./data.20181219/IVFFlat.dat', 'w')
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
        else:
            file.write(revised_query_list[idx] + '\t' + '####'.join(kb[label_kid_list[idx]]) + '\t' + '####'.join([kb_query_list[int(item)] for item in recall_list]) + '\n')
    file.close()
    print("IVFFlat index probe %d precision is %f" % (index_probe, float(correct_counter) / float(query_size)))
#IVFFlatIndex()

def IVFPQIndex():
    nlist = 50    # coarse-grained cluster size
    m = 8
    quantizer = faiss.IndexFlatL2(dim_size)  # coarse-grained cluster centro-ids
    index = faiss.IndexIVFPQ(quantizer, dim_size, nlist, m, 8)
    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(index_kb)
    index.add(index_kb)
    D, I = index.search(query_list, recall_size) # sanity check

    correct_counter = 0
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
    print("IVFPQ index probe 1 precision is %f" % (float(correct_counter) / float(query_size)))
    index.nprobe = index_probe               # default nprobe is 1, try a few more
    start_time = time.time()
    D, I = index.search(query_list[:1], recall_size)
    print("IVFPQ index %s seconds" % (time.time() - start_time))
    D, I = index.search(query_list, recall_size)

    correct_counter = 0
    file = open('./data.20181219/IVFPQ.dat', 'w')
    for idx, recall_list in enumerate(I):
        if len(np.intersect1d(recall_list, np.array(label[idx], dtype=np.int32))) > 0:
            correct_counter += 1
        else:
            file.write(revised_query_list[idx] + '\t' + '####'.join(kb[label_kid_list[idx]]) + '\t' + '####'.join([kb_query_list[int(item)] for item in recall_list]) + '\n')
    file.close()
    print("IVFPQ index probe %d precision is %f" % (index_probe, float(correct_counter) / float(query_size)))
#IVFPQIndex()