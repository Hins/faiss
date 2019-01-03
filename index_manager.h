//
// Created by xtpan on 2018/12/26.
//

#ifndef FAISS_INDEX_MANAGER_H
#define FAISS_INDEX_MANAGER_H

#include <string>
#include <map>

using namespace std;

namespace faiss {
    struct IndexFlatL2;
    struct IndexIVFFlat;
    struct IndexIVFPQ;
};

class IndexManager {
public:
    static IndexManager *CreateInstance();
    static IndexManager *GetInstance();
    static int ReleaseInstance();
    IndexManager();
    ~IndexManager();

public:
    long long GetMapSizeFlatL2();
    long long SetMapKeyFlatL2(string key, int ivec);
    bool DelMapKeyFlatL2(string key);
    bool DelMapKeyAllFlatL2();
    faiss::IndexFlatL2 *GetMapKeyFlatL2(string key);

public:
    long long GetMapSizeIVFFlat();
    long long SetMapKeyIVFFlat(string key, int d, int nlist, int metricType);
    faiss::IndexIVFFlat *GetMapKeyIVFFlat(string key);
    bool DelMapKeyAllIVFFlat();

public:
    long long SetMapKeyIVFPQ(string key, int d, int nlist, int m, int nbits);
    faiss::IndexIVFPQ *GetMapKeyIVFPQ(string key);
    bool DelMapKeyAllIVFPQ();

private:
    static IndexManager *pInstance_;
    std::map<string, faiss::IndexFlatL2 *> flatl2_map_;
    std::map<string, faiss::IndexIVFFlat *> ivfflat_map_;
    std::map<string, faiss::IndexIVFPQ *> ivfpq_map_;
};

#endif //FAISS_INDEX_MANAGER_H
