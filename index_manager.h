//
// Created by xtpan on 2018/12/26.
//

#ifndef FAISS_INDEX_MANAGER_H
#define FAISS_INDEX_MANAGER_H

#include <string>
#include <vector>
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
    long long SetMapKeyKB(string key, const map<string, string>& kb);
    bool DelMapKeyKB(string key);
    long long SetListKeyKB(string key, const vector<string>& kb);
    bool DelListKeyKB(string key);
    string GetKBStanQ(string key, int index);

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
    std::map<string, map<string, string> > kb_map;
    std::map<string, vector<string> > index_map;
    std::map<string, faiss::IndexFlatL2 *> flatl2_map_;
    std::map<string, faiss::IndexIVFFlat *> ivfflat_map_;
    std::map<string, faiss::IndexIVFPQ *> ivfpq_map_;
};

#endif //FAISS_INDEX_MANAGER_H
