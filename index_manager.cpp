//
// Created by xtpan on 2018/12/26.
//
#include "index_manager.h"
#include "IndexFlat.h"
#include "IndexIVF.h"
#include "IndexIVFFlat.h"
#include "IndexIVFPQ.h"
#include <assert.h>

using namespace std;

IndexManager* IndexManager::pInstance_ = nullptr;

IndexManager::IndexManager() {}

IndexManager::~IndexManager() {
    for (auto iter : index_map) {
        string* kbs = nullptr;
        kbs = iter.second;
        if (kbs != nullptr) {
            delete kbs;
            kbs = nullptr;
        }
        index_map.erase(iter.first);
    }
    for (auto iter : flatl2_map_) {
        faiss::IndexFlatL2* pindex = nullptr;
        pindex = iter.second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        flatl2_map_.erase(iter.first);
    }

    for (auto iter : ivfflat_map_) {
        faiss::IndexIVFFlat* pindex = nullptr;
        pindex = iter.second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        ivfflat_map_.erase(iter.first);
    }

    for (auto iter : ivfpq_map_) {
        faiss::IndexIVFPQ* pindex = nullptr;
        pindex = iter.second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        ivfpq_map_.erase(iter.first);
    }
}

// singletion instance
IndexManager* IndexManager::CreateInstance() {
    if (pInstance_ == nullptr) {
        pInstance_ = new IndexManager();
        assert(pInstance_ != nullptr);
    }
    assert(pInstance_ != nullptr);
    return pInstance_;
}

IndexManager* IndexManager::GetInstance() {
    assert(pInstance_ != nullptr);
    return pInstance_;
}

int IndexManager::ReleaseInstance() {
    if (pInstance_ != nullptr) {
        delete pInstance_;
        pInstance_ = nullptr;
        return 1;
    }
    return 0;
}

long long IndexManager::SetMapKeyKB(string key, map<string, string>& kb) {
    kb_map[key] = kb;
    return false;
}

bool IndexManager::DelMapKeyKB(string key) {
    if (kb_map.size() < 1)
        return true;
    auto iter = kb_map.find(key);
    if (iter == kb_map.end())
        return true;
    kb_map.erase(key);
    return false;
}

long long IndexManager::SetListKeyKB(string key, string* kb) {
    index_map[key] = kb;
    return false;
}

bool IndexManager::DelListKeyKB(string key) {
    if (index_map.size() < 1)
        return true;
    auto iter = index_map.find(key);
    if (iter == index_map.end())
        return true;
    index_map.erase(key);
    return false;
}

long long IndexManager::GetMapSizeFlatL2() { return flatl2_map_.size(); }

long long IndexManager::SetMapKeyFlatL2(string key, int ivec) {
    faiss::IndexFlatL2* pindex = new faiss::IndexFlatL2(ivec);
    if (pindex == nullptr)
        return -1;
    flatl2_map_.insert(std::make_pair(key, pindex));
    return 0;
}

bool IndexManager::DelMapKeyFlatL2(string key) {
    if (flatl2_map_.size() < 1)
        return true;
    faiss::IndexFlatL2* pindex = nullptr;
    pindex = flatl2_map_[key];
    if (pindex != nullptr) {
        delete pindex;
        pindex = nullptr;
    }
    flatl2_map_.erase(key);
    return false;
}

bool IndexManager::DelMapKeyAllFlatL2() {
    auto iter = flatl2_map_.begin();
    while (iter != flatl2_map_.end()) {
        faiss::IndexFlatL2* pindex = nullptr;
        pindex = iter->second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        string key = iter->first;
        iter++;
        flatl2_map_.erase(key);
    }
    return 0;
}

faiss::IndexFlatL2* IndexManager::GetMapKeyFlatL2(string key) {
    if (flatl2_map_.size() < 1)
        return nullptr;
    faiss::IndexFlatL2* pindex = nullptr;
    pindex = flatl2_map_[key];
    return pindex;
}

long long IndexManager::GetMapSizeIVFFlat() { return ivfflat_map_.size(); }

long long IndexManager::SetMapKeyIVFFlat(string key, int d, int nlist, int metricType) {
    faiss::IndexFlatL2* pindexl2 = GetMapKeyFlatL2(key);
    faiss::MetricType mt;
    if (pindexl2 == nullptr)
        return -1;
    if (metricType == 0) {
        mt = faiss::METRIC_INNER_PRODUCT;
    } else if (metricType == 1) {
        mt = faiss::METRIC_L2;
    } else {
        return -2;
    }
    faiss::IndexIVFFlat* pindex = new faiss::IndexIVFFlat(pindexl2, d, nlist, mt);
    if (pindex == nullptr)
        return -3;
    ivfflat_map_.insert(std::make_pair(key, pindex));
    return 0;
}

faiss::IndexIVFFlat* IndexManager::GetMapKeyIVFFlat(string key) {
    if (ivfflat_map_.size() < 1)
        return nullptr;
    faiss::IndexIVFFlat* pindex = nullptr;
    pindex = ivfflat_map_[key];
    return pindex;
}

bool IndexManager::DelMapKeyAllIVFFlat() {
    auto iter = ivfflat_map_.begin();
    while (iter != ivfflat_map_.end()) {
        faiss::IndexIVFFlat* pindex = nullptr;
        pindex = iter->second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        string key = iter->first;
        iter++;
        ivfflat_map_.erase(key);
    }
    return 0;
}

long long IndexManager::SetMapKeyIVFPQ(string key, int d, int nlist, int m, int nbits) {
    faiss::IndexFlatL2* pindexl2 = GetMapKeyFlatL2(key);
    if (pindexl2 == nullptr)
        return -1;
    faiss::IndexIVFPQ* pindex = new faiss::IndexIVFPQ(pindexl2, d, nlist, m, nbits);
    if (pindex == nullptr)
        return -2;
    ivfpq_map_.insert(std::make_pair(key, pindex));
    return 0;
}

faiss::IndexIVFPQ* IndexManager::GetMapKeyIVFPQ(string key) {
    if (ivfpq_map_.size() < 1)
        return nullptr;
    faiss::IndexIVFPQ* pindex = nullptr;
    pindex = ivfpq_map_[key];
    return pindex;
}

bool IndexManager::DelMapKeyAllIVFPQ() {
    auto iter = ivfpq_map_.begin();
    while (iter != ivfpq_map_.end()) {
        faiss::IndexIVFPQ* pindex = nullptr;
        pindex = iter->second;
        if (pindex != nullptr) {
            delete pindex;
            pindex = nullptr;
        }
        string key = iter->first;
        iter++;
        ivfpq_map_.erase(key);
    }
    return 0;
}