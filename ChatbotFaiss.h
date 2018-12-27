//
// Created by xtpan on 2018/12/20.
//

#ifndef FAISS_CHATBOTFAISS_H
#define FAISS_CHATBOTFAISS_H

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>

class IndexFactory
{
    public:
        static IndexFactory& instance()
        {
            static IndexFactory INSTANCE;
            return INSTANCE;
        }

        static int CreateIndex(IndexType indexType, int dim, int db_size, float* db);
        // for IVF index, define probe_size to control search region
        static void Search(IndexType indexType, long* kb_index_array, int query_size, int recall_size, int probe_size=1);
    private:
        IndexFactory(){};

        faiss::IndexFlatL2* m_flat_index;
        enum IndexType
        {
            FlatL2 = 0,
            IVFFlat,
            IVFPQ
        };
};

#endif //FAISS_CHATBOTFAISS_H