//
// Created by xtpan on 2018/12/20.
//

#include <ChatbotFaiss.h>

static int IndexFactory::CreateIndex(IndexType indexType, int dim, int db_size, float* db)
{
    if (indexType != IndexType::FlatL2 || db == null)
    {
        printf("not support index type besides FlatL2\n");
        return -1;
    }

    faiss::IndexFlatL2 index(dim);
    if (index.is_trained == 0)
    {
        printf("train status is false\n");
        return -1;
    }
    index.add(db_size, db);

    this->m_flat_index = &index;
    return 0;
};

static void IndexFactory::Search(IndexType indexType, float* query, float* kb_index_array, int query_size, int recall_size, int probe_size=1)
{
    if (indexType != IndexType::FlatL2 || query == null || kb_index_array == null)
    {
        printf("not support index type besides FlatL2\n");
        return -1;
    }
    float* D = new float[recall_size * query_size];
    this->m_flat_index.search(query_size, query, recall_size, D, kb_index_array);

    delete I, D;
    return 0;
};
