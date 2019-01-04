#include "faiss_jni.h"
#include "index_manager.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <jni.h>
#include <Index.h>
#include <IndexFlat.h>
#include <vector>
#include <map>
#include <memory>

#include<stdio.h>

using namespace std;
using namespace faiss;

/*
 * Class:     faiss_FaissJni
 * Method:    cppCtor
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_cppCtor
(JNIEnv *env, jobject obj, jstring group, jint dim, jint index_type, jobject kbIndexMap, jobject trainIndexList) {
    string sGroup = env->GetStringUTFChars(group, 0);
    IndexManager* pinstance = IndexManager::CreateInstance();
    pinstance->DelMapKeyFlatL2(sGroup);
    pinstance->SetMapKeyFlatL2(sGroup, dim);
    //env->ReleaseStringUTFChars(group, sGroup);
}

/*
 * Class:     faiss_FaissJni
 * Method:    add
 * Signature: (II)V
 */
JNIEXPORT jint JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_add
(JNIEnv *env, jobject obj, jstring group, jint n, jfloatArray x) {
    string sGroup = env->GetStringUTFChars(group, 0);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr)
        return -1;
    jfloat* pbuffer = env->GetFloatArrayElements(x, JNI_FALSE);
    pindex->add(n, pbuffer);
    env->ReleaseFloatArrayElements(x, pbuffer, 0);
    //env->ReleaseStringUTFChars(group, sGroup);
    return 0;
}

/*
 * Class:     faiss_FaissJni
 * Method:    search
 * Signature: (II)V
 */
JNIEXPORT jint JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_search
(JNIEnv *env, jobject obj, jstring group, jint n, jfloatArray x, jint k, jfloatArray distances, jlongArray labels) {
    string sGroup = env->GetStringUTFChars(group, 0);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr)
        return -1;
    jfloat* pxq = env->GetFloatArrayElements(x, JNI_FALSE);
    jfloat* pres = env->GetFloatArrayElements(distances, JNI_FALSE);
    jlong* plindex = env->GetLongArrayElements(labels, JNI_FALSE);

    pindex->search(n, pxq, k, pres, plindex);
    env->ReleaseFloatArrayElements(x, pxq, 0);
    env->ReleaseFloatArrayElements(distances, pres, 0);
    env->ReleaseLongArrayElements(labels, plindex, 0);
    //env->ReleaseStringUTFChars(group, sGroup);
    return 0;
}

/*
 * Class:     faiss_FaissJni
 * Method:    is_trained
 * Signature: (II)V
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_isTrained
        (JNIEnv *env, jobject obj, jstring group) {
    string sGroup = env->GetStringUTFChars(group, 0);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr)
        return false;
    return pindex->is_trained;
}