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
(JNIEnv *env, jobject obj, jstring group, jint dim) {
    const char *cstr = env->GetStringUTFChars(group, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    pinstance->DelMapKeyFlatL2(sGroup);
    pinstance->SetMapKeyFlatL2(sGroup, dim);
    env->ReleaseStringUTFChars(group, cstr);
}

/*
 * Class:     faiss_FaissJni
 * Method:    add
 * Signature: (II)V
 */
JNIEXPORT jint JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_add
(JNIEnv *env, jobject obj, jstring group, jint n, jfloatArray x) {
    const char *cstr = env->GetStringUTFChars(group, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr)
        return -1;
    jfloat* pbuffer = env->GetFloatArrayElements(x, JNI_FALSE);
    pindex->add(n, pbuffer);
    env->ReleaseFloatArrayElements(x, pbuffer, 0);
    env->ReleaseStringUTFChars(group, cstr);
    return 0;
}

JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_setMapKeyKB
        (JNIEnv *env, jobject obj, jstring key, jobject kb) {
    const char *cstr = env->GetStringUTFChars(key, NULL);
    string sGroup = std::string(cstr);

    map<string, string> cmap;
    jclass jmapclass = env->FindClass("java/util/HashMap");
    jmethodID jkeysetmid = env->GetMethodID(jmapclass, "keySet", "()Ljava/util/Set;");
    jmethodID jgetmid = env->GetMethodID(jmapclass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
    jobject jsetkey = env->CallObjectMethod(kb,jkeysetmid);
    jclass jsetclass = env->FindClass("java/util/Set");
    jmethodID jtoArraymid = env->GetMethodID(jsetclass, "toArray", "()[Ljava/lang/Object;");
    jobjectArray jobjArray = (jobjectArray)env->CallObjectMethod(jsetkey,jtoArraymid);
    if(jobjArray == nullptr) {
        env->ReleaseStringUTFChars(key, cstr);
        return true;
    }
    jsize arraysize = env->GetArrayLength(jobjArray);
    for (int i = 0; i < arraysize; i++) {
        jstring jkey = (jstring)env->GetObjectArrayElement(jobjArray, i);
        jstring jvalue = (jstring)env->CallObjectMethod(kb, jgetmid, jkey);
        string mkey = (char*)env->GetStringUTFChars(jkey,0);
        string mvalue = (char*)env->GetStringUTFChars(jvalue,0);
        cmap[mkey] = mvalue;
    }
    IndexManager* pinstance = IndexManager::CreateInstance();
    bool status = pinstance->SetMapKeyKB(sGroup, cmap);
    env->ReleaseStringUTFChars(key, cstr);
    return status;
}

/*
 * Class:     com_xiaomi_chatbot_services_faiss_model_FaissJNI
 * Method:    delMapKeyKB
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_delMapKeyKB
        (JNIEnv *env, jobject obj, jstring key) {
    const char *cstr = env->GetStringUTFChars(key, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    bool status = pinstance->DelMapKeyKB(sGroup);
    env->ReleaseStringUTFChars(key, cstr);
    return status;
}

/*
 * Class:     com_xiaomi_chatbot_services_faiss_model_FaissJNI
 * Method:    setListKeyKB
 * Signature: (Ljava/lang/String;Ljava/util/List;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_setListKeyKB
        (JNIEnv *env, jobject obj, jstring key, jobject kb) {
    const char *cstr = env->GetStringUTFChars(key, NULL);
    string sGroup = std::string(cstr);

    jclass clsArrayList = env->GetObjectClass(kb);
    jmethodID arrayListGet = env->GetMethodID(clsArrayList, "get", "(I)Ljava/lang/Object;");
    jmethodID arrayListSize = env->GetMethodID(clsArrayList, "size", "()I");
    jint len = env->CallIntMethod(kb, arrayListSize);
    vector<string> cvec;
    for (int i = 0; i < len; i++) {
        jstring kbText = (jstring)env->CallObjectMethod(kb, arrayListGet, i);
        cvec.push_back((char*)env->GetStringUTFChars(kbText, nullptr));
    }
    IndexManager* pinstance = IndexManager::CreateInstance();
    bool status = pinstance->SetListKeyKB(sGroup, cvec);
    env->ReleaseStringUTFChars(key, cstr);
    return status;
}

/*
 * Class:     com_xiaomi_chatbot_services_faiss_model_FaissJNI
 * Method:    delListKeyKB
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_delListKeyKB
        (JNIEnv *env, jobject obj, jstring key) {
    const char *cstr = env->GetStringUTFChars(key, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    bool status = pinstance->DelListKeyKB(sGroup);
    env->ReleaseStringUTFChars(key, cstr);
    return status;
}

/*
 * Class:     faiss_FaissJni
 * Method:    search
 * Signature: (II)V
 */
JNIEXPORT jint JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_search
(JNIEnv *env, jobject obj, jstring group, jint n, jfloatArray x, jint k, jfloatArray distances, jobjectArray kbs) {
    const char *cstr = env->GetStringUTFChars(group, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr) {
        env->ReleaseStringUTFChars(group, cstr);
        return -1;
    }
    jfloat* pxq = env->GetFloatArrayElements(x, JNI_FALSE);
    jfloat* pres = env->GetFloatArrayElements(distances, JNI_FALSE);
    int stringCount = env->GetArrayLength(kbs);
    jlong* index = new jlong[stringCount];
    pindex->search(n, pxq, k, pres, index);

    for (int i = 0; i < stringCount; i++) {
        int kbIndex = (int)index[i];
        string kbText = pinstance->GetKBStanQ(sGroup, kbIndex);
        env->SetObjectArrayElement(kbs, i, env->NewStringUTF(kbText.c_str()));
    }
    env->ReleaseFloatArrayElements(x, pxq, 0);
    env->ReleaseFloatArrayElements(distances, pres, 0);
    env->ReleaseStringUTFChars(group, cstr);
    delete index;
    return 0;
}

/*
 * Class:     faiss_FaissJni
 * Method:    is_trained
 * Signature: (II)V
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_isTrained
        (JNIEnv *env, jobject obj, jstring group) {
    const char *cstr = env->GetStringUTFChars(group, NULL);
    string sGroup = std::string(cstr);
    IndexManager* pinstance = IndexManager::CreateInstance();
    faiss::IndexFlatL2* pindex = pinstance->GetMapKeyFlatL2(sGroup);
    if (pindex == nullptr) {
        env->ReleaseStringUTFChars(group, cstr);
        return false;
    }
    bool status = pindex->is_trained;
    env->ReleaseStringUTFChars(group, cstr);
    return status;
}