#include "faiss_jni.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <jni.h>
#include <Index.h>
#include <IndexFlat.h>
#include <vector>

#include <fstream>

using namespace std;
using namespace faiss;
namespace
{
    static Index *indexInterface;
}

/*
 * Class:     faiss_FaissJni
 * Method:    cppCtor
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_cppCtor
(JNIEnv *env, jobject obj, jint dim, jint index_type) {
switch(index_type) {
case 0:
indexInterface = new IndexFlat(dim);
case 1:
indexInterface = new IndexFlatL2(dim);
}
std::cout << ":)!!! jint=" << dim <<std::endl;
std::cout << "index_type"<< index_type <<std::endl;
}

/*
 * Class:     faiss_FaissJni
 * Method:    add
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_add
(JNIEnv *env, jobject obj, jint n, jfloatArray x) {
jfloat* pbuffer = env->GetFloatArrayElements(x, JNI_FALSE);
indexInterface->add(n, pbuffer);
env->ReleaseFloatArrayElements(x, pbuffer, 0);
}

/*
 * Class:     faiss_FaissJni
 * Method:    search
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_search
(JNIEnv *env, jobject obj, jint n, jfloatArray x, jint k, jfloatArray distances, jlongArray labels) {
jfloat* pxq = env->GetFloatArrayElements(x, JNI_FALSE);
jfloat* pres = env->GetFloatArrayElements(distances, JNI_FALSE);
jlong* plindex = env->GetLongArrayElements(labels, JNI_FALSE);

indexInterface->search(n, pxq, k, pres, plindex);
env->ReleaseFloatArrayElements(x, pxq, 0);
env->ReleaseFloatArrayElements(distances, pres, 0);
env->ReleaseLongArrayElements(labels, plindex, 0);
}

/*
 * Class:     faiss_FaissJni
 * Method:    is_trained
 * Signature: (II)V
 */
JNIEXPORT jboolean JNICALL Java_com_xiaomi_chatbot_services_faiss_model_FaissJNI_is_1trained
        (JNIEnv *env, jobject obj) {
return indexInterface->is_trained;
}