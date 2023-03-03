/*
 *   Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

#include "com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex.h"

#include "init.h"
#include "index.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"

using std::vector;

using similarity::initLibrary;
using similarity::AnyParams;
using similarity::Index;
using similarity::MethodFactoryRegistry;
using similarity::SpaceFactoryRegistry;
using similarity::AnyParams;
using similarity::Space;
using similarity::ObjectVector;
using similarity::Object;
using similarity::KNNQuery;
using similarity::KNNQueue;

extern "C"

const similarity::LabelType DEFAULT_LABEL = -1;

struct IndexWrapper {
  IndexWrapper(string spaceType) {
    space.reset(SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceType, AnyParams()));
    index.reset(MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceType, *space, data));
  }
  std::unique_ptr<Space<float>> space;
  std::unique_ptr<Index<float>> index;
  // Index gets constructed with a reference to data (see above) but is otherwise unused
  ObjectVector data;
};

struct JavaException {
    JavaException(JNIEnv* env, const char* type = "", const char* message = "")
    {
        jclass newExcCls = env->FindClass(type);
        if (newExcCls != NULL)
            env->ThrowNew(newExcCls, message);
    }
};

inline void has_exception_in_stack(JNIEnv* env)
{
    if (env->ExceptionCheck() == JNI_TRUE)
        throw std::runtime_error("Exception Occured");
}

void catch_cpp_exception_and_throw_java(JNIEnv* env)
{
    try {
        throw;
    }
    catch (const std::bad_alloc& rhs) {
        JavaException(env, "java/io/IOException", rhs.what());
    }
    catch (const std::runtime_error& re) {
        JavaException(env, "java/lang/Exception", re.what());
    }
    catch (const std::exception& e) {
        JavaException(env, "java/lang/Exception", e.what());
    }
    catch (...) {
        JavaException(env, "java/lang/Exception", "Unknown exception occured");
    }
}

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_saveIndex(JNIEnv* env, jclass cls, jintArray ids, jobjectArray vectors, jstring indexPath, jobjectArray algoParams, jstring spaceType)
{
    Space<float>* space = NULL;
    ObjectVector dataset;
    Index<float>* index = NULL;
    int* object_ids = NULL;

    try {
        const char *spaceTypeCStr = env->GetStringUTFChars(spaceType, 0);
        string spaceTypeString(spaceTypeCStr);
        env->ReleaseStringUTFChars(spaceType, spaceTypeCStr);
        has_exception_in_stack(env);
        space = SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceTypeString, AnyParams());
        object_ids = env->GetIntArrayElements(ids, 0);

        int numVectors = env->GetArrayLength(vectors);
        int dim = 0;
        if (numVectors > 0) {
             auto vectorArray = (jfloatArray)env->GetObjectArrayElement(vectors, 0);
             dim = env->GetArrayLength(vectorArray);
             if (env->ExceptionCheck() == JNI_TRUE) {
                 throw std::runtime_error("Exception occurred in JNI when retrieving dimension information");
             }
         }
        size_t vectorSizeInBytes = dim*sizeof(float);

        // Allocate a large buffer that will contain all the vectors. Allocating the objects in one large buffer as
        // opposed to individually will prevent heap fragmentation. We have observed that allocating individual
        // objects causes RSS to rise throughout the lifetime of a process. This is because, in typical systems, small
        // allocations will reside on some kind of heap managed by an allocator. Once freed, the allocator does not
        // always return the memory to the operating system. If the heap gets fragmented, this will cause the allocator
        // to ask for more memory, causing RSS to grow. On large allocations (> 128 kb), most allocators will
        // internally use mmap. Once freed, unmap will be called, which will immediately return memory to the OS
        // which in turn prevents RSS from growing out of control. Wrap with a smart pointer so that buffer will be
        // freed once variable goes out of scope. For reference, the code that specifies the layout of the buffer can be
        // found: https://github.com/nmslib/nmslib/blob/v2.1.1/similarity_search/include/object.h#L61-L75
        std::unique_ptr<char[]> objectBuffer(new char[(similarity::ID_SIZE + similarity::LABEL_SIZE + similarity::DATALENGTH_SIZE + vectorSizeInBytes) * numVectors]);
        char* ptr = objectBuffer.get();

        for (int i = 0; i < numVectors; i++) {
            dataset.push_back(new similarity::Object(ptr));

            // Copy object metadata
            memcpy(ptr, &object_ids[i], similarity::ID_SIZE);
            ptr += similarity::ID_SIZE;
            memcpy(ptr, &DEFAULT_LABEL, similarity::LABEL_SIZE);
            ptr += similarity::LABEL_SIZE;
            memcpy(ptr, &vectorSizeInBytes, similarity::DATALENGTH_SIZE);
            ptr += similarity::DATALENGTH_SIZE;

            // Copy vector to data location in object buffer
            jfloatArray vectorArray = (jfloatArray)env->GetObjectArrayElement(vectors, i);
            float* vector = env->GetFloatArrayElements(vectorArray, 0);
            memcpy(ptr, vector, vectorSizeInBytes);
            env->ReleaseFloatArrayElements(vectorArray, vector, JNI_ABORT);
            ptr += vectorSizeInBytes;
        }
        // free up memory
        env->ReleaseIntArrayElements(ids, object_ids, 0);
        index = MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceTypeString, *space, dataset);

        int paramsCount = env->GetArrayLength(algoParams);
        vector<string> paramsList;
        for (int i=0; i<paramsCount; i++) {
            jstring param = (jstring) (env->GetObjectArrayElement(algoParams, i));
            const char *rawString = env->GetStringUTFChars(param, 0);
            paramsList.push_back(rawString);
            env->ReleaseStringUTFChars(param, rawString);
        }

        index->CreateIndex(AnyParams(paramsList));
        has_exception_in_stack(env);
        const char *indexString = env->GetStringUTFChars(indexPath, 0);
        index->SaveIndex(indexString);
        env->ReleaseStringUTFChars(indexPath, indexString);
        has_exception_in_stack(env);

        // Free each object in the dataset. No need to clear the vector because it goes out of scope
        // immediately
        for (auto it = dataset.begin(); it != dataset.end(); it++) {
             delete *it;
        }
        delete index;
        delete space;
    }
    catch (...) {
        if (object_ids) { env->ReleaseIntArrayElements(ids, object_ids, 0); }
        for (auto it = dataset.begin(); it != dataset.end(); it++) {
             delete *it;
        }
        if (index) { delete index; }
        if (space) { delete space; }
        catch_cpp_exception_and_throw_java(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_queryIndex(JNIEnv* env, jclass cls, jlong indexPointer, jfloatArray queryVector, jint k)
{
    try {
        IndexWrapper *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointer);

        float* rawQueryvector = env->GetFloatArrayElements(queryVector, 0);
        std::unique_ptr<const Object> queryObject(new Object(-1, -1, env->GetArrayLength(queryVector)*sizeof(float), rawQueryvector));
        env->ReleaseFloatArrayElements(queryVector, rawQueryvector, 0);
        has_exception_in_stack(env);

        KNNQuery<float> knnQuery(*(indexWrapper->space), queryObject.get(), k);
        indexWrapper->index->Search(&knnQuery);
        std::unique_ptr<KNNQueue<float>> result(knnQuery.Result()->Clone());
        has_exception_in_stack(env);
        int resultSize = result->Size();
        jclass resultClass = env->FindClass("com/amazon/opendistroforelasticsearch/knn/index/KNNQueryResult");
        jmethodID allArgs = env->GetMethodID(resultClass, "<init>", "(IF)V");
        jobjectArray results = env->NewObjectArray(resultSize, resultClass, NULL);
        for (int i = 0; i < resultSize; i++) {
            float distance = result->TopDistance();
            long id = result->Pop()->id();
            env->SetObjectArrayElement(results, i, env->NewObject(resultClass, allArgs, id, distance));
        }
        has_exception_in_stack(env);
        return results;
    } catch(...) {
        catch_cpp_exception_and_throw_java(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_init(JNIEnv* env, jclass cls,  jstring indexPath, jobjectArray algoParams, jstring spaceType)
{
    IndexWrapper *indexWrapper = NULL;
    try {
        const char *indexPathCStr = env->GetStringUTFChars(indexPath, 0);
        string indexPathString(indexPathCStr);
        env->ReleaseStringUTFChars(indexPath, indexPathCStr);
        has_exception_in_stack(env);

        // Load index from file (may throw)
        const char *spaceTypeCStr = env->GetStringUTFChars(spaceType, 0);
        string spaceTypeString(spaceTypeCStr);
        env->ReleaseStringUTFChars(spaceType, spaceTypeCStr);
        has_exception_in_stack(env);
        IndexWrapper *indexWrapper = new IndexWrapper(spaceTypeString);
        indexWrapper->index->LoadIndex(indexPathString);

        // Parse and set query params
        int paramsCount = env->GetArrayLength(algoParams);
        vector<string> paramsList;
        for (int i=0; i<paramsCount; i++) {
            jstring param = (jstring) (env->GetObjectArrayElement(algoParams, i));
            const char *rawString = env->GetStringUTFChars(param, 0);
            paramsList.push_back(rawString);
            env->ReleaseStringUTFChars(param, rawString);
        }
        indexWrapper->index->SetQueryTimeParams(AnyParams(paramsList));
        has_exception_in_stack(env);

        return (jlong) indexWrapper;
    }
    // nmslib seems to throw std::runtime_error if the index cannot be read (which
    // is the only known failure mode for init()).
    catch (...) {
        if (indexWrapper) delete indexWrapper;
        catch_cpp_exception_and_throw_java(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_gc(JNIEnv* env, jclass cls,  jlong indexPointer)
{
    try {
        IndexWrapper *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointer);
        has_exception_in_stack(env);
        delete indexWrapper;
        has_exception_in_stack(env);
    }
    catch (...) {
        catch_cpp_exception_and_throw_java(env);
    }
}

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_initLibrary(JNIEnv *, jclass)
{
    initLibrary();

}
