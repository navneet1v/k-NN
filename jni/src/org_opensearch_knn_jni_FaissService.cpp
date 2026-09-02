/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include "org_opensearch_knn_jni_FaissService.h"

#include <jni.h>

#include <vector>
#include <algorithm>
#include <queue>

#include "faiss_wrapper.h"
#include "jni_util.h"
#include "faiss_stream_support.h"
#include "faiss/impl/FaissException.h"
#include "faiss_index_service.h"
#include "sq/faiss_sq_hnsw.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_FAISS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env, vm);

    return KNN_FAISS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION);
    faiss::InterruptCallback::instance.get()->clear_instance();
    jniUtil.Uninitialize(env);
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initIndex(JNIEnv * env, jclass cls,
                                                                           jlong numDocs, jint dimJ,
                                                                           jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &indexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jlong numDocs, jint dimJ,
                                                                                 jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &binaryIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initByteIndex(JNIEnv * env, jclass cls,
                                                                               jlong numDocs, jint dimJ,
                                                                               jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &byteIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                              jlong vectorsAddressJ, jint dimJ,
                                                                              jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &indexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToBinaryIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                    jlong vectorsAddressJ, jint dimJ,
                                                                                    jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &binaryIndexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToByteIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                  jlong vectorsAddressJ, jint dimJ,
                                                                                  jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &byteIndexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeIndex(JNIEnv * env,
                                                                           jclass cls,
                                                                           jlong indexAddress,
                                                                           jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &indexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeBinaryIndex(JNIEnv * env,
                                                                                 jclass cls,
                                                                                 jlong indexAddress,
                                                                                 jobject output,
                                                                                 jboolean skipFlat)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &binaryIndexService, skipFlat);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeByteIndex(JNIEnv * env,
                                                                               jclass cls,
                                                                               jlong indexAddress,
                                                                               jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &byteIndexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(JNIEnv * env,
                                                                                        jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jlong vectorsAddressJ,
                                                                                        jint dimJ,
                                                                                        jobject output,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndexFromTemplate(&jniUtil,
                                                        env,
                                                        idsJ,
                                                        vectorsAddressJ,
                                                        dimJ,
                                                        output,
                                                        templateIndexJ,
                                                        parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate(JNIEnv * env,
                                                                                              jclass cls,
                                                                                              jintArray idsJ,
                                                                                              jlong vectorsAddressJ,
                                                                                              jint dimJ,
                                                                                              jobject output,
                                                                                              jbyteArray templateIndexJ,
                                                                                              jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateBinaryIndexFromTemplate(&jniUtil,
                                                              env,
                                                              idsJ,
                                                              vectorsAddressJ,
                                                              dimJ,
                                                              output,
                                                              templateIndexJ,
                                                              parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createByteIndexFromTemplate(JNIEnv * env,
                                                                                            jclass cls,
                                                                                            jintArray idsJ,
                                                                                            jlong vectorsAddressJ,
                                                                                            jint dimJ,
                                                                                            jobject output,
                                                                                            jbyteArray templateIndexJ,
                                                                                            jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateByteIndexFromTemplate(&jniUtil,
                                                            env,
                                                            idsJ,
                                                            vectorsAddressJ,
                                                            dimJ,
                                                            output,
                                                            templateIndexJ,
                                                            parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
  try {
      return knn_jni::faiss_wrapper::LoadIndex(&jniUtil, env, indexPathJ);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStream(JNIEnv * env,
                                                                                     jclass cls,
                                                                                     jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadIndexWithStream(
                 &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
    try {
        return knn_jni::faiss_wrapper::LoadBinaryIndex(&jniUtil, env, indexPathJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndexWithStream(JNIEnv * env,
                                                                                           jclass cls,
                                                                                           jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadBinaryIndexWithStream(
            &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStreamADCParams
(JNIEnv * env, jclass cls, jobject readStreamJ, jobject parametersJ) {
    try {
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStreamJ};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        return knn_jni::faiss_wrapper::LoadIndexWithStreamADCParams(&faissOpenSearchIOReader, &jniUtil, env, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired(JNIEnv * env,
                                                                                               jclass cls,
                                                                                               jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::IsSharedIndexStateRequired(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::InitSharedIndexState(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::SetSharedIndexState(indexPointerJ, shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, parentIdsJ);

    }
    catch (const faiss::FaissException& e) {
        std::cout << "====> query get faiss exception:" << e.what() << "\n";
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jbyteArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free(JNIEnv * env, jclass cls, jlong indexPointerJ, jboolean isBinaryIndexJ)
{
    try {
        return knn_jni::faiss_wrapper::Free(indexPointerJ, isBinaryIndexJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
        (JNIEnv * env, jclass cls, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::FreeSharedIndexState(shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary(JNIEnv * env, jclass cls)
{
    try {
        knn_jni::faiss_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainBinaryIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainByteIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainByteIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors(JNIEnv * env, jclass cls,
                                                                                 jlong vectorsPointerJ,
                                                                                 jobjectArray vectorsJ)
{
    std::vector<float> *vect;
    if ((long) vectorsPointerJ == 0) {
        vect = new std::vector<float>;
    } else {
        vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
    }

    int dim = jniUtil.GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil.Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);
    vect->insert(vect->begin(), dataset.begin(), dataset.end());

    return (jlong) vect;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex(JNIEnv * env, jclass cls,
                                                                                         jlong indexPointerJ,
                                                                                         jfloatArray queryVectorJ,
                                                                                         jfloat radiusJ, jobject methodParamsJ,
                                                                                         jint maxResultWindowJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearch(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter(JNIEnv * env, jclass cls,
                                                                                                   jlong indexPointerJ,
                                                                                                   jfloatArray queryVectorJ,
                                                                                                   jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ,
                                                                                                   jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearchWithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, filterIdsJ, filterIdsTypeJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setMergeInterruptCallback(JNIEnv * env, jclass cls)
{
    try {
        faiss::InterruptCallback::instance.reset(
            new knn_jni::faiss_wrapper::OpenSearchMergeInterruptCallback(&jniUtil)
        );
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initFaissSQIndex(
    JNIEnv * env, jclass cls, jint totalLiveDocs, jint dimJ, jobject parametersJ, jfloat centroidDp, jint quantizedVecBytes) {
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitFaissSQIndex(&jniUtil, env, totalLiveDocs, dimJ, parametersJ, &binaryIndexService, centroidDp, quantizedVecBytes);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_addDocsToSQIndex(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress, jintArray docIdsJ, jint numDocs, jint numAdded) {

    try {
        // Grab Faiss SQ stuff
        auto binaryIdMap = (faiss::IndexBinaryIDMap*) indexMemoryAddress;
        auto faissSQHnsw = (knn_jni::FaissSQHnsw*) binaryIdMap->index;
        auto faissSQFlat = (knn_jni::FaissSQFlat*) faissSQHnsw->storage;

        // Allocate in stack
        int64_t docIds[numDocs];

        // Copy docs : int32_t -> int64_t
        jint* docIdsPtr = static_cast<jint*>(env->GetPrimitiveArrayCritical(docIdsJ, nullptr));
        for (int32_t i = 0 ; i < numDocs; ++i) {
            docIds[i] = docIdsPtr[i];
        }
        env->ReleasePrimitiveArrayCritical(docIdsJ, docIdsPtr, 0);

        // Get the next batch of vectors to be added to HNSW
        // Note that we've already inserted quantized vectors in storage, just that it's not visible yet.
        auto* vecPtr = faissSQFlat->quantizedVectorsAndCorrectionFactors.data()
                       + (numAdded * faissSQFlat->oneElementSize);

        // Keep building HNSW
        binaryIdMap->add_with_ids(numDocs, vecPtr, &docIds[0]);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_passSQVectorsWithCorrectionFactors(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress, jbyteArray buffer, jint numElements) {

    try {
        // Grab Faiss SQ stuff
        auto binaryIdMap = (faiss::IndexBinaryIDMap*) indexMemoryAddress;
        auto faissSQHnsw = (knn_jni::FaissSQHnsw*) binaryIdMap->index;
        auto faissSQFlat = (knn_jni::FaissSQFlat*) faissSQHnsw->storage;

        // Start copying
        jbyte* jb = static_cast<jbyte*>(env->GetPrimitiveArrayCritical(buffer, nullptr));
        knn_jni::JNIReleaseElements release([=]{
            env->ReleasePrimitiveArrayCritical(buffer, jb, 0);
        });

        // This does not involve memory doubling, we already allocated required memory space
        faissSQFlat->quantizedVectorsAndCorrectionFactors.insert(
            faissSQFlat->quantizedVectorsAndCorrectionFactors.end(),
            reinterpret_cast<uint8_t*>(jb),
            reinterpret_cast<uint8_t*>(jb) + numElements * faissSQFlat->oneElementSize
        );
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_releaseFaissSQIndex(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress) {
    try {
        auto* binaryIdMap = reinterpret_cast<faiss::IndexBinaryIDMap*>(indexMemoryAddress);
        delete binaryIdMap;
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setFaissSQHnswToSingleLayer(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress) {
    try {
        if (indexMemoryAddress == 0) {
            throw std::runtime_error("setFaissSQHnswToSingleLayer: null index address");
        }
        // Reinterpret to the base IndexBinary* (valid: the address is an IndexBinaryIDMap, which
        // IS-A IndexBinary), then dynamic_cast each level so a wrong/unexpected index type throws a
        // clear error instead of being blindly reinterpreted.
        auto* indexBinary = reinterpret_cast<faiss::IndexBinary*>(indexMemoryAddress);
        auto* binaryIdMap = dynamic_cast<faiss::IndexBinaryIDMap*>(indexBinary);
        if (binaryIdMap == nullptr) {
            throw std::runtime_error("setFaissSQHnswToSingleLayer: index is not a faiss::IndexBinaryIDMap");
        }
        auto* faissSQHnsw = dynamic_cast<knn_jni::FaissSQHnsw*>(binaryIdMap->index);
        if (faissSQHnsw == nullptr) {
            throw std::runtime_error("setFaissSQHnswToSingleLayer: wrapped index is not a knn_jni::FaissSQHnsw");
        }
        faiss::HNSW & hnsw = faissSQHnsw->hnsw;

        // Must run before any vectors are added: random_level() is consumed during add_with_ids, and
        // the neighbor storage is laid out per the multi-layer cum_nneighbor_per_level at add time.
        if (faissSQHnsw->ntotal != 0) {
            throw std::runtime_error(
                "setFaissSQHnswToSingleLayer must be called before adding vectors (ntotal="
                + std::to_string((int64_t) faissSQHnsw->ntotal) + ")");
        }

        // Force every node to level 0: with assign_probas = {1.0}, random_level() returns 0 for any
        // draw, so no upper layers are ever created. Keep the layer-0 neighbor width (2*M) the
        // constructor already computed (cum_nneighbor_per_level[1]); drop all higher levels.
        const int layer0Width = hnsw.cum_nneighbor_per_level.at(1);
        hnsw.assign_probas = {1.0};
        hnsw.cum_nneighbor_per_level = {0, layer0Width};
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

// ============================================================================================
// buildOrderingOfVectorsUsingIndexStructure — page-locality-aware greedy vector ordering
// ============================================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
// -----------------------------
// The quantized vectors are stored on disk one record per "slot", read as base + slot*recordSize.
// The OS/SSD reads whole pages (~32 KB). During a graph search we hop from a node to its HNSW
// neighbors; if a node and its neighbors sit on DIFFERENT pages, every hop faults in a new page.
// If instead we lay neighbors out on the SAME page, one page read serves many hops → far fewer
// page faults, higher throughput. This function decides that on-disk layout: it assigns every
// vector a physical slot such that graph-adjacent vectors tend to share a page.
//
// TERMINOLOGY
// -----------
//   originalOrdinal  : a vector's id in the graph / insertion order (0..n-1). Faiss uses these as
//                      node ids in the layer-0 adjacency.
//   physicalPosition : the slot the vector's record occupies on disk (its position in the reordered
//                      file). This is what we compute.
//   cluster          : one contiguous run of up to `pageCapacity` records grown from a seed. It is the
//                      target size of a 32 KB SSD page (pageCapacity = pageBytes / recordBytes) and
//                      bounds how large a single dense pocket may grow before we seed a new one.
//   affinity(node)   : while a cluster is being filled, how many of `node`'s neighbors are ALREADY on
//                      that cluster. High affinity ⇒ placing `node` here keeps many of its edges local.
//
// THE ALGORITHM (greedy page-growing, hub-seeded — mirrors greedy_page_growing_permutation in
// scripts/locality_simulation.py)
// ---------------------------------------------------------------------------------------------
//   1. Compute each node's layer-0 out-degree.
//   2. Sort nodes by degree, descending (ties by id). The highest-degree nodes are graph "hubs" —
//      the centers of the densest clusters — so we seed pages from them first.
//   3. For each still-unassigned hub, GROW A CLUSTER:
//        a. Put the seed on the cluster; assign it the next physical slot.
//        b. Its unassigned neighbors become "candidates", each with affinity = 1.
//        c. Repeatedly take the candidate with the HIGHEST affinity, add it (next slot),
//           and bump the affinity of ITS unassigned neighbors (adding new candidates as they appear).
//        d. Stop when the cluster holds `pageCapacity` nodes OR no candidate remains (a pocket smaller
//           than a page — "close early").
//        e. DENSE PACKING: do NOT pad. The next cluster starts at the very next slot, so records are a
//           contiguous 0..n-1 with no gaps. A cluster may straddle a physical 32 KB boundary (vs the
//           deferred strict-page variant which padded each cluster to its own page), but the file stays
//           exactly n records — far fewer physical pages when clusters (~avg degree) ≪ pageCapacity.
//   4. After all hubs, a safety pass seeds any leftover node (there are none in a connected graph,
//      but degree-0/unreachable nodes get their own singleton cluster here).
//
// PICKING THE BEST CANDIDATE FAST — the "bucket queue"
// ----------------------------------------------------
// Naively, step 3c is "scan all candidates for the max affinity" = O(candidates) per pick = too slow.
// Instead we keep `buckets[k]` = a list of candidates whose affinity == k, plus `topBucket` = the
// highest non-empty bucket index. Picking the best is: look in buckets[topBucket]; if empty, walk
// topBucket down. Affinity only ever INCREASES (we never remove a neighbor from a page mid-grow), so
// when a candidate's affinity goes k→k+1 we just append it to buckets[k+1] and leave the old entry in
// buckets[k] as a STALE duplicate. popBest() discards stale entries by re-checking, when it pops a
// node, that the node is still a candidate AND its current affinity equals the bucket it came from.
// This gives amortized O(1) selection with O(degree) work per placed node.
//
// OUTPUT / CONTRACT
// -----------------
//   ordering[originalOrdinal] = physicalPosition   (the FORWARD map; length n; bijective, DENSE 0..n-1)
// The forward map is what the reader's ordToPhysicalOrdMap needs. Physical positions are DENSE (a
// contiguous 0..n-1 with no padding gaps), the same shape the BFS variant emits — so the locality
// writer needs no special sparse/padding handling; the greedy clustering just yields a better layout.
//
// PERFORMANCE / MEMORY
// --------------------
// Runs entirely on the in-place native adjacency (zero copy — the 2*N*M neighbor array never crosses
// JNI). Scratch is O(n): degree[n], assigned[n] (1 bit), isCandidate[n] (1 bit), affinity[n],
// buckets (≈maxDegree small vectors), plus the queue-like page/candidate lists. Time is ~O(n + edges)
// with small constants (each edge is walked a few times: degree pass, page growth, per-page cleanup).
// ============================================================================================
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_buildOrderingOfVectorsUsingIndexStructure(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress, jintArray orderingJ, jint pageCapacityJ, jintArray hubsJ) {
    try {
        if (indexMemoryAddress == 0) {
            throw std::runtime_error("buildOrderingOfVectorsUsingIndexStructure: null index address");
        }
        // Same defensive down-cast chain as setFaissSQHnswToSingleLayer: a wrong index type throws
        // instead of being blindly reinterpreted.
        auto* indexBinary = reinterpret_cast<faiss::IndexBinary*>(indexMemoryAddress);
        auto* binaryIdMap = dynamic_cast<faiss::IndexBinaryIDMap*>(indexBinary);
        if (binaryIdMap == nullptr) {
            throw std::runtime_error("buildOrderingOfVectorsUsingIndexStructure: index is not a faiss::IndexBinaryIDMap");
        }
        auto* faissSQHnsw = dynamic_cast<knn_jni::FaissSQHnsw*>(binaryIdMap->index);
        if (faissSQHnsw == nullptr) {
            throw std::runtime_error("buildOrderingOfVectorsUsingIndexStructure: wrapped index is not a knn_jni::FaissSQHnsw");
        }
        faiss::HNSW & hnsw = faissSQHnsw->hnsw;

        const int pageCapacity = (int) pageCapacityJ;
        if (pageCapacity < 1) {
            throw std::runtime_error(
                "buildOrderingOfVectorsUsingIndexStructure: pageCapacity must be >= 1, got "
                + std::to_string(pageCapacity));
        }

        // ordering is the FORWARD permutation: ordering[originalOrdinal] = physicalPosition, one slot
        // per vector, so its length must equal ntotal. Fail loudly on any mismatch — a wrong-length
        // permutation would corrupt the reordered store. The 64-bit compare avoids truncation if
        // ntotal somehow exceeds INT_MAX (a Java int[] can't, so this can only be a mismatch).
        // Note: physical positions are STRICT-PAGE aligned and therefore SPARSE — an early-closed page
        // pads to the next pageCapacity boundary, so the max physical position is (numPages*pageCapacity
        // - 1), which can exceed n. The forward map itself is still gap-free and length n.
        const jsize outLen = env->GetArrayLength(orderingJ);
        if ((int64_t) outLen != (int64_t) faissSQHnsw->ntotal) {
            throw std::runtime_error(
                "buildOrderingOfVectorsUsingIndexStructure: ordering length " + std::to_string((int64_t) outLen)
                + " != ntotal (" + std::to_string((int64_t) faissSQHnsw->ntotal) + ")");
        }
        // n == ntotal; node ids and positions are int32 (Faiss storage_idx_t) and bounded by the Java
        // int[] length, so int is sufficient. Only the flat-neighbor edge offsets need to be wider.
        const int n = (int) outLen;
        if (n == 0) {
            return;
        }

        // Reads a node's valid layer-0 neighbors in place (zero-copy over hnsw.neighbors). Neighbor ids
        // are bounds-checked (>=0 and <n) so a corrupt graph can't index out of range.
        auto forEachNeighbor = [&](int node, auto&& fn) {
            size_t begin = 0, end = 0;
            hnsw.neighbor_range(node, 0, &begin, &end);
            for (size_t e = begin; e < end; ++e) {
                const int nb = hnsw.neighbors[e];
                if (nb >= 0 && nb < n) {
                    fn(nb);
                }
            }
        };

        // Per-node layer-0 out-degree. Used to seed pages from the highest-degree "hub" nodes first —
        // the most-connected nodes anchor the densest clusters (best page seeds).
        std::vector<int> degree(n, 0);
        int maxDegree = 0;
        for (int i = 0; i < n; ++i) {
            int d = 0;
            forEachNeighbor(i, [&](int) { ++d; });
            degree[i] = d;
            if (d > maxDegree) {
                maxDegree = d;
            }
        }

        // Hub seed order: highest degree first, ties broken by ascending node id for determinism.
        std::vector<int> hubs(n);
        for (int i = 0; i < n; ++i) {
            hubs[i] = i;
        }
        std::sort(hubs.begin(), hubs.end(), [&degree](int a, int b) {
            if (degree[a] != degree[b]) {
                return degree[a] > degree[b];
            }
            return a < b;
        });

        // Export the top hubs as search entry-point candidates. The caller sizes hubsJ (at most ~32);
        // we fill min(hubsLen, n) entries from the front of the degree-sorted `hubs` list — i.e. the
        // highest-degree node ids — and pad any leftover slots with -1. Values are ORIGINAL ordinals
        // (graph node ids), the space graph-search entry points live in. Filled in its own critical
        // section that ends before we pin the ordering array below (never two pinned arrays at once).
        if (hubsJ != nullptr) {
            const jsize hubsLen = env->GetArrayLength(hubsJ);
            const int numHubs = std::min((int) hubsLen, n);
            jint* hubOut = static_cast<jint*>(env->GetPrimitiveArrayCritical(hubsJ, nullptr));
            knn_jni::JNIReleaseElements hubRelease([=]{ env->ReleasePrimitiveArrayCritical(hubsJ, hubOut, 0); });
            for (int i = 0; i < numHubs; ++i) {
                hubOut[i] = hubs[i];
            }
            for (jsize i = numHubs; i < hubsLen; ++i) {
                hubOut[i] = -1;   // unused slot sentinel
            }
        }

        // ---- Working state (all O(n), reused across every page) -----------------------------------
        // Acquire a raw pointer to the Java int[] output. GetPrimitiveArrayCritical may pin the array
        // (no GC while held), so the critical section below does no JNI calls and no allocation beyond
        // the std::vectors already sized; JNIReleaseElements releases it on scope exit (RAII).
        jint* out = static_cast<jint*>(env->GetPrimitiveArrayCritical(orderingJ, nullptr));
        knn_jni::JNIReleaseElements release([=]{ env->ReleasePrimitiveArrayCritical(orderingJ, out, 0); });

        std::vector<bool> assigned(n, false);     // has this node been placed on some page yet?
        std::vector<bool> isCandidate(n, false);  // is this node currently a candidate for the page being grown?
        std::vector<int> affinity(n, 0);          // for a candidate: #neighbors already on the current page
        // Bucket queue: buckets[k] holds candidates whose affinity == k (with possible stale duplicates,
        // see popBest). affinity[node] <= degree[node] <= maxDegree, so indices fit in [0, maxDegree].
        std::vector<std::vector<int>> buckets(maxDegree + 1);

        std::vector<int> page;                // nodes placed on the page currently being grown
        std::vector<int> candidatesToClear;   // every node that became a candidate this page (for cleanup)
        page.reserve(pageCapacity);
        int topBucket = 0;      // highest bucket index that (may) contain a live candidate
        int nextOrd = 0;        // next free physical slot; always a multiple of pageCapacity at a page start

        // Register `node` as a candidate for the current page at its current affinity score.
        auto addCandidate = [&](int node) {
            isCandidate[node] = true;
            candidatesToClear.push_back(node);   // remember it so we can reset it when the page closes
            const int score = affinity[node];
            buckets[score].push_back(node);
            if (score > topBucket) {
                topBucket = score;
            }
        };

        // Place `node` on the current page: give it the next physical slot, then raise the affinity of
        // each of its still-unassigned neighbors by 1 (they are now one edge "closer" to this page).
        auto addToPage = [&](int node) {
            assigned[node] = true;
            page.push_back(node);
            out[node] = nextOrd;   // FORWARD map: this original ordinal lives at physical slot nextOrd
            ++nextOrd;
            isCandidate[node] = false;   // it's now placed, not a candidate
            forEachNeighbor(node, [&](int nbr) {
                if (assigned[nbr]) {
                    return;   // neighbor already placed (maybe on an earlier page) — nothing to do
                }
                const int newScore = affinity[nbr] + 1;
                affinity[nbr] = newScore;
                if (isCandidate[nbr]) {
                    // Already a candidate: re-insert at the higher bucket. The old (lower) bucket entry
                    // is left behind as a STALE duplicate; popBest() detects and discards it.
                    buckets[newScore].push_back(nbr);
                    if (newScore > topBucket) {
                        topBucket = newScore;
                    }
                } else {
                    // First time we see this neighbor for this page — it becomes a candidate.
                    addCandidate(nbr);
                }
            });
        };

        // Return the unassigned candidate with the highest affinity, or -1 if none remain. Walks
        // topBucket downward; within a bucket, pops entries and skips STALE ones — a popped node is
        // "live" only if it's still a candidate AND its current affinity equals this bucket's index
        // (if it had been re-bucketed higher, or already placed, the current values won't match).
        auto popBest = [&]() -> int {
            while (topBucket >= 0) {
                std::vector<int>& bucket = buckets[topBucket];
                while (!bucket.empty()) {
                    const int cand = bucket.back();
                    bucket.pop_back();
                    if (isCandidate[cand] && affinity[cand] == topBucket) {
                        return cand;
                    }
                }
                --topBucket;   // this bucket is exhausted; drop to the next-highest
            }
            return -1;
        };

        // Grow one full page starting from `seed` (a hub, or a leftover node in the safety pass).
        auto growPage = [&](int seed) {
            if (assigned[seed]) {
                return;   // already pulled onto an earlier page — not a fresh seed
            }
            page.clear();
            candidatesToClear.clear();
            topBucket = 0;

            addToPage(seed);
            // Fill the cluster with the most-affine candidates until it reaches pageCapacity or no
            // candidate is left. pageCapacity bounds a cluster to roughly one physical page's worth of
            // records so a single dense pocket does not grow unbounded across the whole component.
            while ((int) page.size() < pageCapacity) {
                const int best = popBest();
                if (best < 0) {
                    break;   // cluster exhausted its affine candidates — close it and seed a new one
                }
                isCandidate[best] = false;
                addToPage(best);
            }

            // DENSE PACKING: no page-boundary padding. nextOrd was advanced by exactly one per node in
            // addToPage, so the next cluster begins immediately after this one (contiguous 0..n-1). A
            // cluster may straddle a physical 32 KB page boundary (the trade-off vs strict padding), but
            // the file stays exactly n records instead of numClusters*pageCapacity — far fewer physical
            // pages overall when clusters (~avg degree) are much smaller than pageCapacity.

            // Reset all scratch touched by this page so the next page starts from a clean slate.
            // (1) affinity was bumped on neighbors of placed nodes — zero those.
            for (int node : page) {
                forEachNeighbor(node, [&](int nbr) { affinity[nbr] = 0; });
            }
            // (2) any node that became a candidate (even if never placed) — clear its flag/affinity.
            for (int c : candidatesToClear) {
                isCandidate[c] = false;
                affinity[c] = 0;
            }
            // (3) empty every bucket (may still hold stale/unplaced candidates). clear() keeps the
            //     backing capacity, so pages after the first reuse the allocation — no churn.
            for (auto& b : buckets) {
                b.clear();
            }
        };

        // Seed pages from hubs in descending-degree order; each grows into a full page (pulling in its
        // cluster). A node reached as a candidate on an earlier page is skipped as a seed here.
        for (int hub : hubs) {
            if (!assigned[hub]) {
                growPage(hub);
            }
        }
        // Safety net: `hubs` already contains every node, so this normally does nothing. It only fires
        // for nodes no page ever reached (e.g. degree-0/unreachable), giving each its own page.
        for (int i = 0; i < n; ++i) {
            if (!assigned[i]) {
                growPage(i);
            }
        }

        // Every node must have been assigned exactly one physical slot (bijective forward map).
        for (int i = 0; i < n; ++i) {
            if (!assigned[i]) {
                throw std::runtime_error(
                    "buildOrderingOfVectorsUsingIndexStructure: node " + std::to_string(i) + " was left unassigned");
            }
        }
        // Dense packing invariant: exactly n slots consumed (contiguous 0..n-1, no padding gaps).
        if (nextOrd != n) {
            throw std::runtime_error(
                "buildOrderingOfVectorsUsingIndexStructure: emitted " + std::to_string(nextOrd)
                + " physical slots for " + std::to_string(n) + " nodes (expected dense 0..n-1)");
        }
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

// ============================================================================================
// buildOrderingOfVectorsUsingBFS — page-locality-aware BFS vector ordering (simpler alternative)
// ============================================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
// -----------------------------
// Same goal as buildOrderingOfVectorsUsingIndexStructure (see that function's header for the full
// motivation): choose an on-disk slot for every vector so that graph-adjacent vectors tend to land
// near each other, cutting page faults during graph search. This is the SIMPLER of the two ordering
// strategies — a plain breadth-first traversal — kept alongside the greedy one for experimentation
// and as an easy-to-reason-about baseline.
//
// TERMINOLOGY (same as the greedy variant)
// -----------------------------------------
//   originalOrdinal  : a vector's id in the graph / insertion order (0..n-1) = a Faiss node id.
//   physicalPosition : the on-disk slot the vector's record occupies (what we compute).
//   hub              : a high-degree (well-connected) node; the centers of dense clusters.
//
// THE ALGORITHM (hub-seeded BFS — a generalization of bfs_permutation in
// scripts/locality_simulation.py, which seeds from Faiss's single entry point)
// ---------------------------------------------------------------------------------------------
//   1. Compute each node's layer-0 out-degree and order all nodes by it, descending (ties by id).
//      This ordered list is the sequence of BFS start points ("hubs").
//   2. Walk the hubs; from each still-unvisited hub run a breadth-first traversal of the layer-0
//      graph, assigning each node the NEXT physical slot as it is dequeued. Because BFS visits a
//      node's neighbors right after the node, a node and its neighborhood land in nearby slots.
//   3. A hub already reached by an earlier hub's BFS is skipped. In a connected graph the first
//      (highest-degree) hub's BFS already reaches everything; later hubs only seed additional
//      connected components. Degree-0 / unreachable nodes become their own singleton starts.
//
// HOW IT DIFFERS FROM THE GREEDY VARIANT
// --------------------------------------
//   * DENSE packing, no page awareness: physical positions are a contiguous 0..n-1 with NO padding,
//     so it takes no pageCapacity and the writer never zero-fills gaps. The trade-off: a run of
//     graph-adjacent nodes can straddle a physical 32 KB page boundary, so locality is weaker than
//     the greedy strict-page layout (which guarantees one logical page == one physical page).
//   * No affinity / bucket queue — just FIFO BFS order. Cheaper and simpler, but it doesn't actively
//     maximize how many of each node's edges stay on the same page.
//
// OUTPUT / CONTRACT (identical shape to the greedy variant)
// ---------------------------------------------------------
//   ordering[originalOrdinal] = physicalPosition   (FORWARD map; length n; bijective; here DENSE 0..n-1)
//   hubs[rank]                = originalOrdinal     (top-degree node ids as search entry-point
//                                                    candidates; min(hubs.length, n) filled, -1-padded)
// The forward map is what the reader's ordToPhysicalOrdMap needs.
//
// PERFORMANCE / MEMORY
// --------------------
// Runs on the in-place native adjacency (zero copy — the 2*N*M neighbor array never crosses JNI).
// Scratch is O(n): degree[n], visited[n] (1 bit), the BFS queue (<= n). Time is ~O(n + edges): the
// degree pass walks every edge once, the BFS walks every edge once.
// ============================================================================================
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_buildOrderingOfVectorsUsingBFS(
    JNIEnv * env, jclass cls, jlong indexMemoryAddress, jintArray orderingJ, jintArray hubsJ) {
    try {
        if (indexMemoryAddress == 0) {
            throw std::runtime_error("buildOrderingOfVectorsUsingBFS: null index address");
        }
        // Same defensive down-cast chain as the greedy variant: a wrong index type throws instead of
        // being blindly reinterpreted.
        auto* indexBinary = reinterpret_cast<faiss::IndexBinary*>(indexMemoryAddress);
        auto* binaryIdMap = dynamic_cast<faiss::IndexBinaryIDMap*>(indexBinary);
        if (binaryIdMap == nullptr) {
            throw std::runtime_error("buildOrderingOfVectorsUsingBFS: index is not a faiss::IndexBinaryIDMap");
        }
        auto* faissSQHnsw = dynamic_cast<knn_jni::FaissSQHnsw*>(binaryIdMap->index);
        if (faissSQHnsw == nullptr) {
            throw std::runtime_error("buildOrderingOfVectorsUsingBFS: wrapped index is not a knn_jni::FaissSQHnsw");
        }
        faiss::HNSW & hnsw = faissSQHnsw->hnsw;

        // ordering is the FORWARD permutation: ordering[originalOrdinal] = physicalPosition. Unlike the
        // strict-page greedy variant, BFS packs densely (no padding), so physical positions are a
        // contiguous 0..ntotal-1. Length must equal ntotal.
        const jsize outLen = env->GetArrayLength(orderingJ);
        if ((int64_t) outLen != (int64_t) faissSQHnsw->ntotal) {
            throw std::runtime_error(
                "buildOrderingOfVectorsUsingBFS: ordering length " + std::to_string((int64_t) outLen)
                + " != ntotal (" + std::to_string((int64_t) faissSQHnsw->ntotal) + ")");
        }
        const int n = (int) outLen;
        if (n == 0) {
            return;
        }

        // Reads a node's valid layer-0 neighbors in place (zero-copy over hnsw.neighbors), bounds-checked.
        auto forEachNeighbor = [&](int node, auto&& fn) {
            size_t begin = 0, end = 0;
            hnsw.neighbor_range(node, 0, &begin, &end);
            for (size_t e = begin; e < end; ++e) {
                const int nb = hnsw.neighbors[e];
                if (nb >= 0 && nb < n) {
                    fn(nb);
                }
            }
        };

        // HUB SEEDING: BFS is started from graph "hubs" — the highest-degree (most connected) nodes,
        // which sit at the centers of the densest clusters. Starting there means the traversal expands
        // outward from cluster centers, so a hub and its dense neighborhood land in consecutive physical
        // slots. (The Faiss single-entry-point BFS in the simulation is a special case of this: in a
        // connected graph the first, highest-degree hub already reaches every node; the remaining hubs
        // only matter to seed additional connected components.)
        //
        // We compute per-node layer-0 out-degree and order all nodes by it, descending (ties by id for
        // determinism). This ordered list `hubs` is the sequence of BFS start points.
        std::vector<int> degree(n, 0);
        for (int i = 0; i < n; ++i) {
            int d = 0;
            forEachNeighbor(i, [&](int) { ++d; });
            degree[i] = d;
        }
        std::vector<int> hubs(n);
        for (int i = 0; i < n; ++i) {
            hubs[i] = i;
        }
        std::sort(hubs.begin(), hubs.end(), [&degree](int a, int b) {
            if (degree[a] != degree[b]) {
                return degree[a] > degree[b];
            }
            return a < b;
        });

        // Export the top hubs as search entry-point candidates (same contract as the greedy variant):
        // fill min(hubsLen, n) highest-degree node ids into hubsJ (ORIGINAL ordinals), pad the rest
        // with -1. Own critical section, ends before the ordering array is pinned below.
        if (hubsJ != nullptr) {
            const jsize hubsLen = env->GetArrayLength(hubsJ);
            const int numHubs = std::min((int) hubsLen, n);
            jint* hubOut = static_cast<jint*>(env->GetPrimitiveArrayCritical(hubsJ, nullptr));
            knn_jni::JNIReleaseElements hubRelease([=]{ env->ReleasePrimitiveArrayCritical(hubsJ, hubOut, 0); });
            for (int i = 0; i < numHubs; ++i) {
                hubOut[i] = hubs[i];
            }
            for (jsize i = numHubs; i < hubsLen; ++i) {
                hubOut[i] = -1;   // unused slot sentinel
            }
        }

        // ---- BFS traversal --------------------------------------------------------------------------
        // Acquire a raw pointer to the Java int[] output. GetPrimitiveArrayCritical may pin the array
        // (no GC while held), so the loop below does no JNI calls and no allocation beyond the vectors
        // already sized; JNIReleaseElements releases it on scope exit (RAII).
        jint* out = static_cast<jint*>(env->GetPrimitiveArrayCritical(orderingJ, nullptr));
        knn_jni::JNIReleaseElements release([=]{ env->ReleasePrimitiveArrayCritical(orderingJ, out, 0); });

        std::vector<bool> visited(n, false);   // has this node been dequeued/placed yet? (bit-packed)
        std::queue<int> frontier;              // FIFO of discovered-but-not-yet-placed nodes
        int pos = 0;                           // next free physical slot (dense: 0..n-1)

        // Visit hubs in descending-degree order; each unvisited hub starts a fresh BFS. Physical slot
        // is assigned in dequeue order, so a node's neighbors get slots right after it.
        for (int h = 0; h < n; ++h) {
            const int hub = hubs[h];
            if (visited[hub]) {
                continue;   // this hub was already reached by an earlier hub's BFS
            }
            visited[hub] = true;
            frontier.push(hub);
            while (!frontier.empty()) {
                const int node = frontier.front();
                frontier.pop();
                out[node] = pos;   // forward map: physical slot for this original ordinal
                ++pos;
                forEachNeighbor(node, [&](int nbr) {
                    if (!visited[nbr]) {
                        visited[nbr] = true;
                        frontier.push(nbr);
                    }
                });
            }
        }

        // Every node must be emitted exactly once (bijective permutation).
        if (pos != n) {
            throw std::runtime_error(
                "buildOrderingOfVectorsUsingBFS: emitted " + std::to_string(pos)
                + " of " + std::to_string(n) + " nodes");
        }
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}
