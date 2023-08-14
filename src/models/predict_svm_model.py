import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from src.models.train_svm_model import vectorize, newTokenBasisFile
from src.common.user_function import exist_vec_basis


def testbasisDF(trainBasis, testBasis):
    testDict = {}
    for key, res in testBasis.items():
        if key in trainBasis:
            testDict[key] = trainBasis[key]
    return testDict


def newTokenTestDataFile(svm_testDF, trainVectorSize, svm_testBasis):
    # 새롭게 만든 token 값 List화
    content = svm_testDF['new_tokens'].tolist()
    tokenList = []
    # row별 token결과를 vector에 적용
    for txt in content:
        tokenList.append(vectorize(txt, trainVectorSize, svm_testBasis))
    # vector 적용한 결과를 csr_matrix로 변환
    tokenArray = np.array(tokenList)
    tokenNomarlizeVector = normalize(tokenArray)
    csr_mat = csr_matrix(tokenNomarlizeVector)

    return csr_mat



# predict_svm : 저장한 model과 test Matrix를 통한 predict값 추출습
def predict_svm(logger, train_dt, test_dt):
    logger.info("START")
    try:
        data_path = './data/base/{}/'.format(train_dt)
        result_path = './data/result/{}/'.format(test_dt)
        test_df = pd.read_parquet(data_path+"test_data.parquet")
        # 카운트 벡터, 모델 로드
        svm_trainBasis = pickle.load(open(data_path + "svm_trainBasis.pkl", 'rb'))
        svm_model = joblib.load(data_path+'svm_train_model.pkl')
        # 벡터만들기
        svm_testDF, testBasisPre = newTokenBasisFile(test_df)
        # 학습에 사용되는 토큰
        svm_testDF["exist_basis"] = svm_testDF.apply(lambda x: exist_vec_basis(x['tokens_split'], svm_trainBasis),axis=1)
        svm_testBasis = testbasisDF(svm_trainBasis, testBasisPre)
        svm_testMatrix = newTokenTestDataFile(svm_testDF, len(svm_trainBasis), svm_testBasis)
        # 예측
        svm_predict = svm_model.predict_proba(svm_testMatrix)
        svm_predict_df = pd.DataFrame(svm_predict, columns=["zeroPer", "onePer"])
        svm_predict_df["SVM_PREDICT"] = 0
        svm_predict_df.loc[svm_predict_df["zeroPer"] <= svm_predict_df["onePer"], "SVM_PREDICT"] = 1
        # 결과 저장
        svm_result_df = pd.concat([svm_testDF, svm_predict_df], axis = 1)
        svm_result_df.to_parquet(result_path+"svm_result_df.parquet")
        logger.info("END")
    except Exception as e:
        logger.error(e)
        raise Exception("SVM PREDICT ERROR")
