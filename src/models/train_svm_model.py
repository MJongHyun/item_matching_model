import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize
from src.common.user_function import exist_vec_basis
import joblib


# set_token : 필요한 token(size)를 채워주는 함수
# txt : 세금계산서에 작성된 아이템명
# sz : 세금계산서에 작성된 사이즈명
def set_token(txt, sz):
    try :
        # token list화 한 후, 용기를 구분하는 값이 없을 경우 사이즈에 따라 값 추가
        tokens = list(txt)
        sz = [str(int(sz))]
        num_lst = []
        if not set(['케그','캔','페트']).intersection(set(tokens)) :
            if int(sz[0]) < 1000 :
                tokens = tokens + ['병']
            elif int(sz[0]) in [5000,8000,10000,12000,12500,15000,18000,19000,19500,20000,25000,30000,50000]:
                tokens = tokens + ['케그']
        # token에 sz값과 같은 값이 존재한다면 추가하지 않음, 존재하지 않는다면 token에 size값 추가
        if set(tokens).intersection(set(sz)):
            return ' '.join(tokens)
        else :
             return ' '.join(tokens + sz)
    except:
        print(txt)


# newTokenBasisFile : set_token을 통해 신규 token 추출 후, token별 vector 위치 값 추출
def newTokenBasisFile(data):
    # set_token 함수를 통한 신규 token값 추출
    data['new_tokens'] = data[['tokens', 'ITEM_SZ']].apply(lambda x: set_token(x['tokens'], x['ITEM_SZ']), axis=1)
    data['tokens_split'] = data.new_tokens.str.split(' ')
    content = data['new_tokens'].tolist()
    # CountVectorizer를 통해 모든 token을 vector화
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(content)
    # token에 따르는 위치 (index) 및 token 값을 추출
    token_list = []
    for key, val in vectorizer.vocabulary_.items():
        token_list.append([key, val])
    tokenDFpre = pd.DataFrame(token_list, columns=['key', 'val'])
    keyList = tokenDFpre.reset_index(drop=True)['key'].tolist() + ['병', '캔', 'J&B']
    tokenDF = pd.DataFrame(keyList).reset_index()
    tokenDF.index = tokenDF[0]
    basis = tokenDF['index'].to_dict()
    return data, basis


def vectorize(tokens, length, basis):
    if isinstance(tokens, str):
        tokens = tokens.split(' ')
    vector = np.zeros([length])
    for val in tokens:
        if val in basis:
            idx = basis[val]
            vector[idx] = 1
    return vector


# row별 token결과를 vector에 적용
# vector 적용한 결과를 csr_matrix로 변환
def newTokenTrainDataFile(svm_trainDF, svm_trainBasis):
    # 새롭게 만든 token 값 List화
    content = svm_trainDF['new_tokens'].tolist()
    tokenList = []
    # row별 token결과를 vector에 적용
    for txt in content:
        vectorSize = len(svm_trainBasis)
        tokenList.append(vectorize(txt, vectorSize, svm_trainBasis))
    # vector 적용한 결과를 csr_matrix로 변환
    tokenArray = np.array(tokenList)
    tokenNomarlizeVector = normalize(tokenArray)
    csr_mat = csr_matrix(tokenNomarlizeVector)

    return csr_mat


# svm train/test Matrix 및 train Category Array 값 추출 및 모델 학습
def run_train_svm_model(logger, train_dt):
    logger.info("START")
    try:
        data_path = "data/base/{}/".format(train_dt)
        # trainDF 불러오기
        trainDF = pd.read_parquet(data_path+'month_data.parquet')
        # set_token을 통해 적용한 train vector index(basis) 데이터 추출
        svm_trainDF, svm_trainBasis = newTokenBasisFile(trainDF)
        # 학습에 사용되는 토큰
        svm_trainDF["exist_basis"] = svm_trainDF.apply(lambda x: exist_vec_basis(x['tokens_split'], svm_trainBasis), axis=1)
        # 비모집단/모집단 구분하는 값 추출
        svm_trainDF["CATEGORY"] = 0
        svm_trainDF.loc[~svm_trainDF.ITEM.str.contains("비모집단|비주류"), "CATEGORY"] = 1
        logger.info("Start svm trainMatrix")
        # train Category Array로 변환 후 저장
        svm_trainMatrix = newTokenTrainDataFile(svm_trainDF, svm_trainBasis)
        trainCategoryList = svm_trainDF.CATEGORY.tolist()
        trainCategoryArray = np.array(trainCategoryList).reshape(len(trainCategoryList), 1)
        # 위에서 만든 파일들을 통해 train matrix 추출
        svm_model = svm.SVC(probability = True, kernel = 'rbf', C = 10, gamma = 1, verbose = True)
        svm_model.fit(svm_trainMatrix, trainCategoryArray.ravel())
        logger.info("END svm trainMatrix")
        pickle.dump(svm_trainBasis, open(data_path+"svm_trainBasis.pkl", "wb"))
        joblib.dump(svm_model, data_path+'svm_train_model.pkl')
        svm_trainDF.to_parquet(data_path+"svm_train_data.parquet")
        logger.info("SAVE FILES")
        logger.info("END")
    except Exception as e:
        logger.error(e)
        raise Exception("SVM TRAIN ERROR")

