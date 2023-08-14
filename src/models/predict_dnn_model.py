import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from src.common.user_function import token_vec, exist_vec_basis, stop_words
import atexit

def load_model(train_dt):
    model = tf.keras.models.load_model('data/base/{}/DNN_model.h5'.format(train_dt))
    return model


def mg_test_df(df, predict, score):
    if score > 0.01:
        score = 0.01
    else:
        score = score * (2 / 3)
    df['predict_vec'] = predict
    # 최대한 모집단으로 분류하도록 임계치 설정
    predict_y_lst = []
    for i in range(len(predict)):
        if predict[i]>score:
            predict_y_lst.append(1)
        else:
            predict_y_lst.append(0)
    df['predict_dnn_label'] = predict_y_lst
    return df


def batch_generator_test(count_vectorizer, X, batch_size):
    indices = np.arange(len(X))
    batch=[]
    number = len(indices)
    while True:
        for i in indices:
            batch.append(i)
            number -=1
            if (len(batch)==batch_size) | (number==0):
                x_df = X.iloc[batch]
                train_content = x_df['new_tokens'].tolist()
                train_x = count_vectorizer.transform(train_content).toarray()
                yield train_x
                batch=[]


def predict_dnn(logger, train_dt, test_dt, mirrored_strategy):
    try:
        logger.info("START")
        data_path = 'data/base/{}/'.format(train_dt)
        result_path = 'data/result/{}/'.format(test_dt)

        count_vectorizer = pickle.load(open(data_path+"count_vectorizer.pkl", 'rb'))
        # 테스트 데이터
        test_data = pd.read_parquet(data_path+"test_data.parquet")
        test_df = token_vec(test_data)
        test_df["exist_basis"] = test_df.apply(lambda x: exist_vec_basis(x['tokens_split'], count_vectorizer.vocabulary_), axis=1)
        # 나눠서 예측
        batch_size = 1000
        test_batches = int(len(test_df) / batch_size + 1)
        test_generator = batch_generator_test(count_vectorizer, test_df, batch_size)
        # dataset 저장
        model = load_model(train_dt)
        # 빈 값일 경우 score 확인 (score가 0.01보다 작으면 그 값으로 필터 (신제품때문))
        blank_vec_tmp = np.zeros(len(count_vectorizer.vocabulary_))
        blank_vec = np.array([blank_vec_tmp])
        with mirrored_strategy.scope():
            predict_y = model.predict(test_generator, steps=test_batches)
            score = model.predict(blank_vec)[0][0]
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close)  # type: ignore
        test_df = mg_test_df(test_df, predict_y, score)
        test_df.to_parquet(result_path+"dnn_result_df.parquet", index=False)
        logger.info("END")
    except Exception as e:
        logger.error(e)
        raise Exception("DNN PREDICT ERROR")
