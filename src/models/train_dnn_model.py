import pickle
import numpy as np
import pandas as pd
from src.common.user_function import token_vec, exist_vec_basis, stop_words
from keras import models
from keras import layers
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import atexit


def get_model(input_shapes):
    model = models.Sequential()
    model.add(layers.Dense(1000, activation='relu', input_shape=(input_shapes,)))
    model.add(layers.Dropout(0.2))  # 과적합 방지
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def load_data(train_dt):
    train_data = pd.read_parquet('data/base/{}/cumul_data.parquet'.format(train_dt), columns=['WR_DT','SAW','NAME','ITEM_SZ','ITEM','tokens'])
    train_data.loc[(train_data['ITEM'].str.contains("비모집단")) | (train_data['ITEM'].str.contains("비주류")), 'ITEM_label'] = 0
    train_data.loc[(~train_data['ITEM'].str.contains("비주류")) & (~train_data['ITEM'].str.contains("비모집단")), 'ITEM_label'] = 1
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    return train_data


def mk_dnn_train_dataset(logger, train_dt, num):
    try:
        logger.info("START")
        data_tmp = load_data(train_dt)
        if len(data_tmp) == int(num):
            # 학습 데이터
            train_data = token_vec(data_tmp)
            train_data = train_data[train_data['new_tokens'] != '']
            train_data = train_data.drop_duplicates('new_tokens')
            train_content = train_data['new_tokens'].tolist()
            # 분석에 사용될 토큰 지정
            count_vectorizer = CountVectorizer(stop_words=stop_words(), min_df=2)
            count_vectorizer.fit(train_content)  # 벡터라이저가 단어들을 학습합니다.
            pickle.dump(count_vectorizer, open("data/base/{}/count_vectorizer.pkl".format(train_dt), "wb"))
            # 학습에 사용되는 토큰
            train_data["exist_basis"] = train_data.apply(lambda x: exist_vec_basis(x['tokens_split'], count_vectorizer.vocabulary_), axis=1)
            train_df = train_data[train_data['exist_basis'].str.len() != 0]
            # 검증 데이터
            validation_df = train_df.copy()
            validation_df = validation_df.sample(frac=0.3)
            validation_df = validation_df.reset_index()
            logger.info("END")
            return count_vectorizer, train_df, validation_df
        else:
            raise Exception("data count 오류")
    except Exception as e:
        print(e)
        logger.error(e)
        raise


def batch_generator(count_vectorizer, X, batch_size):
    indices = np.arange(len(X))
    batch=[]
    while True:
        for i in indices:
            batch.append(i)
            if len(batch)==batch_size:
                x_df = X.iloc[batch]
                train_content = x_df['new_tokens'].tolist()
                train_x = count_vectorizer.transform(train_content).toarray()
                train_y = x_df['ITEM_label']
                yield train_x, np.array(train_y)
                batch=[]


def get_data(count_vectorizer, train_df, val_df, batch_size):
    train_generator = batch_generator(count_vectorizer, train_df, batch_size)
    val_generator = batch_generator(count_vectorizer, val_df, batch_size)

    return train_generator, val_generator


def train_dnn_model(logger, train_dt, mirrored_strategy, num):
    logger.info("START")
    try:
        model_data_path = 'data/base/{}/'.format(train_dt)
        # 토큰 전처리
        count_vectorizer, train_df, validation_df = mk_dnn_train_dataset(logger, train_dt, num)
        batch_size = 4096
        input_shape = len(count_vectorizer.vocabulary_)
        epochs = 50
        num_batches = int(len(train_df) / batch_size + 1)
        val_batches = int(len(validation_df) / batch_size + 1)
        logger.info("train dnn model")
        with mirrored_strategy.scope():
            model = get_model(input_shape)
            train_generator, val_generator = get_data(count_vectorizer, train_df, validation_df, batch_size)
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=3)
            mc = tf.keras.callbacks.ModelCheckpoint(model_data_path + 'DNN_model.h5', monitor='val_loss', mode='auto',
                                                    verbose=1, save_best_only=True)
            logger.info("fit dnn model")
            model.fit(train_generator, steps_per_epoch=num_batches, epochs=epochs,
                      validation_data=val_generator, validation_steps=val_batches,
                      callbacks=[es, mc])
        train_df.to_parquet(model_data_path+"dnn_train_df.parquet")
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close)
        logger.info("END")
    except Exception as e:
        print(e)
        logger.error(e)
        raise Exception("DNN TRAIN ERROR")


