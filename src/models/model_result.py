import pandas as pd
from sklearn.metrics import confusion_matrix


def check_col(saw, n_gram, dnn, svm):
    if saw == '봄':
        return n_gram
    else:
        value_lst = [n_gram, dnn, svm]
        if value_lst.count(1) > 0:
            return 1
        else:
            return 0


def mk_coufuison(df):
    coufuison_tb = confusion_matrix(df['ITEM_label'], df['predict_label'], labels=[0, 1])
    coufuison_tb = pd.DataFrame(coufuison_tb)
    coufuison_tb = coufuison_tb.reset_index()
    coufuison_tb.columns = ['True\Predict', '0', '1']
    print(coufuison_tb)


def load_result(result_path, answer_path):
    # data load
    result_ng = pd.read_excel(result_path + "item_info_AFTER.xlsx")
    result_dnn = pd.read_parquet(result_path + "dnn_result_df.parquet")[['NAME', 'ITEM_SZ', 'predict_dnn_label']]
    result_svm = pd.read_parquet(result_path + "svm_result_df.parquet")[['NAME', 'ITEM_SZ', 'SVM_PREDICT']]
    test_df = pd.read_parquet(answer_path + "item_info_FINAL.parquet")   # 정답 파일

    # n_gram 비모집단, 모집단 구분
    result_ng.loc[(result_ng['ITEM'].str.contains("비모집단")) | (result_ng['ITEM'].str.contains("비주류")), 'ng_ITEM'] = 0
    result_ng.loc[(~result_ng['ITEM'].str.contains("비주류")) & (~result_ng['ITEM'].str.contains("비모집단")), 'ng_ITEM'] = 1
    result_ng = result_ng[['NAME','ITEM_SZ','ng_ITEM']]

    # 타입 맞추기
    test_df.dropna(subset=['NAME'], inplace=True)
    test_df = test_df.astype({'ITEM_SZ':'float'})
    test_df.loc[(test_df['ITEM'].str.contains("비모집단")) | (test_df['ITEM'].str.contains("비주류")), 'ITEM_label'] = 0
    test_df.loc[(~test_df['ITEM'].str.contains("비주류")) & (~test_df['ITEM'].str.contains("비모집단")), 'ITEM_label'] = 1

    result_ng = result_ng.astype({'ITEM_SZ': 'float'})
    result_dnn = result_dnn.astype({'ITEM_SZ': 'float'})
    result_svm = result_svm.astype({'ITEM_SZ': 'float'})

    # 데이터 합
    mg = pd.merge(test_df, result_ng, on=['NAME','ITEM_SZ'], how='left')
    mg_rst = pd.merge(mg, result_dnn, on=['NAME','ITEM_SZ'], how='left')
    mg_rst_f = pd.merge(mg_rst, result_svm, on=['NAME','ITEM_SZ'], how='left')
    if mg_rst_f.shape[0] == test_df.shape[0]:
        return mg_rst_f


def load_result_not_answer(result_path):
    # data load
    result_ng = pd.read_excel(result_path + "item_info_AFTER.xlsx")
    result_dnn = pd.read_parquet(result_path + "dnn_result_df.parquet")[['NAME', 'ITEM_SZ', 'predict_dnn_label']]
    result_svm = pd.read_parquet(result_path + "svm_result_df.parquet")[['NAME', 'ITEM_SZ', 'SVM_PREDICT']]

    # n_gram 비모집단, 모집단 구분
    result_ng.loc[(result_ng['ITEM'].str.contains("비모집단")) | (result_ng['ITEM'].str.contains("비주류")), 'ng_ITEM'] = 0
    result_ng.loc[(~result_ng['ITEM'].str.contains("비주류")) & (~result_ng['ITEM'].str.contains("비모집단")), 'ng_ITEM'] = 1
    # result_ng = result_ng[['SAW','NAME','ITEM_SZ','ng_ITEM']]


    result_ng = result_ng.astype({'ITEM_SZ': 'float'})
    result_dnn = result_dnn.astype({'ITEM_SZ': 'float'})
    result_svm = result_svm.astype({'ITEM_SZ': 'float'})

    # 데이터 합
    mg_rst = pd.merge(result_ng, result_dnn, on=['NAME','ITEM_SZ'], how='left')
    mg_rst_f = pd.merge(mg_rst, result_svm, on=['NAME','ITEM_SZ'], how='left')
    if mg_rst_f.shape[0] == result_ng.shape[0]:
        return mg_rst_f


def mk_mg_label(df):
    df['predict_label'] = df[['SAW','ng_ITEM','predict_dnn_label', 'SVM_PREDICT']]\
                        .apply(lambda x: check_col(x['SAW'], x['ng_ITEM'], x['predict_dnn_label'], x['SVM_PREDICT']), axis=1)
    df["correct"] = df['ITEM_label'] == df['predict_label']
    print("score: ", len(df[df['correct']])/len(df))
    return df


def mk_mg_label_not_answer(df):
    df['predict_label'] = df[['SAW','ng_ITEM','predict_dnn_label', 'SVM_PREDICT']]\
                        .apply(lambda x: check_col(x['SAW'], x['ng_ITEM'], x['predict_dnn_label'], x['SVM_PREDICT']), axis=1)
    return df


def mg_result_not_answer(logger, test_dt):
    try:
        logger.info("START")
        result_path = 'data/result/{}/'.format(test_dt)
        mg_rst = load_result_not_answer(result_path)
        mg_rst = mk_mg_label_not_answer(mg_rst)
        mg_rst.to_excel(result_path+'item_matching_result.xlsx', index=False)
        logger.info("END")
    except Exception as e:
        print("ERROR")
        logger.error(e)
        raise


def mg_result_answer(logger, test_dt):
    try:
        logger.info("START")
        result_path = 'data/result/{}/'.format(test_dt)
        answer_path = 'data/row_data/{}/'.format(test_dt)
        mg_rst = load_result(result_path, answer_path)
        mg_rst = mk_mg_label(mg_rst)
        mk_coufuison(mg_rst)
        mg_rst.to_excel(result_path+'item_matching_result.xlsx', index=False)
        logger.info("END")
    except Exception as e:
        print("ERROR")
        logger.error(e)
        raise
