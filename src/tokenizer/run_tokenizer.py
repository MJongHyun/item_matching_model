import pandas as pd
import re
from src.tokenizer.Dionysus_tokenizer import Dionysus_tokenizer


# 누적사전 변경값 적용
def new_item_nm(txt1, txt2):
    if (txt2 is not None) & (pd.isna(txt2) == False):
        return txt2
    else:
        return txt1


# 사이즈 제거
def rmv_sz(txt, sz=False):
    if sz:
        com = "%d|%.2f|%dL" % (int(sz), sz / 1000., int(sz / 1000.))
        com = com.replace(".", "\.")
        if com[-1] == '0':
            com = com[:-1] + "\d*"
    else:
        com = "\d+$"
    return re.sub(com, '', txt)


# 전처리 (train data에 신규 아이템값 적용)
def pre_process(item_info, cumul):
    item_info = item_info.dropna(subset=['NAME'])
    cumul = cumul.dropna(subset=['NAME'])
    item_info.rename(columns={'ITEM': 'BF_ITEM'}, inplace=True)
    cumul.drop(['SAW', 'AMT', 'ITEM_UP'], axis=1, inplace=True)
    cumul.rename(columns={'ITEM': 'AF_ITEM'}, inplace=True)
    item_info = item_info.astype({'ITEM_SZ': 'float'})
    cumul = cumul.astype({'ITEM_SZ': 'float'})
    diony = pd.merge(item_info, cumul, on=['WR_DT', 'BARCODE', 'TYPE', 'NAME', 'ITEM_SZ'], how='left')
    diony['ITEM'] = diony[['BF_ITEM', 'AF_ITEM']].apply(lambda x: new_item_nm(x['BF_ITEM'], x['AF_ITEM']), axis=1)
    diony.drop(['BF_ITEM', 'AF_ITEM'], axis=1, inplace=True)
    return diony


# train data tokenization
def run_train_tokenizer(item_info, cumul, logger, train_dt, num):
    logger.info("start")
    try:
        cumul_copy = cumul.copy()
        col = 'NAME'
        col_2 = 'ITEM_SZ'
        data_path = 'data/base/{}/'.format(train_dt)
        # DB에서 품목을 불러와서 기본 토큰을 생성 후 커스텀 토크나이즈
        DT = Dionysus_tokenizer()
        DT.set_base_token()
        repl_item = pd.read_excel("src/tokenizer/data/add_token.xlsx")
        DT.set_custom_token(repl_item)
        diony = pre_process(item_info, cumul)
        diony = diony.drop_duplicates().reset_index().copy()
        # 전달 데이터 토큰화
        diony['tokens'] = diony[[col, col_2]].apply(lambda x: DT.extract_keywords(rmv_sz(x[col], x[col_2])), axis=1)
        diony.to_parquet(data_path+"month_data.parquet")
        # 누적사전 토큰화
        cumul_copy = cumul_copy[cumul_copy['SAW'] != "안봄"]
        cumul_copy = cumul_copy.sort_values(['WR_DT'], ascending=False)[:num]
        cumul_copy['tokens'] = cumul_copy[[col, col_2]].apply(lambda x: DT.extract_keywords(rmv_sz(x[col], x[col_2])), axis=1)
        cumul_copy.to_parquet(data_path + "cumul_data.parquet")
        logger.info("END")
    except Exception as e:
        logger.error(e)
        raise Exception("TOKENIZER ERROR")


# test data tokenization
def run_test_tokenizer(df, logger, train_dt):
    logger.info("START")
    try:
        df = df.dropna(subset=['NAME'])
        diony = df.drop_duplicates().reset_index().copy()
        diony = diony.astype({'ITEM_SZ':'float'})

        # DB에서 품목을 불러와서 기본 토큰을 생성 후 커스텀 토크나이즈
        DT = Dionysus_tokenizer()
        DT.set_base_token()
        repl_item = pd.read_excel("src/tokenizer/data/add_token.xlsx")
        DT.set_custom_token(repl_item)
        col = 'NAME'
        col_2 = 'ITEM_SZ'
        diony['tokens'] = diony[[col, col_2]].apply(lambda x: DT.extract_keywords(rmv_sz(x[col], x[col_2])), axis=1)
        diony.to_parquet("data/base/{}/test_data.parquet".format(train_dt))
        logger.info("END")
    except Exception as e:
        logger.error(e)
        raise Exception("TOKENIZER ERROR")


def one_word(word):
    print("START test_tokenizer one_word")
    try:
        # DB에서 품목을 불러와서 기본 토큰을 생성 후 커스텀 토크나이즈
        DT = Dionysus_tokenizer()
        DT.set_base_token()
        repl_item = pd.read_excel("src/tokenizer/data/add_token.xlsx")
        DT.set_custom_token(repl_item)
        test = DT.extract_keywords(word)
        print('test', test)
        print("END test_tokenizer one_word")
    except Exception as e:
        print("ERROR")
        raise Exception("TOKENIZER ERROR")

