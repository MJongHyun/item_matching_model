import re


# count vector 쓰기 위해서 리스트를 텍스트로 변환
def set_token(txt):
    txt2 = [element.upper() for element in txt];txt
    return ' '.join(txt2)


def rm_num(txt):
    txt = list(txt)
    num_lst = []
    for idx in range(0, len(txt)):
        num = re.findall('\d+', txt[idx])
        if len(num) > 0:
            num_lst.append(idx)
    num_lst.reverse()
    if len(num_lst) > 0:
        for num in num_lst:
            del txt[num]
    return txt


# 분석 토큰만 추출
def exist_vec_basis(tokens, basis):
    return [item for item in tokens if item.lower() in basis]


def token_vec(df):
    # 토큰 전처리
    df['n_tokens'] = df.apply(lambda row: rm_num(row['tokens']), axis=1)  # 숫자 제거
    df['new_tokens'] = df.apply(lambda row: set_token(row['n_tokens']), axis=1)  # 리스트를 텍스트로 변환
    df['tokens_split'] = df.new_tokens.str.split(' ')
    return df


def stop_words():
    stop_words = ['행사', '지점', 'cs', '발주불가', '한정판','에디션', \
               'NEW', '세트', 'plus', '리뉴얼', \
               'ml',  '행사', '케그', 'can', 'btl', 'keg', '소캔', '대캔', '중캔',\
               '행사', '지점','발주불가', 'new', '세트', 'PLUS', '페트', 'pack', 'aa', 'bottle' ]
    return stop_words


