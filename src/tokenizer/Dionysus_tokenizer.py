import pandas as pd
import re
from src.tokenizer.BizD_KeywordProcessor import BizD_KeywordProcessor
from src.tokenizer.DBinfo import Access


class Dionysus_tokenizer():
    def __init__(self):
        self.keyword_processor = BizD_KeywordProcessor()
        #        self.keyword_processor.non_word_boundaries = {}
        return

    def set_base_token(self):
        base_item = self.get_base_item()
        base_item['len'] = base_item['PRDLST_NM'].str.len()
        base_item['PRDLST_NM_tmp'] = base_item['PRDLST_NM'].str.replace(" ", "")
        idx = base_item.groupby(['PRDLST_NM_tmp'])['len'].transform(max) == base_item['len']
        base_item = base_item[idx][['BSSH_NM', 'PRDLST_NM', 'PRDLST_DCNM']].copy()

        BSSH_NM = base_item[['BSSH_NM']].drop_duplicates().copy()
        BSSH_NM['BSSH_NM'] = BSSH_NM['BSSH_NM'].apply(lambda x: self.split_korean(x))
        BSSH_NM['PRDLST_NM'] = base_item['BSSH_NM']
        base_item = base_item.append(BSSH_NM)

        base_token = self.__extract_token(base_item)
        self.base_token = base_token
        self.base_item = base_item
        self.__base_token_add(base_token)

        return

    def set_custom_token(self, repl_token, debug=False):

        self.__set_custom_token(repl_token, debug=debug)

        return

    def remove_token(self, repl_token):

        self.__remove_token(repl_token)

        return

    def extract_keywords(self, txt):
        txt = self.split_korean(txt)
        keywords_found = self.keyword_processor.extract_keywords(txt)

        res = []
        for val in keywords_found:
            if isinstance(val, list):
                for v in val:
                    if v != '':
                        res.append(v)
            #                 res.extend(val)
            else:
                res.append(val)
        return res

    def get_base_item(self):
        db = Access()
        db.cursor.execute("SELECT * from DOMESTIC_ITEM")
        data = db.cursor.fetchall()
        Kor_items = pd.DataFrame(data)
        Kor_items = Kor_items[~Kor_items['PRDLST_NM'].str.contains('수출(용|\d|\)| |,|전용|품|주)|수출$|B2B|주정')].reset_index(
            drop=True)
        #         Kor_items = Kor_items[~Kor_items['PRDLST_NM'].str.contains('주정')].reset_index(drop=True)

        db.cursor.execute("SELECT * from FOREIGN_ITEM")
        data = db.cursor.fetchall()
        IntN_items = pd.DataFrame(data)

        df = IntN_items[['BSN_OFC_NAME', 'GOODS_KOR_NAME', 'ITEM_NAME']].drop_duplicates().copy()

        df.columns = ['BSSH_NM', 'PRDLST_NM', 'PRDLST_DCNM']
        df = df.append(Kor_items[['BSSH_NM', 'PRDLST_NM', 'PRDLST_DCNM']])
        df = df.drop_duplicates().reset_index(drop=True)
        db._close()
        return df

    def __extract_token(self, df):
        NM = []
        for i, nm in df['PRDLST_NM'].iteritems():
            name = re.sub('\[|\]|\(|\)|\"|,|/|-', " ", nm)
            name = self.split_korean(name)
            NM.extend(name.split())
        NM = pd.DataFrame(NM, columns=['token']).drop_duplicates()

        return NM

    def __base_token_add(self, NM):
        for item in NM['token'].values:
            self.keyword_processor.add_keyword(item)
        return

    def __set_custom_token(self, NM, debug):
        for item, repl_item, flag, flag_desc in NM.values:
            item = str(item)
            repl_item = str(repl_item)
            item = self.split_korean(item)
            if str(flag) == '0':
                self.keyword_processor.add_keyword(item)
            elif str(flag) == '1':
                if debug:
                    rpl_items = repl_item.split(',')
                else:
                    tmp = repl_item.split(',')
                    rpl_items = []
                    for ri in tmp:
                        if not "-삭" in ri:
                            rpl_items.append(ri)
                        else:
                            rpl_items.append('')
                self.keyword_processor.add_keyword(item, rpl_items)
            elif str(flag) == '2':
                self.keyword_processor.remove_keyword(item)
            else:
                if debug:
                    print('Warning! check flag in add_token table!')
        return

    def __remove_token(self, NM):
        for item, repl_item in NM.values:
            self.keyword_processor.remove_keyword(item)
        return

    def split_korean(self, txt):
        passingWDS = ["[가-힣ㄱ-ㅎㅏ-ㅣ][^가-힣ㄱ-ㅎㅏ-ㅣ]", "[^가-힣ㄱ-ㅎㅏ-ㅣ][가-힣ㄱ-ㅎㅏ-ㅣ]", "\d[a-zA-Z]", "[a-zA-Z]\d"]
        for passingWD in passingWDS:
            t = ''
            sst = 0
            TF = False
            for fd in re.finditer(passingWD, txt):
                st, ed = fd.span()
                space = ' '
                if re.search('\d[입본]', txt[st:ed]):
                    space = ''
                t += txt[sst:st + 1] + space + txt[st + 1]
                sst = ed
                TF = True
            if TF:
                t += txt[ed:]
                txt = t
        txt = re.sub("\s+|_|,", ' ', txt)

        return txt