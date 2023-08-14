import os
import string
import io
import pandas as pd


class Check_nGram():
    def __init__(self):
        self.mapping_table = pd.read_excel("./mapping_token_Table.xlsx")
        self.mapping_table = self.mapping_table.astype(str)
        return

    def diff_master_tokens(self, master, token, sz, except_tokens=[], debug=False):
        master_set = set(master) - set(['년'])
        keta_set = set(token + sz)
        if not set(['케그', '캔', '페트']).intersection(set(token)):
            if int(sz[0]) < 1000:
                keta_set = keta_set | set(['병'])
            elif int(sz[0]) in [5000, 8000, 10000, 12000, 12500, 15000, 18000, 19000, 19500, 20000, 25000, 30000,
                                50000]:
                keta_set = keta_set | set(['케그'])

        condition = False
        for brand, except_token in except_tokens:
            if not (set(brand.split(',')) - master_set):
                if debug: print(brand, master_set)
                master_set = master_set - set(except_token.split(','))
                if debug: print(brand, master_set)

        for brand, key, val in self.mapping_table.values:
            if not (set(brand.split(',')) - master_set):
                if debug: print(brand, key, val)
                if not (set(key.split(',')) - keta_set):
                    if debug: print(keta_set)
                    keta_set = keta_set | set(val.split(','))
                    if debug: print(keta_set)
        a = master_set - keta_set
        if debug: print(keta_set)
        if len(a) == 0:
            return master
        return ''