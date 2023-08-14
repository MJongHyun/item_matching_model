import pymysql
import pandas as pd
import re
import numpy as np


# DB 접속 정보
class Access:

    def __init__(self):
        """
        DB 접근 class
        """

        self.host = ''
        self.user = ''
        self.port = ''
        self.pw = ''
        self.db_nm = ''
        self.charset = 'utf8'

        self._connect()

    def _connect(self):
        """
        DB 접속 후, cursor, connect 변수 처리
        """
        # DB connect

        self.connect = pymysql.connect(
            host=self.host,
            user=self.user,
            port=self.port,
            passwd=self.pw,
            db=self.db_nm,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor
        )
        # cursor 객체
        self.cursor = self.connect.cursor()

    def _close(self):
        """
        DB 종료
        """
        self.cursor.close()
        self.connect.close()
