import logging
import io
from logging.handlers import TimedRotatingFileHandler

class Log:

    def __init__(self):
        # log 파일 위치
        self.path = './data/log/ITEM_MODEL.log'
        # log 출력 포맷
        self.form = '[%(asctime)s] [%(levelname)s] [%(filename)s] [%(funcName)s] [%(lineno)s] [%(message)s]'
        self.format = logging.Formatter(self.form, datefmt='%Y-%m-%d %H:%M:%S')
        # 로그 생성
        self.logger = logging.getLogger('ITEM_MODEL')

        # error message 가져오기 위함
        self.errors = io.StringIO()
        # 핸들러 생성
        ## 콘솔 출력 핸들러
        self.console_hdl = logging.StreamHandler()
        ## 파일 출력 핸들러
        self.file_hdl = logging.FileHandler(self.path)
        ## 파일 따로 생성 핸들러
        self.time_hdl = TimedRotatingFileHandler(self.path, when="d", interval=30, backupCount=5, encoding='utf-8')
        self.time_hdl.suffix = "%Y%m%d"
        # 포맷 지정
        self.console_hdl.setFormatter(self.format)
        # self.file_hdl.setFormatter(self.format)
        self.time_hdl.setFormatter(self.format)
        # 핸들러 추가
        self.logger.addHandler(self.console_hdl)
        # self.logger.addHandler(self.file_hdl)
        self.logger.addHandler(self.time_hdl)

        # 출력 레벨 설정 (DEBUG < INFO < WARNING < ERROR < CRITICAL)
        self.logger.setLevel(logging.INFO)
