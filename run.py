import sys
import pandas as pd
from src.common.config import mk_init
from src.common.log import Log
from src.models.predict_dnn_model import predict_dnn
from src.models.train_dnn_model import train_dnn_model
from src.models.train_svm_model import run_train_svm_model
from src.models.predict_svm_model import predict_svm
from src.tokenizer.run_tokenizer import run_train_tokenizer, run_test_tokenizer
from src.models.model_result import mg_result_not_answer
import warnings
import os
import os.path
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import requests
import datetime


def send_msg_fun(flag, task, file):
    startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    level = "INFO"
    url = "https://hooks.slack.com/services/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"
    payload = {"blocks": [
        {"type": "section",
         "text": {
             "type": "mrkdwn",
             "text": "[{} : {}] \n{}  `{}`  {}".format(flag, task, startTime, level, file)}
         }
    ]}
    return requests.post(url, json=payload)
def send_error_msg_fun(flag, task, file, error):
    startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    level = "WARN"
    url = "https://hooks.slack.com/services/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"
    payload = {"blocks": [
        {"type": "section",
         "text": {
             "type": "mrkdwn",
             "text": "[{} : {}] \n{}  `{}`  {} \n Error Message: {}".format(flag, task, startTime, level, file, error)}
         }
    ]}
    return requests.post(url, json=payload)
def send_slack(flag, task,file):
    rst = send_msg_fun(flag, task, file)
    if rst.status_code != 200:
        rst = send_msg_fun(flag, task, file)
        if rst.status_code != 200:
            print("슬랙 전송 오류")
            logger.info("slack error")
def send_error_slack(flag, task, file, ex):
    rst = send_error_msg_fun(flag, task, file, ex)
    
    if rst.status_code != 200:
        rst = send_error_msg_fun(flag, task, file)
        if rst.status_code != 200:
            print("슬랙 전송 오류")
            logger.info("slack error")
            

def setting_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
    # os.environ["NCCL_SHM_DISABLE"]='1'
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    if len(tf.config.experimental.list_physical_devices('GPU')) == mirrored_strategy.num_replicas_in_sync:
        print("[GPUs count: {}]".format(len(tf.config.experimental.list_physical_devices('GPU'))))
        return mirrored_strategy
    else:
        raise Exception("GPU 설정 오류")


# 이번달 데이터에 누적변경값 적용 후 토큰화
def run_mk_train_dataset(logger, train_dt):
    # train data set 저장
    data_path = 'data/base/'+train_dt
    item_info = pd.read_parquet(data_path + "/item_info_FINAL.parquet")
    cumul = pd.read_parquet(data_path + "/cumul_nminfo_df.parquet")
    run_train_tokenizer(item_info, cumul, logger, train_dt, 150000)


# test여도 train 경로에 저장
def run_mk_test_dataset(logger, train_dt):
    test_path = 'data/base/'+train_dt
    test_df = pd.read_excel(test_path+"/item_info_AFTER.xlsx")
    run_test_tokenizer(test_df, logger, train_dt)


def main(answer, train_dt, test_dt):
    try:
        # 로그
        log = Log()
        logger = log.logger
        logger.info("START")
        if answer == 'train' or answer == 'all':
            send_slack("dti", "비모집단 모집단 예측 모델 학습 시작","run(gpu)")
            # tokenize
            run_mk_train_dataset(logger, train_dt)
            # train model
            run_train_svm_model(logger, train_dt)
            mirrored_strategy = setting_gpu()
            train_dnn_model(logger, train_dt, mirrored_strategy, 150000)
            send_slack("dti", "비모집단 모집단 예측 모델 학습 완료","run(gpu)")
            logger.info("SUCCESS")
        if answer == 'test' or answer == 'all':
            send_slack("dti", "비모집단 모집단 예측 모델 테스트 시작","run(gpu)")
            run_mk_test_dataset(logger, train_dt)
            # predict
            predict_svm(logger, train_dt, test_dt)
            mirrored_strategy = setting_gpu()
            predict_dnn(logger, train_dt, test_dt, mirrored_strategy)
            send_slack("dti", "비모집단 모집단 예측 모델 테스트 완료","run(gpu)")
            logger.info("SUCCESS")
        if answer == 'result' or answer == 'all':
            # model multi check
            # mg_result_answer(logger, test_dt)
            mg_result_not_answer(logger, test_dt)
            send_slack("dti", "비모집단 모집단 예측 모델 결과 업로드 완료","run(gpu)")
            logger.info("SUCCESS")
    except Exception as e:
        logger.error(e)
        send_error_slack("dti", "비모집단 모집단 예측 모델 결과 에러","run(gpu)", e)


if __name__ == '__main__':

    sort = sys.argv[1]
    train_dt = sys.argv[2]
    test_dt = sys.argv[3]

    # 1. 폴더 없을 경우 생성
    mk_init().mk_dir()
    mk_init().mk_dir_dt(train_dt, 'train')
    mk_init().mk_dir_dt(test_dt, 'test')

    # 2. 실행
    main(sort, train_dt, test_dt)

