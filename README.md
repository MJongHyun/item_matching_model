# item_matching_model
    주류 아이템 태깅 모델 개발

### 목적
        - 2가지 모델(SVM, DNN)과 기존 n_gram 모델로 중복검증을 실시해 모집단, 비모집단 구분
                
### 실행 방법
        - 모델 학습 시 
            ./run_item_model.sh train {train date} {test date}
        - 학습된 모델로 예측시
            ./run_item_model.sh test {train date} {test date}
        - 결과 업로드시 
            ./run_item_model.sh result {train date} {test date}

### 버전관리
        + version 0.1.0 : local model version 
        + version 1.0.0 : server version

### 개발 참고 사항 
        + local version (참고용)
        DNN model train 약 1~2시간 소요 
        SVM model train 약 40분 소요
        전체 실행시 약 3시간 소요
        + server version 
        DNN model train 약 5분 소요 (GPU)
        SVM model train 약 20분 소요 (CPU)
        train 약 40분 소요
        + 참고
        모든 train, test 데이터 및 모델은 학습 기준 날짜 폴더에 저장됨. 
        (test 데이터 생성시 토큰이 학습 기준으로 생성되기때문에 학습 기준 날짜 폴더에 저장)

        실행 시 필요 데이터 
        1. train 
            - item_info_FINAL.parquet (수동검증까지 완료한 학습 데이터)
            - cumul_nminfo_df.parquet (수동검증 후 신제품 반영까지 한 누적사전, dnn 학습)

       2. test
           - item_info_AFTER.xlsx (n_gram 결과)
