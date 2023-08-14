#!/bin/bash
# 변수 설정
SORT=$1
START_DT=$2
END_DT=$3

# directory 만들기
# 엑셀 저장 dir
if [ "$SORT" == "train" ] || [ "$SORT" == "test" ] || [ "$SORT" == "result" ]; then
  BACKUP_PATH="/workspace/project-workspace/sool/data/base/"
  if [ -d ${BACKUP_PATH}${START_DT} ]; then
          echo "[이미 경로 존재, 기존 경로 : ${BACKUP_PATH}${START_DT}]"
  else
          mkdir ${BACKUP_PATH}${START_DT}
          chmod +777 ${BACKUP_PATH}${START_DT}
          echo "[${BACKUP_PATH}${START_DT} 경로 생성 완료]"
  fi
else
  echo "다시 입력하세요"
  exit
fi
# 학습 데이터 다운
if [ "$SORT" == "train" ]; then
  ## train data :지난달 item_info_final, 변경한 cumul 저장
  aws s3 cp \--recursive \--exclude _SUCCESS s3://sool/Base/$START_DT/Model/item_info_FINAL.parquet ${BACKUP_PATH}
  mv ${BACKUP_PATH}part*.parquet ${BACKUP_PATH}${START_DT}/item_info_FINAL.parquet
  aws s3 cp \--recursive \--exclude _SUCCESS s3://sool/Base/$START_DT/Model/NewCumul/cumul_nminfo_df.parquet ${BACKUP_PATH}
  mv ${BACKUP_PATH}part*.parquet ${BACKUP_PATH}${START_DT}/cumul_nminfo_df.parquet
  # 파일 존재할 경우에만 python code 실행
  item_file=${BACKUP_PATH}${START_DT}/item_info_FINAL.parquet
  if [ -f "${item_file}" ];then
          echo "[다운 완료 & run.py 실행시작]"
          /opt/conda/envs/sebae/tf27/bin/python3 /workspace/project-workspace/sool/run.py ${SORT} ${START_DT} ${END_DT}
  else
          echo "[item_info_FINAL 파일이 존재하지 않습니다.]"
          exit
  fi
# 테스트 데이터 다운
elif [ "$SORT" == "test" ]; then
  aws s3 cp s3://sool/Base/$END_DT/Model/item_info_AFTER.xlsx ${BACKUP_PATH}${START_DT}
  after_file=${BACKUP_PATH}${START_DT}/item_info_AFTER.xlsx
  if [ -f "${after_file}" ];then
          echo "[다운 완료 & run.py 실행시작]"
          /opt/conda/envs/sebae/tf27/bin/python3 /workspace/project-workspace/sool/run.py ${SORT} ${START_DT} ${END_DT}
  else
          echo "[item_info_AFTER 파일이 존재하지 않습니다.]"
          exit
  fi
elif [ "$SORT" == "result" ]; then
  RESULT_PATH="/workspace/project-workspace/sool/data/result/"
  # n_gram 결과 다운
  aws s3 cp s3://sool/Base/$END_DT/Model/item_info_AFTER.xlsx ${RESULT_PATH}${END_DT}
  ng_file=${RESULT_PATH}${END_DT}/item_info_AFTER.xlsx
  if [ -f "${ng_file}" ];then
          echo "[다운 완료 & run.py 실행시작]"
          /opt/conda/envs/sebae/tf27/bin/python3 /workspace/project-workspace/sool/run.py ${SORT} ${START_DT} ${END_DT}
  else
          echo "[item_info_AFTER 파일이 존재하지 않습니다.]"
          exit
  fi
  # python 제대로 완료 시에만 s3 업로드
  status=$(grep INFO /workspace/project-workspace/sool/data/log/ITEM_MODEL.log | tail -n 1)
  if [[ "$status" == *"SUCCESS"* ]]; then
          echo "[run.py 정상 종료]"
          file_name=${RESULT_PATH}${END_DT}/item_matching_result.xlsx
          if [ -f "${file_name}" ];then
            aws s3 cp $file_name s3://sool/Base/${END_DT}/Model/item_matching_result.xlsx
          else
            echo "결과파일 없음, 확인바랍니다."
          fi
  else
          echo "[run.py 에러, 실행 중지합니다.]"
          exit
  fi
fi






