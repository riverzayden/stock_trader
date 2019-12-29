import os
import sys
import logging
import settings


stock_code = '005930'  # 삼성전자

# 로그 기록
log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
timestr = settings.get_time_str()
if not os.path.exists('logs/%s' % stock_code):
    os.makedirs('logs/%s' % stock_code)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)


from policy_learner import PolicyLearner
import data_manager
import psycopg2 as pg2 
import pandas as pd
import numpy as np
if __name__ == '__main__':
    # 주식 데이터 준비
    conn = pg2.connect("host = localhost dbname=stock user=postgres password=x port=5432")
    cur = conn.cursor()
    cur.execute("select * from stock_price where company='{}'".format('삼성전자'))
    rows = cur.fetchall()
    #max_date = pd.DataFrame(rows)[0].min()
    chart_data = pd.DataFrame(rows,columns=['company','code','date','open','high','low','close','diff', 'volume'])
    chart_data = chart_data[['date','open','high','low','close','volume']]
    chart_data['date']= [i[0:4]+'-'+i[4:6]+'-'+i[6:8] for i in chart_data['date'].values]
    chart_data['open'] = chart_data['open'].astype('int64')
    chart_data['high'] = chart_data['high'].astype('int64')
    chart_data['low'] = chart_data['low'].astype('int64')
    chart_data['close'] = chart_data['close'].astype('int64')
    chart_data['volume'] = chart_data['volume'].astype('int64')

    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2018-05-04') &
                                  (training_data['date'] <= '2019-06-01')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]

    # 강화학습 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=10, max_trading_unit=20, delayed_reward_threshold=.05, lr=.0001)
    policy_learner.fit(balance=10000000, num_epoches=1000,
                       discount_factor=0, start_epsilon=.5)

    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
