import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


if __name__ == '__main__':
    stock_code = '005930'  # 삼성전자
    model_ver = '20191229182138'

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

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
    training_data = training_data[(training_data['date'] >= '2019-01-01') &
                                  (training_data['date'] <= '2019-12-27')]
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

    # 비 학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=3)
    policy_learner.trade(balance=10000000,
                         model_path=os.path.join(
                             settings.BASE_DIR,
                             'models/{}/model_{}.h5'.format(stock_code, model_ver)))
