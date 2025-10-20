# signal_evaluator.py

import pandas as pd
import logging

# from logger_setup import APP_LOGGER_NAME
# logger = logging.getLogger(APP_LOGGER_NAME)
logger = logging.getLogger('stock_analyzer_app') # main.py などで設定済みのロガーを取得

def evaluate_signals(df_with_indicators: pd.DataFrame, signal_rules: list) -> tuple[str, dict]:
    """
    テクニカル指標が追加されたDataFrameとシグナルルールに基づき、売買シグナルを判定します。

    Args:
        df_with_indicators (pd.DataFrame): テクニカル指標の列が追加された株価データ。
                                           インデックスは日付型が望ましい。
        signal_rules (list): config.yamlから渡されるシグナルルールのリスト。
                             各ルールは辞書で、'rule_name', 'action', 'conditions' キーを持つ想定。
                             'conditions' は評価すべき条件式の文字列リスト。
                             例:
                             [
                                 {
                                     'rule_name': "ゴールデンクロス買い",
                                     'action': "BUY",
                                     'conditions': [
                                         "latest['SMA_5'] > latest['SMA_25']",
                                         "previous['SMA_5'] <= previous['SMA_25']",
                                         "latest['RSI_14'] < 70"
                                     ]
                                 },
                             ]

    Returns:
        tuple[str, dict]: (シグナル文字列, 判定に使用した主要指標の辞書)
                          シグナル: "BUY", "SELL", "HOLD" のいずれか。
                          指標辞書: 通知メッセージ用に使える指標名と値のペア。
                                    ルールに合致しなかった場合は空の辞書や情報メッセージ。
    """
    if df_with_indicators is None or df_with_indicators.empty or len(df_with_indicators) < 2:
        logger.warning("シグナル判定に必要なデータが不足しているため、HOLDとします。")
        return "HOLD", {"message": "データ不足のため判定不可"}

    latest = df_with_indicators.iloc[-1]
    previous = df_with_indicators.iloc[-2]

    # logger.debug(f"最新データ (latest):\n{latest}")
    # logger.debug(f"1つ前のデータ (previous):\n{previous}")

    final_signal = "HOLD"
    evaluation_details = {"message": "合致するシグナルルールなし"}
    
    for rule in signal_rules:
        rule_name = rule.get('rule_name', '無名ルール')
        action = rule.get('action')
        conditions = rule.get('conditions', [])

        if not action or not conditions:
            logger.warning(f"ルール '{rule_name}' にactionまたはconditionsが定義されていません。スキップします。")
            continue

        logger.debug(f"ルール '{rule_name}' (アクション: {action}) の評価を開始します。")
        all_conditions_met = True
        condition_str_being_evaluated = "" # エラー表示用

        try:
            for condition_str in conditions:
                condition_str_being_evaluated = condition_str # エラー表示用に保持
                local_vars = {'latest': latest, 'previous': previous, 'pd': pd}
                
                evaluated_value = eval(condition_str, {"__builtins__": {}}, local_vars)
                
                # 評価結果をブール値として解釈
                condition_is_true = bool(evaluated_value) 

                logger.debug(f"  条件 '{condition_str}' の評価結果(元): {evaluated_value}, ブール解釈: {condition_is_true}")
                
                if not condition_is_true:
                    all_conditions_met = False
                    break 

            if all_conditions_met:
                final_signal = action
                evaluation_details = {
                    "rule_applied": rule_name,
                    "message": f"ルール '{rule_name}' に合致しました。"
                }
                # config.yamlのtechnical_analysis_paramsを参照して、関連指標を抽出するのは今後の課題
                # 例としてRSI_14とSMA_5, SMA_25をevaluation_detailsに追加 (列名が固定と仮定)
                # これらはconfig.yamlのtechnical_analysis_paramsで定義された列名に合わせる必要あり
                # より汎用的にするには、ルールごとにどの指標値を詳細として含めるか定義できるようにする
                if 'RSI_14' in latest: # この列名はテストデータに合わせています
                     evaluation_details['RSI_14'] = round(latest['RSI_14'], 2)
                if 'SMA_5' in latest:
                     evaluation_details['SMA_5'] = round(latest['SMA_5'], 2)
                if 'SMA_25' in latest:
                     evaluation_details['SMA_25'] = round(latest['SMA_25'], 2)
                # 他にも、MACDの各値などを追加できます
                # if 'MACD_6_13_4' in latest: # この列名はテストデータに合わせています
                #      evaluation_details['MACD'] = round(latest['MACD_6_13_4'], 2)
                # if 'MACDs_6_13_4' in latest:
                #      evaluation_details['MACD_signal'] = round(latest['MACDs_6_13_4'], 2)


                logger.info(f"ルール '{rule_name}' に合致。シグナル: {final_signal}")
                break 

        except KeyError as e:
            logger.error(f"ルール '{rule_name}' の条件式評価中にキーエラー: {e}。DataFrameに必要な列が存在するか確認してください。条件式: '{condition_str_being_evaluated}'")
            all_conditions_met = False
        except Exception as e:
            logger.error(f"ルール '{rule_name}' の条件式評価中に予期せぬエラー: {e}。条件式: '{condition_str_being_evaluated}'")
            logger.exception("スタックトレース:")
            all_conditions_met = False

    if final_signal == "HOLD" and not signal_rules:
        logger.info("評価するシグナルルールが設定されていませんでした。")
        evaluation_details = {"message": "シグナルルール未設定"}
    elif final_signal == "HOLD" and evaluation_details.get("message") == "合致するシグナルルールなし": # メッセージが上書きされていない場合
        logger.info("いずれのシグナルルールにも合致しませんでした。")


    return final_signal, evaluation_details

# --- このモジュールを単体で実行した際のテストコード ---
if __name__ == '__main__':
    # 簡易的なロガー設定
    if not logging.getLogger('stock_analyzer_app').hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG, # DEBUGレベルで詳細なログを出力
            format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    logger.info("--- signal_evaluator.py 単体テスト開始 ---")

    # テスト用のDataFrame (technical_analyzer.pyから持ってきたようなイメージ)
    data_for_test = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'Close': [100, 101, 102, 103, 104],
        # テストケース1の条件に合うようにSMAの値を調整
        'SMA_5':  [None, None, 100.0, 101.0, 103.0], 
        'SMA_25': [None, None, 102.0, 101.0, 101.5], 
        'RSI_14': [None, None, 50.0,  60.0,  25.0] 
    }
    test_df_with_ta = pd.DataFrame(data_for_test)
    test_df_with_ta = test_df_with_ta.set_index('Date')
    
    logger.info("テスト用DataFrame (テクニカル指標付き):")
    print(test_df_with_ta)

    # テストケース1: ゴールデンクロス買いルール
    test_rules_buy = [
        {
            'rule_name': "テスト買いルール (ゴールデンクロス & RSI)",
            'action': "BUY",
            'conditions': [
                "latest['SMA_5'] > latest['SMA_25']",       # 最新で短期 > 長期 (103.0 > 101.5 -> True)
                "previous['SMA_5'] <= previous['SMA_25']",  # 1つ前は 短期 <= 長期 (101.0 <= 101.0 -> True)
                "latest['RSI_14'] < 70"                     # RSIが70未満 (25.0 < 70 -> True)
            ]
        }
    ]
    
    logger.info("\n--- テストケース1: 買いシグナル ---")
    # test_df_with_ta をそのまま使用 (上記のダミーデータはテストケース1に合うように調整済み)
    print("評価用DataFrame (最新2行):")
    print(test_df_with_ta.tail(2))
    signal, details = evaluate_signals(test_df_with_ta, test_rules_buy)
    logger.info(f"シグナル: {signal}, 詳細: {details}")
    assert signal == "BUY", f"テストケース1失敗: 期待値 BUY, 実際値 {signal}"

    # テストケース2: デッドクロス売りルール (条件に合致しないケース)
    test_rules_sell = [
        {
            'rule_name': "テスト売りルール (デッドクロス)",
            'action': "SELL",
            'conditions': [
                "latest['SMA_5'] < latest['SMA_25']",       # (103.0 < 101.5 -> False)
                "previous['SMA_5'] >= previous['SMA_25']"
            ]
        }
    ]
    logger.info("\n--- テストケース2: 売りシグナル (合致しないはず) ---")
    print("評価用DataFrame (最新2行):")
    print(test_df_with_ta.tail(2))
    signal, details = evaluate_signals(test_df_with_ta, test_rules_sell)
    logger.info(f"シグナル: {signal}, 詳細: {details}")
    assert signal == "HOLD", f"テストケース2失敗: 期待値 HOLD, 実際値 {signal}"

    # テストケース3: RSI売られすぎ買いルール
    test_rules_rsi_buy = [
        {
            'rule_name': "RSI売られすぎ買い",
            'action': "BUY",
            'conditions': [
                "latest['RSI_14'] < 30" # (25.0 < 30 -> True)
            ]
        }
    ]
    logger.info("\n--- テストケース3: RSI売られすぎ買い ---")
    print("評価用DataFrame (最新1行):")
    print(test_df_with_ta.tail(1))
    signal, details = evaluate_signals(test_df_with_ta, test_rules_rsi_buy)
    logger.info(f"シグナル: {signal}, 詳細: {details}")
    assert signal == "BUY", f"テストケース3失敗: 期待値 BUY, 実際値 {signal}"

    # テストケース4: データ不足
    logger.info("\n--- テストケース4: データ不足 ---")
    signal, details = evaluate_signals(test_df_with_ta.head(1), test_rules_buy) # 1行だけのデータ
    logger.info(f"シグナル: {signal}, 詳細: {details}")
    assert signal == "HOLD", f"テストケース4失敗: 期待値 HOLD, 実際値 {signal}"
    assert details.get('message') == "データ不足のため判定不可", "テストケース4詳細メッセージ不一致"

    logger.info("--- signal_evaluator.py 単体テスト終了 ---")