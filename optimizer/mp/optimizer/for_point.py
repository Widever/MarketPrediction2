import dataclasses

import pandas as pd

import mp.optimizer.init_data as data
import mp.optimizer.mark as mark
from mp.optimizer.comb_v2 import CombGrade, get_select_combs_mask_ignore_scope


def build_marked_points_df_for_point(symbol: str):
    ohlcv_df = data.CURRENCY_DATA_DICT[symbol].ohlcv_df

    marked_points_df_data = []
    for idx, row in ohlcv_df.iterrows():
        timestamp = int(row["timestamp"])
        marked_point_dict = {
            "timestamp": timestamp,
        }

        point_values_ = mark.point_values(symbol, timestamp)
        point_values_dict = dataclasses.asdict(point_values_)
        marked_point_dict.update(point_values_dict)
        marked_points_df_data.append(marked_point_dict)

    marked_points_df = pd.DataFrame(marked_points_df_data)
    marked_points_df = mark.add_tags_for_point_values(marked_points_df)
    marked_points_df = marked_points_df.round(5)
    return marked_points_df

def check_combs_for_point(symbol) -> bool:
    if symbol == "ADAUSDT":
        combs: list[CombGrade] = [
            CombGrade(comb=('#tag_avg_ampl_gt_0.01', '#tag_ada_rise_from_low_lt_0.014', '#tag_ada_log_return_gt_-0.004',
                            '#tag_eth_drop_from_high_gt_0.015', '#tag_doge_ampl_gt_0.009',
                            '#tag_doge_xrp_ampl_ratio_lt_1.701', '#tag_eth_ada_ampl_ratio_lt_1.571',
                            '#tag_btc_eth_log_return_ratio_gt_-1.14', '#tag_doge_ada_drop_from_high_ratio_lt_2.71',
                            '#tag_eth_ada_log_return_ratio_lt_5.468', '#tag_doge_sui_log_return_ratio_lt_4.875',
                            '#tag_avg_log_return_ratio_gt_-0.767', '#tag_eth_doge_ampl_ratio_lt_1.595',
                            '#tag_doge_sui_ampl_ratio_lt_1.633', '#tag_xrp_ampl_gt_0.006'),
                      bayesian_winrate=0.648936170212766, profit=51, total=74),
            CombGrade(comb=('#tag_ada_ampl_gt_0.004', '#tag_doge_ada_log_return_ratio_gt_2.868',
                            '#tag_ada_drop_from_high_lt_0.013', '#tag_doge_sui_ampl_ratio_lt_1.178',
                            '#tag_eth_rise_from_low_lt_0.025', '#tag_eth_ada_ampl_ratio_gt_0.54',
                            '#tag_doge_sui_log_return_ratio_lt_3.042', '#tag_doge_ampl_lt_0.009',
                            '#tag_xrp_rise_from_low_lt_0.052', '#tag_btc_xrp_drop_from_high_ratio_lt_3.327',
                            '#tag_doge_ada_drop_from_high_ratio_lt_7.729', '#tag_avg_rise_from_low_lt_0.043',
                            '#tag_btc_doge_log_return_ratio_lt_2.342', '#tag_avg_log_return_ratio_gt_-3.555',
                            '#tag_avg_ampl_ratio_lt_1.181'), bayesian_winrate=0.3279569892473118, profit=51, total=166),
        ]
    elif symbol == "ETHUSDT":
        combs: list[CombGrade] = [
            CombGrade(comb=('#tag_eth_ampl_gt_0.008', '#tag_eth_ada_log_return_ratio_lt_0.338',
                            '#tag_eth_drop_from_high_gt_0.021', '#tag_sui_ampl_gt_0.013',
                            '#tag_avg_log_return_gt_-0.004', '#tag_eth_log_return_lt_0.003',
                            '#tag_btc_eth_ampl_ratio_lt_0.963', '#tag_eth_xrp_log_return_ratio_lt_3.16',
                            '#tag_xrp_drop_from_high_gt_0.014', '#tag_doge_sui_ampl_ratio_lt_1.633',
                            '#tag_eth_doge_drop_from_high_ratio_lt_3.047', '#tag_xrp_ampl_gt_0.008',
                            '#tag_btc_ada_ampl_ratio_lt_1.06', '#tag_btc_doge_ampl_ratio_lt_1.14',
                            '#tag_btc_xrp_ampl_ratio_lt_0.974'), bayesian_winrate=0.7192982456140351, profit=31,
                      total=37),
            CombGrade(comb=('#tag_eth_ampl_gt_0.004', '#tag_btc_eth_log_return_ratio_lt_-0.129',
                            '#tag_doge_ada_ampl_ratio_lt_0.775', '#tag_eth_drop_from_high_lt_0.015',
                            '#tag_btc_ada_drop_from_high_ratio_lt_1.57', '#tag_sui_rise_from_low_lt_0.037',
                            '#tag_xrp_ampl_gt_0.003', '#tag_ada_log_return_gt_-0.004',
                            '#tag_btc_doge_log_return_ratio_gt_-1.316', '#tag_doge_sui_ampl_ratio_lt_1.178',
                            '#tag_avg_log_return_ratio_gt_-3.555', '#tag_btc_ampl_lt_0.006',
                            '#tag_btc_eth_rise_from_low_ratio_lt_4.374', '#tag_avg_rise_from_low_lt_0.043',
                            '#tag_eth_ada_log_return_ratio_lt_5.468'), bayesian_winrate=0.5774647887323944,
                      profit=31, total=51)
        ]
    elif symbol == "DOGEUSDT":
        combs: list[CombGrade] = [
            CombGrade(
                # {'l': 15, 'm': 45, 'get_by_profit_n': 50, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 1}
                # Total: total_count=np.int64(68), total_sl_count=np.int64(10), total_k=np.float64(5.8)
                comb=('#tag_avg_ampl_gt_0.01', '#tag_doge_drop_from_high_gt_0.031', '#tag_avg_log_return_gt_-0.004',
                      '#tag_eth_log_return_lt_0.003', '#tag_doge_xrp_log_return_ratio_gt_-1.226',
                      '#tag_xrp_ampl_gt_0.006', '#tag_eth_ada_drop_from_high_ratio_lt_2.114',
                      '#tag_avg_rise_from_low_lt_0.033', '#tag_btc_ada_ampl_ratio_lt_1.06',
                      '#tag_eth_xrp_drop_from_high_ratio_lt_2.76', '#tag_avg_drop_from_high_gt_0.021',
                      '#tag_eth_ada_log_return_ratio_lt_5.468', '#tag_eth_doge_log_return_ratio_lt_6.527',
                      '#tag_btc_doge_ampl_ratio_lt_1.14', '#tag_btc_xrp_ampl_ratio_lt_1.277'),
                bayesian_winrate=0.6746987951807228, profit=46, total=63),
            # {'l': 15, 'm': 50, 'get_by_profit_n': 50, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 2}
            # Total: total_count=np.int64(209), total_sl_count=np.int64(55), total_k=np.float64(2.8)
            CombGrade(comb=('#tag_doge_ampl_gt_0.006', '#tag_doge_xrp_log_return_ratio_lt_0.566',
                            '#tag_avg_drop_from_high_lt_0.021', '#tag_doge_ada_ampl_ratio_lt_1.21',
                            '#tag_xrp_ampl_lt_0.011', '#tag_doge_sui_ampl_ratio_lt_1.178',
                            '#tag_btc_xrp_drop_from_high_ratio_lt_2.235', '#tag_sui_ampl_lt_0.013',
                            '#tag_eth_xrp_rise_from_low_ratio_lt_2.472', '#tag_btc_ada_log_return_ratio_lt_2.419',
                            '#tag_avg_ampl_gt_0.005', '#tag_eth_ada_rise_from_low_ratio_lt_2.69',
                            '#tag_eth_xrp_log_return_ratio_lt_3.16', '#tag_btc_doge_ampl_ratio_lt_1.14',
                            '#tag_doge_log_return_lt_0.005'), bayesian_winrate=0.3096446700507614, profit=51,
                      total=177)
        ]
    elif symbol == "XRPUSDT":
        combs: list[CombGrade] = [
            # {'l': 15, 'm': 40, 'get_by_profit_n': 50, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 1}
            # Total: total_count=np.int64(86), total_sl_count=np.int64(20), total_k=np.float64(3.3)
            CombGrade(comb=('#tag_xrp_ampl_gt_0.011', '#tag_doge_xrp_log_return_ratio_gt_1.462',
                            '#tag_xrp_drop_from_high_gt_0.027', '#tag_eth_doge_log_return_ratio_lt_1.081',
                            '#tag_btc_eth_rise_from_low_ratio_lt_2.24', '#tag_sui_rise_from_low_lt_0.055',
                            '#tag_avg_ampl_ratio_lt_0.989', '#tag_btc_ada_log_return_ratio_lt_2.419',
                            '#tag_avg_log_return_ratio_gt_-0.767', '#tag_btc_xrp_ampl_ratio_lt_0.974',
                            '#tag_doge_ada_ampl_ratio_lt_2.081', '#tag_btc_eth_drop_from_high_ratio_lt_4.21',
                            '#tag_doge_xrp_drop_from_high_ratio_lt_3.01',
                            '#tag_btc_doge_log_return_ratio_gt_-1.316', '#tag_btc_xrp_log_return_ratio_lt_5.201'),
                      bayesian_winrate=0.504950495049505, profit=41, total=81)
        ]
    elif symbol == "AVAXUSDT":
        combs: list[CombGrade] = [
            # {'l': 15, 'm': 35, 'get_by_profit_n': 50, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 1}
            # Total: total_count=np.int64(51), total_sl_count=np.int64(12), total_k=np.float64(3.25)
            CombGrade(
                comb=('#tag_avg_ampl_gt_0.01', '#tag_doge_drop_from_high_gt_0.031', '#tag_xrp_log_return_gt_-0.004',
                      '#tag_avg_log_return_lt_0.004', '#tag_eth_drop_from_high_gt_0.021', '#tag_eth_ampl_gt_0.008',
                      '#tag_doge_sui_log_return_ratio_lt_3.042', '#tag_eth_xrp_ampl_ratio_lt_1.646',
                      '#tag_doge_ada_ampl_ratio_lt_1.646', '#tag_btc_eth_log_return_ratio_gt_-1.14',
                      '#tag_doge_sui_ampl_ratio_lt_1.633', '#tag_doge_ampl_gt_0.009',
                      '#tag_doge_rise_from_low_lt_0.074', '#tag_eth_xrp_log_return_ratio_lt_10.314',
                      '#tag_doge_log_return_lt_0.005'), bayesian_winrate=0.71875, profit=36, total=44),
            # {'l': 15, 'm': 30, 'get_by_profit_n': 30, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 2}
            # Total: total_count=np.int64(111), total_sl_count=np.int64(26), total_k=np.float64(3.269230769230769)
            CombGrade(comb=('#tag_avg_ampl_gt_0.005', '#tag_doge_sui_log_return_ratio_lt_-0.623',
                            '#tag_eth_doge_rise_from_low_ratio_lt_0.877', '#tag_xrp_ampl_lt_0.008',
                            '#tag_avg_drop_from_high_lt_0.021', '#tag_eth_ada_drop_from_high_ratio_lt_2.114',
                            '#tag_btc_xrp_ampl_ratio_lt_0.974', '#tag_btc_doge_drop_from_high_ratio_lt_3.461',
                            '#tag_doge_ada_rise_from_low_ratio_lt_2.957', '#tag_sui_ampl_gt_0.004',
                            '#tag_doge_ada_drop_from_high_ratio_lt_2.71', '#tag_eth_doge_log_return_ratio_lt_2.897',
                            '#tag_doge_log_return_gt_-0.005', '#tag_xrp_log_return_gt_-0.004',
                            '#tag_avg_log_return_lt_0.004'), bayesian_winrate=0.3333333333333333, profit=31,
                      total=103)
        ]
    elif symbol == "SUIUSDT":
        combs: list[CombGrade] = [
            # {'l': 15, 'm': 55, 'get_by_profit_n': 50, 'get_by_bayesian_winrate_n': 100, 'profit_signal_mode': 1}
            # Total: total_count=np.int64(89), total_sl_count=np.int64(20), total_k=np.float64(3.45)
            CombGrade(
                comb=('#tag_sui_ampl_gt_0.013', '#tag_sui_rise_from_low_lt_0.02', '#tag_avg_log_return_gt_-0.004',
                      '#tag_eth_drop_from_high_gt_0.021', '#tag_doge_ampl_gt_0.011',
                      '#tag_sui_drop_from_high_gt_0.017', '#tag_doge_xrp_ampl_ratio_lt_1.701',
                      '#tag_doge_drop_from_high_gt_0.016', '#tag_btc_xrp_ampl_ratio_lt_0.974',
                      '#tag_eth_doge_drop_from_high_ratio_lt_2.298', '#tag_eth_doge_log_return_ratio_lt_4.712',
                      '#tag_doge_ada_ampl_ratio_lt_1.646', '#tag_xrp_drop_from_high_gt_0.014',
                      '#tag_eth_ada_log_return_ratio_lt_5.468', '#tag_eth_xrp_log_return_ratio_lt_6.737'),
                bayesian_winrate=0.6875, profit=56, total=76)
        ]
    else:
        raise RuntimeError(f"Unknown main symbol: {symbol}.")

    marked_points_df = build_marked_points_df_for_point(symbol)

    select_mask = get_select_combs_mask_ignore_scope(marked_points_df, [comb_.comb for comb_ in combs])
    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    last_ts = marked_points_df.iloc[-1]["timestamp"]
    is_in_selected = (selected_df["timestamp"] == last_ts).any()

    return is_in_selected