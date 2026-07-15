TRICK_CONFIG = {
    "inside_loop": {
        "axis": "gy",
        "peak_std": 1.0,
        "valley_std": 1.0,
        "max_loop_sec": 1.0,
        "stable_threshold_ratio": 0.30,
    },

    "outside_loop": {
        "axis": "gy",
        "peak_std": 1.0,
        "valley_std": 1.0,
        "max_loop_sec": 1.0,
        "stable_threshold_ratio": 0.30,
    },

    "inout_loop": {
        "axis": "gy",
        # イン側の閾値
        "peak_std_in": 1.0,
        "valley_std_in": 1.0,
        # アウト側の閾値（振幅が小さいためインよりも閾値を低めに設定する想定）
        "peak_std_out": 0.8,
        "valley_std_out": 0.8,
        # サブループ（インまたはアウト）1周あたりの最大秒数
        # 2周を1周として拾うのを防ぐため、1.0秒より短めに設定する
        "max_sub_loop_sec": 0.9,
        # イン+アウトを合わせた1周の最大秒数（必要なら使う）
        "max_loop_sec": 1.5,
        "stable_threshold_ratio": 0.30,
    }
}