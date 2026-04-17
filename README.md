# YoYo Inside Loop Analysis

ヨーヨーのトリック「インサイドループ」を解析するWebアプリです。  
センサーデータ（加速度・ジャイロ）から動作を評価します。

---

## 🚀 デモ

ローカルで動作可能（公開予定）

---

## 🛠 機能

- CSVアップロード
- トリック解析
- グラフ表示
- 履歴保存

---

## 📦 技術スタック

- Python (Flask)
- SQLite
- HTML / JavaScript

---

## 💻 ローカル実行方法
ターミナル開いて

```bash
git clone https://github.com/あなたのユーザー名/yoyo-analysis.git
cd yoyo-analysis

python -m venv venv
source venv/bin/activate

pip install flask flask-cors pandas numpy matplotlib scipy fastdtw ahrs

python app.py
```

別のターミナルで
```bash
python -m http.server 3000
```

ブラウザで以下リンクを開く
```
http://localhost:3000/index.html
```

## ⚠️ 注意
- APIのURLはローカル用に変更しています
- results.db は.gitignoreで除外しています