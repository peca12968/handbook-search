# Ticket Triage｜中文客服工單分流

客服收件匣同時收到扣款、配送、帳號與其他問題。這個離線 Python CLI 會把一張工單送往最匹配的隊列，或交給人工覆核。每筆決策都會列出第一名與第二名的相似度差距、採用的門檻及理由。訓練和推論只使用 Python 標準函式庫。

## 從訓練到分派

需要 Python 3.12 或 3.13。從專案根目錄依序執行。

```sh
python triage.py fit examples/train.jsonl examples/calibration.jsonl --min-precision 0.9 --model model.json
python triage.py route "無法登入帳號，重設密碼連結過期" --model model.json
python triage.py route "宅配地址需要更正" --model model.json
python triage.py evaluate examples/holdout.jsonl --model model.json
```

第一張工單分派給「帳號」。第二張雖然與配送文字有關，但兩個領先類別的分數差距約 0.169，小於校正所得門檻 0.255444131，因此回傳 `review` 與 `below_threshold`。人工覆核是工具的輸出狀態，實際派件由使用工具的系統決定。

## 如何解讀結果

每件合成案例只出現在一個時期。先用一月份建立詞彙、詞權重與各隊列的向量中心，再用二月份的標註選門檻，最後在三月份讀取既有模型進行評估。

| 資料 | 工單數 | 自動分派 | 分派正確 | 交給人工 | 自動覆蓋率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 一月訓練集 | 15 | — | — | — | — |
| 二月校正集 | 10 | 7 | 7 | 3 | 70.0% |
| 三月保留集 | 11 | 4 | 4 | 7 | 36.4% |

三月有 8 件具備明確隊列，4 件獲得正確分派，另外 4 件也被送到人工；其餘 3 件原先就標為需要人工處理。分派精確率的分母是自動分派的 4 件，覆蓋率的分母是全部 11 件。`--min-precision` 約束校正樣本上的表現。這組小型合成資料適合重現流程，實際收件匣的表現仍需另外標註及驗證。[訓練資料](examples/train.jsonl) · [校正資料](examples/calibration.jsonl) · [逐案保留資料](examples/holdout.jsonl)

## 方法與資料規則

1. 對文字執行 NFKC、大小寫折疊與空白正規化，取相鄰二、三字元。從訓練資料計算平滑 IDF，以原始詞頻乘 IDF 並做 L2 正規化。每類將工單向量加總後再正規化，以餘弦相似度排序。[TF-IDF 定義](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
2. 將前兩名相似度差取到小數第九位。校正階段以觀測到的正差距為候選門檻，在滿足指定分派精確率的門檻中選擇覆蓋最多工單者。同分工單整組計入，沒有正差距的工單交給人工。若全部候選門檻都未達要求，模型會對所有工單回傳人工覆核。差距是排序分數，無法直接解讀為正確機率。
3. JSON Lines 每列需要唯一的 `case_id`、帶 UTC 偏移的 `at`、非空的 `text` 及 `queue`。訓練集至少包含兩個隊列。校正與保留資料可用 `null` 表示應交人工，此類工單若被自動分派，會計為錯誤。
4. 工具檢查相鄰時期的時間範圍及案例 ID。跨期重用 ID 或時間重疊會回報錯誤。JSON 模型保存詞權重、隊列中心、門檻、已用 ID 與時間界線；後續評估不會改寫模型。同一事件的多次聯絡應先整理成一筆案例，資料提供者負責識別內容近似但 ID 不同的工單。[實作](triage.py)

## 驗證

```sh
python -m pip install -r requirements-test.txt
python -m unittest -v
```

五項測試涵蓋 CLI 端到端分派、保留集評估、跨期資料檢查、平手與全人工狀態。數值測試使用 scikit-learn 的字元 TF-IDF 獨立計算 21 件校正與保留案例對三個隊列的分數，另以 250 組隨機樣本窮舉核對門檻選擇。[測試](test_triage.py) · [CI](.github/workflows/ci.yml)

**履歷用語**　以 Python 實作可交由人工覆核的中文工單分類器，依時間切分訓練、門檻校正與保留資料，同時量測自動分派精確率與覆蓋率，並用 scikit-learn 交叉驗證相似度。

本專案使用 AI 輔助開發，所有工單皆為合成資料。
