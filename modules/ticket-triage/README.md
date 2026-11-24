# Ticket Triage｜中文客服工單分流

客服收件匣同時收到扣款、配送、帳號與其他問題。這個離線 Python CLI 會把一張工單送往最匹配的隊列，或交給人工覆核。每筆決策都會列出第一名與第二名的相似度差距、採用的門檻及理由。訓練和推論只使用 Python 標準函式庫。

## 從訓練到分派

已訓練的 JSON 模型也能匯出為單一 ONNX 檔供 CPU 批次推論。工單字串先沿用原本的 Unicode 正規化與字元切分，圖內按列累計重複詞頻、套用 IDF、正規化並計算各隊列分數。詞彙順序、隊列名稱及已校正的門檻都隨模型保存。

```sh
python -m pip install --only-binary=:all: -r requirements-onnx.txt
python triage.py fit examples/train.jsonl examples/calibration.jsonl --threshold-mode per-queue --model model.json
python triage_onnx.py export model.json tickets.onnx
python triage_onnx.py route tickets.onnx "包裹還沒送到" "帳號密碼忘了"
```

匯出器支援原有的共用與逐隊列門檻。演練對兩種模型各比對 41 筆輸入，包含空字串、全形字與重複文字；每筆派送或轉人工的完整結果都與 Python 版相同。本機各類分數最大絕對差約為 3.33 × 10⁻¹⁶。對每種模型另跑 123 項補齊、批次組成與順序檢查。這些數字衡量部署前後的一致性，分類品質可用下方獨立保留集查核。[跨專案驗證](onnx_drill.py) · [Batch Parity](https://github.com/peca12968/handbook-search/tree/main/modules/ticket-triage/modules/batch-parity)

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

## 在 CI 檢查工單模型是否可替換

從 `modules/ticket-triage` 訓練現行與候選模型，然後把同一份三月評估集交給 `release_gate.py`。它檢查評估集的時間晚於兩份模型的校正期，並拒絕使用任一模型已見過的案例 ID。

```sh
python triage.py fit examples/train.jsonl examples/calibration.jsonl --min-calibration-cases 3 --model baseline.json
python triage.py fit examples/train.jsonl examples/calibration.jsonl --threshold-mode per-queue --min-calibration-cases 3 --model candidate.json
python release_gate.py baseline.json candidate.json examples/holdout.jsonl --min-cases 11 --min-queue-cases 2 --max-extra-reviews 3 --max-extra-reviews-per-queue 1
```

關卡要求總案例數、每個隊列的標註數、自動分派件數及精確率達標。預設需要至少一件自動分派且精確率至少 90%，新增誤派或新增覆核的預設上限都是零。`--min-assignments` 可調高自動分派樣本下限。總覆核量與依工單真實標籤分組的覆核量分開檢查。若候選改把工單送往新的錯誤隊列，就算一件新增誤派，即使現行模型原先也派錯。

| 11 件相同評估工單 | 人工覆核 | 新增誤派 | 結果 |
| --- | ---: | ---: | --- |
| 逐隊列門檻，維持原覆核上限 | 10 | 0 | 擋下，覆核比原先多 3 件 |
| 逐隊列門檻，允許全體多 3 件且每隊多 1 件 | 10 | 0 | 通過 |
| 共用門檻，校正精確率降為 70% | 2 | 2 | 擋下，新增誤派且評估精確率為 7/9 |

結束碼 `0` 代表達標，`1` 代表關卡擋下，`2` 代表輸入或參數無法評估。輸出的 JSON 記錄逐項門檻、改變的派送決策、模型與評估集 SHA-256。報告省略工單原文。`python release_demo.py` 可重現表中三種結果。[CLI](release_gate.py) · [測試](test_release_gate.py) · [展示資料](examples/release-demo.json)

## 方法與資料規則

### 逐隊列決定是否啟用自動分派

客服主管可以要求每個隊列各自累積足夠的校正案例。執行 `python compare_policies.py` 會用相同三份資料比較共用門檻與逐隊列門檻。兩組都以校正精確率至少 90%、門檻內至少 3 件為條件，模型存放於暫存目錄。若要保留逐隊列模型以便往後分派，可執行以下命令。

```sh
python triage.py fit examples/train.jsonl examples/calibration.jsonl --threshold-mode per-queue --min-precision 0.9 --min-calibration-cases 3 --model queue-model.json
python triage.py evaluate examples/holdout.jsonl --model queue-model.json
```

| 二月模型預測的第一順位 | 候選工單 | 至少收三件時的最佳校正結果 | 逐隊列策略 |
| --- | ---: | ---: | --- |
| 付款 | 3 | 3/3 正確 | 啟用，自動分派所需差距為 0.278395484 |
| 帳號 | 3 | 2/3 正確 | 交人工覆核 |
| 配送 | 3 | 2/3 正確 | 交人工覆核 |

二月另有一件沒有唯一候選。校正時按模型預測的第一順位分組，包含誤判與標為人工的案件。每組從符合件數及精確率的門檻中選覆蓋最多者；若沒有任何合格門檻，就記錄候選數不足或精確率未達標。`best_supported` 列出符合件數要求時精確率最高的工作點，精確率相同時取收件較多者。這一點可能和實際啟用的門檻不同。

| 三月的 11 件保留工單 | 自動分派 | 正確分派 | 人工覆核 | 覆蓋率 |
| --- | ---: | ---: | ---: | ---: |
| 共用門檻 | 4 | 4 | 7 | 36.4% |
| 逐隊列門檻 | 1 | 1 | 10 | 9.1% |

兩組都分派了付款工單 T-028。配送 T-026、帳號 T-027 因隊列未啟用改由人工處理；付款 T-031 因低於付款門檻也進入人工。三件在共用門檻下原本都分派正確。這批資料顯示，要求每個隊列各自達標會增加覆核量。每隊僅有三件校正候選，觀測到的精確率不足以推估正式服務的表現。[比較程式](compare_policies.py)

報告的 `by_candidate_queue` 列出各候選隊列的分派、覆核與正確數。平手及無詞彙匹配的案件單列為 `no_unique_candidate`。隊列沒有任何分派時精確率為 `null`。原有第 1 版模型仍以共用門檻分派；第 2 版模型保存各隊列的門檻與校正依據。

### 訓練、校正與推論

1. 對文字執行 NFKC、大小寫折疊與空白正規化，取相鄰二、三字元。從訓練資料計算平滑 IDF，以原始詞頻乘 IDF 並做 L2 正規化。每類將工單向量加總後再正規化，以餘弦相似度排序。[TF-IDF 定義](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
2. 將前兩名相似度差取到小數第九位。校正階段以觀測到的正差距為候選門檻，在滿足指定分派精確率的門檻中選擇覆蓋最多工單者。同分工單整組計入，沒有正差距的工單交給人工。若全部候選門檻都未達要求，模型會對所有工單回傳人工覆核。差距是排序分數，無法直接解讀為正確機率。
3. JSON Lines 每列需要唯一的 `case_id`、帶 UTC 偏移的 `at`、非空的 `text` 及 `queue`。訓練集至少包含兩個隊列。校正與保留資料可用 `null` 表示應交人工，此類工單若被自動分派，會計為錯誤。
4. 工具檢查相鄰時期的時間範圍及案例 ID。跨期重用 ID 或時間重疊會回報錯誤。JSON 模型保存詞權重、隊列中心、門檻、已用 ID 與時間界線；後續評估不會改寫模型。同一事件的多次聯絡應先整理成一筆案例，資料提供者負責識別內容近似但 ID 不同的工單。[實作](triage.py)

## 驗證

```sh
python -m pip install -r requirements-test.txt
python -m unittest -v
```

九項測試涵蓋 CLI 端到端分派、保留集評估、舊版模型、跨期資料檢查、平手及全人工狀態。數值測試使用 scikit-learn 的字元 TF-IDF 獨立計算 21 件校正與保留案例對三個隊列的分數，以 250 組固定種子樣本窮舉核對門檻選擇，再以 300 組含最低件數限制的樣本核對門檻及最佳可達精確率。[測試](test_triage.py) · [CI](../../.github/workflows/system.yml)

**履歷用語**　以 Python 實作中文客服工單分流，逐隊列校正自動分派門檻，用下一期工單量測分派精確率與人工覆核量，並以 scikit-learn 及窮舉搜尋交叉驗證分數與門檻。

本專案使用 AI 輔助開發，所有工單皆為合成資料。

[同庫模組與整合測試](modules/README.md)
