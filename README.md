# Handbook Search｜可追溯來源的中英文手冊搜尋

值班時搜尋「付款重送」，如何知道命中哪份手冊、哪個段落，以及它是否真的回答了問題？這個 CLI 從 Markdown 建立本機全文索引，回傳原文、標題、相對路徑及行號，並可用有相關段落標註的問題集檢查排名。使用 Python 標準函式庫與 SQLite FTS5。

## 三分鐘展示

需要含 SQLite FTS5 的 Python 3.12 或 3.13。在專案目錄執行：

```sh
python handbook.py index examples/handbook
python handbook.py query "付款重送 payment retry" -k 3
python handbook.py eval examples/queries.json -k 3
python -m unittest -v
```

第一行從三份合成手冊建立 11 個段落。第二行回傳的前兩筆是 [checkout.md 的付款重送段落](examples/handbook/checkout.md#L7) 和 [pipeline.md 的 Payment retries 段落](examples/handbook/pipeline.md#L10)。可直接對照 JSON 裡的 `passage` 與原始檔。預設索引檔為目前目錄的 `handbook.db`；要切換索引，在子命令前指定 `--db PATH`。

## 排名與來源

- 段落由空白行或 Markdown 標題切開，繼承最近一個標題，保留從 1 開始的原文起訖行號。支援以井號標記的標題，略過以反引號或波浪號圍起的程式碼區塊。
- 中英文字先經 NFKC 正規化與大小寫折疊。英文依詞切分，連續漢字使用相鄰雙字，單字查詢也有對應索引。混合查詢以 OR 召回，再以 BM25 排名，標題與內文的權重為 3 比 1。同分時依檔名與行號排序。
- 重新建置以一次 SQLite 交易替換索引。檔案移除後重新建置，舊段落會消失。若遇到無法解碼的檔案或空語料，整批回滾。查詢與評估使用唯讀連線。

## 如何解讀搜尋品質

[評估集](examples/queries.json)有 13 個合成示範問題。其中一題標註中英文兩個相關段落。Recall@3 是每題被前三名找回的相關段落比例再取平均；MRR@3 是第一個相關段落名次的倒數再取平均，前三名均未命中時該題為 0。

「背景執行者當機由誰接手」原本只找到資料庫還原演練。開啟詞彙表後，「背景執行者」也會搜尋英文 `worker`，第一筆結果會指出工作租約到期後由另一位執行者接手。[原文與行號](examples/handbook/operations.md#L7)

```sh
python handbook.py query "背景執行者當機由誰接手" --glossary examples/glossary.json
python handbook.py eval examples/queries.json -k 3 --glossary examples/glossary.json
```

| 同一索引與 13 題標註 | 第一名命中 | 平均 Recall@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| 預設搜尋 | 12/13 | 0.9231 | 0.9231 |
| 套用範例詞彙表 | 13/13 | 1.0000 | 1.0000 |

其餘 12 題的前三名結果與順序完全相同。詞彙表是針對已知漏答新增的規則，這組數字衡量的是已知案例修正效果。若要估計新問題的表現，可另外蒐集未參與調整的查詢並標註相關段落。

## 維護團隊詞彙

[詞彙表](examples/glossary.json)的每條規則以 `from` 指定使用者的說法、以 `to` 指定手冊裡的詞組，例如把 `restore drill` 對應到「還原演練」。更新 JSON 後，下次查詢就會讀到新規則，可沿用既有索引。

英文依完整詞比對，漢字可在連續文字內比對。系統保留原本的搜尋詞，另以完整詞組搜尋規則指定的目標，再以同一套 BM25 排名。每條規則只比對原始查詢，需要反向對應時可另寫一條。空白詞條或正規化後重複的來源會回報規則編號。

自訂標註以 `相對路徑#L起始行` 指向段落。編修手冊導致標註行號不存在時，評估會回報是哪一題需要更新。[測試](test_handbook.py)也檢查第二名命中、部分召回與完全漏失時的指標算式。CI 在 Linux 與 Windows 的 Python 3.12、3.13 上分別跑測試與範例。

**履歷用語**　以 SQLite FTS5 與 BM25 實作附原文引用的中英文手冊搜尋，透過可配置的詞彙表修正跨語言漏答，並以逐題排名比對驗證其餘 12 題結果一致。

本專案為 AI 輔助開發的作品集示範。
