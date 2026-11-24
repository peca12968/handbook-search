# 知識檢索與工單派送

核對手冊檢索的原文依據，並量測工單分類、置信門檻及模型部署。[工具入口與各模組的重跑方式](modules/README.md)。

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

同時保留舊版與現行手冊時，可以用 `--within` 指定索引內的相對檔名或目錄。例如 `python handbook.py query "付款重送 payment retry" -k 1 --within pipeline.md` 只在該手冊選取段落，命中第 10 行。目錄按完整路徑元件比對，`current` 不會包含 `current-old`。候選在 SQL 查詢內先符合路徑，再排序取前 k 筆。BM25 的文件頻率仍來自整座索引。

評估資料可在每題加上 `"within": "current"`。若標註答案位於範圍外，評估會指出該題。跨專案演練在同一批 26 個段落上比較全文與向量檢索。兩題使用相同的現行版答案與原文行號，以詞頻向量取得的前四個全域候選都是舊版；限制範圍的兩種檢索都找到正確段落。CI 使用固定版本的 Vector Sieve 與 Faiss 跑這項比較。一般索引與全文搜尋僅用 Python 標準函式庫。[交叉驗證](scope_demo.py) · [Vector Sieve](https://github.com/peca12968/handbook-search/tree/main/modules/vector-sieve)

- 段落由空白行或 Markdown 標題切開，繼承最近一個標題，保留從 1 開始的原文起訖行號。支援以井號標記的標題，略過以反引號或波浪號圍起的程式碼區塊。
- 中英文字先經 NFKC 正規化與大小寫折疊。英文依詞切分，連續漢字使用相鄰雙字，單字查詢也有對應索引。混合查詢以 OR 召回，再以 BM25 排名，標題與內文的權重為 3 比 1。同分時依檔名與行號排序。
- 重新建置以一次 SQLite 交易替換索引。檔案移除後重新建置，舊段落會消失。若遇到無法解碼的檔案或空語料，整批回滾。查詢與評估使用唯讀連線。

## 引用前核對目前的 Markdown

```sh
python handbook.py query "付款重送" -k 1 --within checkout.md --verify-sources examples/handbook
```

`--verify-sources` 在搜尋結果附加 `source_check`，核對現行檔案的段落標題、內文與起訖行。原行號仍吻合會標記 `current`。相同段落在同一檔案中移到唯一新位置會標記 `moved`，並附上新的行號。原位置失效且有多個相同候選時會標記 `ambiguous`，列出最多 50 個候選與候選總數。

找不到相同段落時會標記 `changed`，找不到原檔會標記 `missing`。無法安全讀取或解碼的檔案會標記 `unavailable`。核對只讀取命中的檔案，每個檔案讀取一次。原索引與排名可以留作稽核，更新索引可另執行 `sync`。[作品站與端對端測試共用的範例](examples/citations-demo.json)

### 只取已驗證的引用

```sh
python handbook.py query "付款重送" -k 2 --verify-sources examples/handbook --verified-only --scan-limit 100
python verified_demo.py
```

`--verified-only` 保留原本的 BM25 與 `--within` 範圍，依排名往後補足最多 `k` 筆已核對的引用。每筆保留原索引 `id` 與 `rank`，對外引用可使用 `citation` 的現行行號。不同索引段落若對應同一個現行位置，只列一次。掃描上限預設 100，可設為 `k` 至 500；`scanned` 與 `skipped` 記錄實際掃描及各種略過原因。

範例在建立索引後改寫第 1 名原文，並把第 2 名的段落從第 2 行移到第 6 行。上限為 2 時只返回 1 筆，`stopped` 為 `scan_limit`。掃描到第 3 名可補足 2 筆，`stopped` 為 `filled`。候選用完仍未補足時回報 `exhausted`，沒有可核對的引用時可回傳空清單。[重跑程式](verified_demo.py) · [實際命令輸出](examples/verified-search-demo.json)

## 工單模型上線前的比較關卡

在同一批獨立評估工單上逐筆比較現行與候選模型，檢查精確率、新增誤派、總覆核量以及各隊列的覆核量。報告列出改變的工單 ID、評估期間與三份輸入的 SHA-256，供 CI 和發布紀錄引用。[重跑步驟](modules/ticket-triage/README.md#在-ci-檢查工單模型是否可替換) · [端對端測試](modules/ticket-triage/test_release_gate.py)

## 手冊改版後同步索引

對同一份資料庫執行 `python handbook.py sync examples/handbook`，會重新讀取 Markdown 的原始位元組並比對 SHA-256。路徑與內容均相同的文件沿用既有段落及行號；新增或修改的文件才重新切段、斷詞並寫入 FTS5。移除文件時會刪除對應段落。檔案改名會計為一筆移除與一筆新增，查詢結果因此會指到新檔名。掃描仍會讀取所有文件，內容雜湊可辨識大小和修改時間都相同的改動。

執行 `python sync_demo.py` 可在臨時目錄重現一次改版，直接看到每一輪的統計。

| 操作 | 新增 | 修改 | 移除 | 沿用 |
| --- | ---: | ---: | ---: | ---: |
| 首次載入四份手冊 | 4 | 0 | 0 | 0 |
| 新增核准段落、將維運手冊改名、移除退役流程 | 1 | 1 | 2 | 1 |
| 原樣重跑 | 0 | 0 | 0 | 3 |

改版後，`supervisorapproval` 指向 `checkout.md#L14`，退役詞 `legacyprocedure` 的命中數為零。第一次對舊格式的索引執行 `sync` 時，會在同一筆交易內重建一次並建立文件清單。後續同步只更新有差異的段落。原有的 `index` 命令仍可強制完整重建。[可執行示範](sync_demo.py)

同步會在取得 SQLite 寫入鎖後才比對清單，並將段落和檔案雜湊一起提交。無效 UTF-8、目錄掃描失敗或整份手冊已無可搜尋段落時，上一個成功版本會保留。這筆交易保護資料庫；來源目錄若仍在編修，下一次同步會重新讀取各檔。測試以兩個獨立程序競爭同一索引，另把連續 32 次修訂的搜尋結果、排名與行號逐次對照完整重建。[同步測試](test_sync.py)

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

**履歷用語**　以 SQLite FTS5 與 BM25 實作可追溯原文的中英文搜尋，透過內容雜湊與單筆交易同步改版及刪頁，並以 32 次連續修訂對照完整重建結果。

本專案為 AI 輔助開發的作品集示範。

[同庫模組與整合測試](modules/README.md)
