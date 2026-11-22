# 結帳服務

## 庫存保留到期
購物車保留到期後，庫存回收會歸還可用數量。相同訂單重送只會讀回原保留。

## 付款重送
付款請求因逾時重送時，沿用原請求鍵。若建立新鍵，系統會視為另一筆付款。

## Settlement reconciliation
Reconcile the payment settlement file against the order ledger by currency. Track partial refunds as separate postings.
