# 維運手冊

## 錯誤預算
服務可用率以成功請求除以全部請求。錯誤預算耗盡時，值班人員應檢查失敗流量與缺資料的日期。

## Worker lease recovery
A worker may crash before acknowledging a job. When its lease expires another worker takes ownership and rejects stale completion.

## 資料庫還原演練
從備份還原後，先核對最近的訂單筆數，再執行讀取測試，將驗證結果寫進演練紀錄。

## Release rollback
When a new release breaks checkout, roll back the deployment to the last verified build and check the error budget.
