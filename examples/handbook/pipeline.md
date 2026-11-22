# Data pipeline

## 資料重送與去重
匯入批次時以事件編號去重。重送同一批次後，資料列數應維持一致。

## Missing observations
Report missing dates separately from days with zero requests. Availability uses request counts across the observed period.

## Payment retries
Reuse the same request key when a payment times out. A retry with another key represents a new payment.

## Pipeline checkpoints
Save the source offset only after the output batch commits. Restart from that checkpoint to resume interrupted ingestion.
