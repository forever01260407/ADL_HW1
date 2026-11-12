# ADL 專案地圖

## 哪一份是主版本？

`r15922a14/` 是完整繳交版，也是這次 GitHub 整理採用的主要程式入口。整理時逐一比對，它的 10 個頂層 Python 檔案與 `ADL_HW1_V2/` 對應檔案在統一換行後完全相同。繳交版另有重現說明、報告與數值證據。

`ADL_HW1_V2/` 是原本的訓練工作區。最終模型來自 `outputs/v02_ctx128_weighted/best_model`，由保存的設定與指標確認。

## 演進與實驗

| 版本／實驗 | 內容 | GitHub 保留位置 |
| --- | --- | --- |
| XLM-R baseline | 初期訓練、增加步數、加權 loss | `r15922a14/experiments/baseline/` |
| V2 | 輸入順序、歷史預算、完整輪次、角色標記、分開 split/model seed、調整選模與校準 | `r15922a14/` |
| V2 消融 | 無歷史、128／256／512 tokens、加權／不加權、不同 seeds、舊式打包 | `r15922a14/report_sources/evidence/` 的各實驗數據 |
| mBERT | 使用另一個多語言 encoder 比較 | `r15922a14/experiments/mbert/` |

## 程式關係

1. `common.py` 定義標籤、資料驗證、測試檔合併、前處理、Dataset、collator、切分與評分工具。
2. `train.py` 使用這些工具訓練，儲存 checkpoint、前處理設定、分類門檻與指標。
3. `inference.py` 讀取 tokenizer、權重、前處理設定及各類門檻，輸出並檢查 CSV。
4. `download.sh` 下載固定模型壓縮檔；`run.sh` 需要解壓縮時會先檢查 SHA-256，再啟用離線推論。
5. `analyze_validation.py` 分析相同的 validation IDs；`compare_runs.py` 比較多次實驗。
6. `report_sources/` 保存報告製作程式、原報告與彙總證據；`verification/` 是既有驗證紀錄與選用 helper。

## 本機空間盤點

整理時的檔案邏輯容量如下，單位 GiB；不是 GitHub 上傳容量，也不等同磁碟實際占用。

| 資料夾 | 約 GiB | 主要內容 |
| --- | ---: | --- |
| `ADL_HW1_baseline/` | 4.21 | 訓練權重與實驗結果 |
| `ADL_HW1_V2/` | 14.15 | 消融權重、模型封裝與測試產出 |
| `ADL_mbert/` | 2.01 | mBERT 權重與 Hugging Face 快取 |
| `r15922a14/` | 0.03 | 87 個繳交檔案，約 30 MB，含報告 |
| `submission_validation/` | 3.60 | 解壓模型、乾淨環境驗證與預測輸出 |
| `.venv/` | 6.89 | 本機 Python 環境 |
| `.venv-ta/` | 4.46 | TA 相容環境 |
| `tmp/` | 0.04 | 報告渲染圖片等暫存 |

這次以 Git 納入範圍和文件整理主要入口，保留原始檔案、權重與路徑，沒有刪除／搬移舊實驗，也沒有重新訓練。

## 本次整理驗證

- 在獨立暫存 Git repository 模擬加入檔案，選到 93 個檔案，合計約 31.0 MB；最大單檔約 15.3 MB。
- 13 個排除檢查全部通過，涵蓋資料集、模型封裝、虛擬環境、輸出、快取與個別樣本內容。
- 確認 shell 的 Git 換行屬性為 LF，並檢查新增文件的本機連結。
- 原始 ZIP 的 SHA-256 與既有 audit 相同；其中 87 個檔案與本機繳交版逐一比對完全一致。
- 模擬用的 Git 暫存目錄已移除，專案根目錄尚未初始化 Git，也未 commit／push。這次未重新執行模型訓練或推論。

## 後續修改與繳交

- 維護 GitHub 版本時，修改 `r15922a14/`；新實驗使用新的 `--output-dir`。
- 若在 `ADL_HW1_V2/` 修改程式，GitHub 不會追蹤那些變更，需將確認過的變更同步到主版本。
- `submission_validation/prepare_submission.py` 會從舊工作目錄複製檔案到 `r15922a14/`，可能覆蓋主版本後來的修改，執行前先檢查來源。
- 原始 `r15922a14.zip` 保留原狀，修改程式不會自動更新 ZIP。
- `.gitignore` 只控制未追蹤檔案是否納入 Git，不會刪除本機檔案，也不會自動移除已經 commit 的檔案。
