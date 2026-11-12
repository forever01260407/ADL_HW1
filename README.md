# ADL HW1 — User State Prediction

中英文對話的多標籤意圖分類專案。輸入「使用者目前的訊息＋對話歷史」，輸出 10 個標籤各自的 0/1 預測；一句話可以同時屬於多個類別。

**主要程式在 [r15922a14/](r15922a14/)**。完整重現步驟見 [繳交版 README](r15922a14/README.md)，首次上傳請看 [GitHub 中文教學](docs/GITHUB_PUSH.md)。

## 專案流程

```text
train.jsonl → 訓練／驗證切分 → XLM-R + 10-label classifier
                             ↓ 選擇 checkpoint、校準門檻
context.json + test.json → 依 id 合併 → 前處理 → 模型 → prediction.csv
```

- 資料：3,000 筆訓練資料，切成 2,400 筆訓練／600 筆驗證；本機 public test 為 500 筆。
- 最終設定：`FacebookAI/xlm-roberta-base`、加權 BCE loss、model seed 42。
- 輸入：目前訊息優先，歷史最多 128 tokens，保留近期完整輪次與角色標記；總長最多 512 tokens。
- 訓練：最多 12 epochs，以固定門檻的 macro/micro-F1 調和平均選 checkpoint；最終使用 epoch 9，再校準每類門檻。
- 推論：對每個標籤獨立做 sigmoid 並比較門檻，保持 `test.json` 順序輸出 CSV；下載完成後離線執行。

標籤為 `Ask_Price`、`Ask_Service`、`Ask_Spec`、`Compare_Competitor`、`Confirm_Order`、`Doubt`、`Need_More_Evidence`、`Need_Time_To_Think`、`State_Need`、`Too_Expensive`。

## 已記錄的驗證結果

| 指標 | 最終 checkpoint 校準後 |
| --- | ---: |
| Macro-F1 | 0.835055 |
| Micro-F1 | 0.843026 |
| 兩者調和平均 | 0.839021 |

來源：[保存的驗證指標](r15922a14/report_sources/evidence/v02_ctx128_weighted/validation_metrics.json)。這是 600 筆 validation 的結果；選模與校準共用此 split，不能當作獨立 test 成績。

## 資料夾導覽

| 路徑 | 用途 | 預設納入 Git |
| --- | --- | --- |
| `r15922a14/` | 最終程式、報告、歷史實驗來源與彙總數據 | 是，排除產生的模型／資料 |
| `docs/` | 專案地圖、原始資料說明、GitHub 教學 | 是 |
| `ADL_HW1_V2/` | V2 原始實驗、消融結果與權重 | 否 |
| `ADL_HW1_baseline/` | 初期 XLM-R 實驗與權重 | 否 |
| `ADL_mbert/` | mBERT 對照實驗、快取與權重 | 否 |
| `submission_validation/` | 本機打包、下載、相容性與效能驗證工作區 | 否 |
| 根目錄 JSON／JSONL／CSV | 課程資料、答案與輸出範例 | 否 |
| `r15922a14.zip` | 原有 COOL 繳交壓縮檔 | 否 |
| `hw1.py` | 早期預訓練模型載入與 GPU 測試 | 否 |
| `.venv/`、`.venv-ta/`、`tmp/` | 本機環境與暫存 | 否 |

原有實驗和資料維持路徑，讓歷史指令與打包腳本仍能找到它們。GitHub 以繳交版為入口，baseline 和 mBERT 的歷史程式已在它的 `experiments/`。詳細用途與容量見 [專案地圖](docs/PROJECT_MAP.md)。原根目錄 README 已保存在 [原始資料說明](docs/course_data_readme.md)。

## 從哪個檔案開始讀

| 要做的事 | 入口（位於 r15922a14/） |
| --- | --- |
| 資料檢查、前處理、切分、F1 與門檻邏輯 | [common.py](r15922a14/common.py) |
| 訓練與 checkpoint 選擇 | [train.py](r15922a14/train.py) |
| 讀取已儲存的模型、前處理設定與門檻做預測 | [inference.py](r15922a14/inference.py) |
| 下載及評測介面 | [download.sh](r15922a14/download.sh)、[run.sh](r15922a14/run.sh) |
| 分析錯誤／每類別表現 | [analyze_validation.py](r15922a14/analyze_validation.py) |
| 比較消融實驗 | [compare_runs.py](r15922a14/compare_runs.py) |
| 看報告與數據 | [report.pdf](r15922a14/report.pdf)、[report_sources/](r15922a14/report_sources/) |

## 執行方式

使用 Python 3.10 和 [繳交版 README 的套件版本](r15922a14/README.md#2-environment)。以下在 Ubuntu／WSL 的 Bash 執行：

```bash
cd r15922a14
bash download.sh
bash run.sh ../context.json ../test.json ../prediction.csv
python3 validate_output.py --test-file ../test.json --prediction-file ../prediction.csv
```

上述資料路徑適用於本機目前的工作區。GitHub clone 不包含課程資料，須自行取得官方資料，放到對應位置。訓練請使用繳交版 README 第 5 節的完整指令；`train.py` 的預設參數不等於最終實驗設定。

## GitHub 版控範圍

`.gitignore` 採根目錄明確納入清單，只納入主程式、文件及彙總實驗證據；新增根目錄程式或資料夾時須更新清單。原始資料、個別樣本錯誤內容、權重、快取與虛擬環境會被排除。`.gitattributes` 固定 shell 等文字檔為 LF 換行。

這次只準備檔案；尚未初始化本機 Git repository、建立 commit 或上傳 GitHub。
