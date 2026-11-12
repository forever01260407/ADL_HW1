# 第一次把 ADL 專案 push 到 GitHub

適用於目前的 `C:\Users\yuting\Desktop\HW1_ADL`。整理完成時尚未執行 `git init`、commit 或 push；電腦已有 Git，不需安裝 GitHub CLI。

## 先了解四個動作

| 指令 | 意義 |
| --- | --- |
| `git init` | 讓本機資料夾開始由 Git 管理 |
| `git add` | 選定這次要存的檔案，放進暫存區 |
| `git commit` | 在本機儲存一個版本 |
| `git push` | 將已儲存的 commit 上傳到 GitHub |

`commit` 成功只代表本機有版本；執行 `push` 後 GitHub 才會收到。

## 1. 在 GitHub 建立空 repository

登入 [GitHub 建立頁](https://github.com/new)，名稱可以填 `ADL-HW1-User-State-Prediction`，Visibility 先選 **Private**。不要勾選新增 README、`.gitignore` 或 License，因為本機已有準備好的檔案。按下 **Create repository**。

課程講義第 11 頁禁止截止前公開程式，以及與他人交換模型／預測；第 2 頁記載截止時間為 2026/10/4 23:59。這裡先用私人庫備份；若日後公開，應另行確認分享範圍，包含 `download.sh` 中的模型連結。

## 2. 開啟 PowerShell，初始化本機 repository

```powershell
cd "C:\Users\yuting\Desktop\HW1_ADL"
git init -b main
```

只在第一次執行；之後更新不用重做。

## 3. 設定這個 repository 的 commit 身分

將名字和信箱替換成你的資料，再執行。`--local` 只影響這個專案。

```powershell
git config --local user.name "你的名字或 GitHub 名稱"
git config --local user.email "你的 GitHub 信箱"
```

若不想讓私人信箱出現在 commit，可從 [GitHub Email 設定](https://github.com/settings/emails) 複製你帳號的 `noreply` 信箱。

## 4. 選取檔案並檢查

```powershell
git add .gitignore .gitattributes README.md docs r15922a14
git status --short
git diff --cached --stat
```

應看到文件和 `r15922a14/` 的程式、報告、彙總數據。`.venv/`、原始實驗工作目錄、資料集、權重及下載壓縮檔應被排除。不要使用 `git add -f`，它會跳過排除規則。

GitHub 一般 Git repository 阻擋超過 100 MiB 的單檔；模型檔沿用既有下載方式，不納入 Git。見 [官方檔案限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

確認內容後建立第一個版本：

```powershell
git commit -m "Organize ADL HW1 source, report, and experiment evidence"
```

## 5. 設定 GitHub 位置並上傳

從剛建立的 repository 頁面複製 HTTPS URL，將 `YOUR_USERNAME` 和 `YOUR_REPOSITORY` 替換成實際值：

```powershell
git remote add origin "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
git remote -v
git push -u origin main
```

若開啟登入視窗，依畫面使用自己的 GitHub 帳號登入。HTTPS Git 操作不使用帳號密碼驗證；若沒有自動登入流程，依 [GitHub HTTPS 驗證說明](https://docs.github.com/en/get-started/git-basics/about-remote-repositories#cloning-with-https-urls) 設定憑證或 token，不要將 token 放進 remote URL 或專案檔案。

完成後重新整理 repository 頁面，應能看到專案 README 和 `r15922a14/`。此流程依 [GitHub 官方上傳本機專案說明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 之後每次修改

```powershell
cd "C:\Users\yuting\Desktop\HW1_ADL"
git status
git add .gitignore .gitattributes README.md docs r15922a14
git diff --cached --stat
git commit -m "Describe what changed"
git push
```

新增其他根目錄程式或資料夾時，先更新 `.gitignore` 的納入清單，再將新路徑加入 `git add`。

## 常見訊息

| 訊息 | 處理方式 |
| --- | --- |
| `Author identity unknown` | 完成第 3 步，再 commit |
| `remote origin already exists` | 先 `git remote -v`；若 URL 確實不對，用 `git remote set-url origin "正確的 HTTPS URL"` |
| `Repository not found`／403 | 核對 URL、登入帳號和私人庫權限 |
| `src refspec main does not match any` | 檢查是否已成功 commit，並用 `git branch --show-current` 確認分支 |
| `non-fast-forward`／`fetch first` | 遠端已有 commit；先檢查遠端內容及 `git log`，再合併，不要直接 force push |

排除規則詳見 [GitHub 官方 `.gitignore` 說明](https://docs.github.com/en/get-started/git-basics/ignoring-files)。
