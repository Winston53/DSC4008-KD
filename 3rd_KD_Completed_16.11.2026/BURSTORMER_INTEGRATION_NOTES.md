# Burstormer Teacher 整合與交接紀錄

## 1. 目前整合狀態

- `models/burstormer/Network.py`：專案使用的 Burstormer model definition。
- `models/burstormer/burstormer_teacher.py`：載入 pretrained checkpoint、凍結參數、維持 evaluation mode，並提供 `create_teacher` factory。
- `weights/Burstormer.ckpt`：目前專案使用的 Teacher 權重，已在 CPU 上成功載入。
- `train_kd.py` 與 smoke runner 均透過 adapter 使用 Teacher。
- 上游 `Color_denoising_testing.py`、testing dataset、metrics 和 post-processing 沒有接入 KD training path。

本地 Teacher forward 要求輸入 `[B, 8, 6, H, W]`，目前明確限制 `B=1`。8 個 frame 各含 3-channel noisy RGB 與 3-channel per-pixel sigma estimate；輸出為 `[1, 3, H, W]`。

## 2. 已執行的驗證

1. Synthetic smoke test：使用實際 checkpoint、placeholder input 和 Student，完成一次 CPU optimizer update。
2. Real-image integration：使用 Open Images 下載圖片生成一筆 KD sample，搭配 NAFNet 與實際 Teacher 完成 CPU KD update。
3. 小型訓練：9 張可用圖片，以 7/2 train/validation split 完成 1 個 CPU epoch，並儲存 `checkpoints_kd/nafnet_kd_best.pt`。

Lightning 載入 legacy checkpoint 時出現 checkpoint migration 與 callback state warning；但上述載入、forward、update 和 epoch 均成功。這些 run 沒有改寫原始 Teacher checkpoint。

以上證據只表示目前的 runtime integration 可執行，不代表 Burstormer restoration quality、論文 benchmark parity，或目前合成 noise 分佈與 Teacher pretraining 完全相容。

## 3. KD Dataset 與輸入契約

`data.nafnet_kd_dataset.NAFNetKDBurstDataset` 從一張 clean source image 產生一筆 8-frame burst，合成異方差 noise，並用該次採樣的 noise parameters 計算 sigma map。Dataset 同時回傳：

| 欄位 | 用途 | 未加 batch 維度的 shape |
|---|---|---|
| `teacher_input` | noisy burst 與 sigma map 串接，送入 Burstormer | `[8, 6, 128, 128]` |
| `reference_rgb` | noisy reference frame 0，送入 NAFNet | `[3, 128, 128]` |
| `synthetic_target` | 對應 frame 0 的 clean target | `[3, 128, 128]` |
| `sigma_map` | 每幀、每個 RGB channel 的 noise estimate | `[8, 3, 128, 128]` |

加入 DataLoader batch 維度後，Teacher input 為 `[1, 8, 6, 128, 128]`。Noisy burst、sigma map 和 target 在 `__getitem__` 即時產生；目前不會把處理後的 burst 寫進 `data/processed/`。

`data/burst_dataset.py` 是另一個 `.npz` loader，沒有接入 `train_kd.py`。使用前須核對 `.npz` producer 的 field names、shape、參數意義、frame order 和數值尺度。

## 4. Noise estimate 與待確認的 domain

目前 sigma map 依此次合成 noise 所用的 `sigma_read`、`sigma_shot` 參數計算，故它和本專案生成器的參數在數學上相關。這不等同已證明 noise level、參數單位、gamma/linearization、white-level 或輸入範圍符合 Burstormer pretrained data。

論文提及的四組 noise settings 是該實驗的 testing variants；不要未核實就視為目前 training generator 唯一可用的參數。Smoke Dataset 使用固定 placeholder sigma，只供驗證張量和訓練 plumbing，不代表正式 sigma estimate。

正式宣稱 benchmark parity 或公平比較前，請與組員核對：

1. Read/shot parameter 的定義、單位和採樣範圍。
2. `white_level`、gamma／linearization 和 model input/output 的 numeric range。
3. Teacher output 是否需 clamp、inverse transform 或其他官方後處理。
4. NAFNet baseline 和 KD 的 source split、crop、noise realization 是否逐筆一致。
5. 20K 的計數單位、train/validation/test 分配和固定測試 noise settings。

## 5. Factory 與 checkpoint 使用

正式 trainer 使用 `models.burstormer.burstormer_teacher:create_teacher`，並以 `device`、checkpoint path 建立 Teacher。未指定 checkpoint 時，adapter 依自身位置尋找專案根目錄下的 `weights/Burstormer.ckpt`。Teacher 的參數會 freeze，forward 使用 no-grad；optimizer 只更新 Student。

Lightning 曾印出自動 migration 與 callback-state key collision warning。現有 KD path 只載入 Teacher 做 inference，不還原 Teacher optimizer 或 callback state，因此該 warning 沒有阻止已執行的流程。不要因 warning 未備份就執行永久改寫 checkpoint 的 upgrade command；若日後需升級，先複製備份並重新驗證。

## 6. 交接清單

- 保留 `Network.py`、`burstormer_teacher.py`、`__init__.py`。
- 保留 `weights/Burstormer.ckpt`，或明確確認收件者能取得相同權重。
- 不需要複製整個上游 benchmark dataset 或 `Color_denoising_testing.py` 才能啟動本專案 KD trainer。
- 收件者需安裝相容的 PyTorch、torchvision、pytorch-lightning、einops 等依賴；GPU 機器須另外選擇適用的 CUDA-enabled PyTorch build。
- 建議先下載少量 clean images、執行 Dataset verifier，再用 `--epochs 1` 驗證整條訓練流程。

Teacher 能載入不代表它已在本專案合成噪聲上通過品質驗證；正式資料 domain 和 test-score protocol 仍需核查。
