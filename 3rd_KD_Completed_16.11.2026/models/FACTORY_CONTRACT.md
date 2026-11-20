# Model Factory 與 Tensor 契約

Smoke runner 和正式 trainer 都透過 factory function 建立模型。Factory 回傳放置於指定 device 的 `torch.nn.Module`。下列 shape 為 batch 維度已加入時的格式。

| 模型 | Factory | 輸入 | 輸出 |
|---|---|---|---|
| Burstormer Teacher | `models.burstormer.burstormer_teacher:create_teacher` | `[1, 8, 6, H, W]`：8 frames，各為 noisy RGB + 3-channel sigma map | `[1, 3, H, W]` |
| NAFNet Student | `models.nafnet.nafnet_student:create_student` | `[1, 3, H, W]`：noisy reference RGB | `[1, 3, H, W]` |

兩個 factory 都接受 `device`，並可選擇性接受 `checkpoint_path`。NAFNet checkpoint 須為 state dictionary，或含 `model_state_dict` 欄位的 dictionary。`train_kd.py` 預設建構正式 NAFNet adapter；`demo_student.py` 僅供 smoke plumbing regression，不是正式 Student architecture。

## Dataset 回傳契約

每筆 Dataset sample 同時含有同一 source/crop 對齊的：

- `teacher_input`：完整 noisy burst 與依此次噪聲參數計算的 sigma maps。
- `reference_rgb`：noisy frame 0，也是 Teacher 使用的 reference frame。
- `synthetic_target`：對應 frame 0 的 clean target。

未經 DataLoader batching 時，shape 分別為 `[8, 6, H, W]`、`[3, H, W]`、`[3, H, W]`；DataLoader 加入 batch dimension 後再送入兩個 factory 建立的 model。

## Training 與 smoke 的差異

目前本地 Burstormer forward 要求 batch size 1；正式 trainer 因此使用 `batch_size=1`。Trainer freeze Teacher，只更新 NAFNet Student，loss 為：

```text
L_total = L_GT + lambda_kd × L_KD
```

Smoke Dataset 使用 placeholder sigma channels 和無關的 toy target，只驗證 tensor shape、model/loss/optimizer plumbing。正式 KD Dataset 從 clean image 即時合成 burst，並依同一組 noise parameters 計算 sigma map。兩種流程跑通都不能單獨證明模型品質或實驗公平。

## 值域與對齊要求

張量 shape 相同仍不足以確保正確蒸餾；Teacher output、Student output 和 clean target 必須代表同一 reference frame、相同 crop 與相容的 channel order/value range。Factory 不代替 Dataset 的 gamma、white-level、normalization 或 evaluation post-processing。此類處理須由明確資料／實驗協議定義並一致套用。

若修改 model signature、tensor layout 或 factory arguments，需同步更新 trainer、smoke runner、dataset checks 和本文件，並重新執行對應 integration test。
