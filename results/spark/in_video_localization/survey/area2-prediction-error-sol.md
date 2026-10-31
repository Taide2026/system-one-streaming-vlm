# 因果式影片事件偵測：prediction-error 分數調查

**結論：先在已儲存的 clip features 上建立「預測下一個 window」基線。** 這能直接測試 prediction error 是否改善同一影片內的事件排序。已發表的 frame AUC 不能回答這個問題；下表方法均未提供可直接對照的 UCF-Crime 同影片內 AUC 或固定攝影機告警延遲。

| 方法與來源 | 構想、因果性與標籤 | 輸入、對現有 features 的適用性與程式碼 | 已報告結果 |
|---|---|---|---|
| **V-JEPA 2**，[*Self-Supervised Video Models Enable Understanding, Prediction and Planning*](https://arxiv.org/abs/2506.09985)，arXiv 2025，2506.09985 | Predictor 從可見 video tokens 預測被遮蔽的 latent tokens；預訓練的空間遮蔽**不等於**因果預測。可改用「只顯示過去、完全遮蔽目標時間」的 mask；不需事件標籤。 | 需要原始 frames 與 encoder 的**未池化 tokens**；現有每 window pooled features 不足。官方 [程式碼與 checkpoints](https://github.com/facebookresearch/vjepa2) 包含 predictor，並非只釋出 encoder。 | **UCF-Crime 同影片內結果：未驗證。** 原論文未報告適用於此部署的分數。 |
| **V-JEPA surprise**，[*Intuitive physics understanding emerges from self-supervised pretraining on natural videos*](https://arxiv.org/abs/2502.11831)，arXiv 2025，2502.11831 | 只以先前 frames 預測後續 latent tokens，再計算預測與觀測 token 的平均絕對誤差；可因果部署，不需事件標籤。 | 需要 frames、target encoder 與 predictor，不能直接使用 pooled features。[評估程式碼](https://github.com/facebookresearch/jepa-intuitive-physics)。這項實驗使用 **V-JEPA，不是 V-JEPA 2**。 | IntPhys／GRASP／InfLevel-lab 的**成對影片選擇準確率**為 98%／66%／62%（[Garrido 等，Figure 1A 與結果段落](https://arxiv.org/html/2502.11831)）；不是 VAD AUC。 |
| **FPDM**，[*Feature Prediction Diffusion Model for Video Anomaly Detection*](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf)，ICCV 2023；arXiv ID：未找到 | 以先前 frame features **及目前 frame 的加噪 feature**做 diffusion 預測與 refinement，再比較 feature。只需正常訓練資料；評估可在目前 frame 到達後執行，但不是純過去輸入的預測。 | 原方法使用 frame feature maps；套用現有 pooled clip features 必須重新設計並訓練 predictor。公開可用的官方程式碼：未驗證。 | UCF-Crime 74.7%、ShanghaiTech 78.6%、Avenue 90.1%，均為 frame AUC（[Yan 等，Table 1](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf)）。未報告同影片內 AUC。 |
| **Future Frame Prediction**，[*Future Frame Prediction for Anomaly Detection – A New Baseline*](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf)，CVPR 2018，1712.09867 | 從過去 frames 預測下一張 RGB frame，依預測品質評分；因果；只需正常訓練影片。 | 必須重新讀取 frames 並訓練 pixel predictor；不能直接使用儲存的 features。[官方程式碼](https://github.com/StevenLiuWen/ano_pred_cvpr2018)。 | Avenue 85.1%、ShanghaiTech 72.8% frame AUC（[Liu 等，Table 1](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf)）。**其 PSNR 以整部測試影片的最小值與最大值正規化，會使用未來 frames；原報告流程不能直接部署。** |
| **RGB error-guided predictive coding**，[*Noise-Resistant Video Anomaly Detection via RGB Error-Guided Multiscale Predictive Coding and Dynamic Memory*](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Noise-Resistant_Video_Anomaly_Detection_via_RGB_Error-Guided_Multiscale_Predictive_Coding_CVPR_2025_paper.pdf)，CVPR 2025；arXiv ID：未找到 | 結合 RGB 預測誤差、multiscale predictive coding 與 memory；只需正常訓練資料。過去輸入的部署可行性需逐項核查其實作。 | 需要 frames 或 object crops，不能直接使用 pooled features。程式碼：未找到。 | UCF-Crime frame／object 輸入分別為 79.9%／82.1% frame AUC（[Hu 等，Table 4](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Noise-Resistant_Video_Anomaly_Detection_via_RGB_Error-Guided_Multiscale_Predictive_Coding_CVPR_2025_paper.pdf)）；未報告同影片內 AUC。 |

## 關鍵判讀

- **V-JEPA 2 的 2 s 測試需要重新定義 mask。** 對目標 window \(t\)，只提供 \(t\) 以前的 frames；在 \(t\) 到達後，才編碼其實際 tokens 並計算誤差。原始預訓練採 multiblock masking，而非固定的 future-only mask（[V-JEPA 2 方法](https://arxiv.org/html/2506.09985)）。[V-JEPA physics 評估設定](https://github.com/facebookresearch/jepa-intuitive-physics/blob/main/evaluation_code/evals/intuitive_physics/configs/default_intphys.yaml)使用滑動 clip 與多種 context 長度；它不是現有 2 s window 的現成評估。原始 frames、取樣率與 context 長度都必須重新核對。

- **Prediction error 不會按設計消除 scene bias。** 靜態背景可能容易預測，但模型仍可能利用攝影機、背景或畫質；可預測的異常也可能得到低誤差。FPDM 的 ShanghaiTech 分場景結果差異明顯（[Yan 等，Table 4](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Feature_Prediction_Diffusion_Model_for_Video_Anomaly_Detection_ICCV_2023_paper.pdf)），不能據此宣稱已消除場景效應。能證明符合部署目標的證據，應是**同影片內 AUC、事件開始後的告警延遲，以及逐攝影機固定閾值的誤報**；上述方法未提供完整對應結果。

- **失敗情況**包括 camera motion、擁擠且多樣的正常活動、cuts，以及正常但難預測的事件。相反地，持續且規律的異常可能很快變得可預測。請分別檢查事件開始與事件持續期間，不要只報整段異常的平均分數。

## 優先試驗

1. **現有 features 的因果下一個-window predictor。** 以正常 windows 訓練小型 autoregressive predictor；在 window \(t\) 到達後計算 \(\lVert z_t-\hat z_t\rVert\)。先用簡單線性或 ridge predictor 作基線，再比較較長 context。這是**建議的新實驗，不是上表論文的已驗證結果**。
2. **V-JEPA 2 token-level surprise。** 用已釋出的 predictor 測試 past-only／future-only mask。這最接近 physics surprise，但需要重新讀取 frames，且不能用現有 pooled features 代算。
3. **目前 feature 條件化的 denoising error。** 借鑑 FPDM，將當前 \(z_t\) 的加噪版本與過去 features 一起輸入小模型；與第一項的純預測誤差分開報告，避免把「看到當前輸入後的修復」誤稱為未來預測。

三項試驗都應優先報告異常影片的 per-video AUC 分布、事件開始延遲及固定閾值誤報；不要只依 pooled frame AUC 排名。