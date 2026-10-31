# 讓偵測器學到事件，而不是辨認異常影片的場景

**結論：**最直接的訓練訊號，是要求模型區分**同一影片內**的事件視窗與事件前視窗。場景去偏方法可作輔助，但現有跨場景結果不能證明它們會改善 UCF-Crime 的事件定位。部署時，分數只能使用目前及過去的視窗。

## 方法表

「可因果」表示可以改成只使用目前及過去視窗；不表示原論文已驗證串流部署。「Head」指儲存特徵上的 linear 或 small head。

| 方法、論文與出處 | 構想；類型 | 因果？所需標籤 | 適用方式；Code | 已報告的相關結果 |
|---|---|---|---|---|
| **同影片 hard negatives**；此處為建議實驗，**不是單一已驗證論文** | 用同一影片的事件前視窗作 negative，對事件視窗施加分類或排序 loss；**資料抽樣＋loss**。影片不再能單靠固定場景解決這組比較。 | **可**；需要可信的 window 標籤，並避免將事件後視窗誤當 normal。 | **兩者**。Head 可直接讀儲存特徵；LoRA 可在現有 Yes/No loss 中加入同影片配對。Code：不適用。 | 對此精確配方的跨場景或 within-video 改善：**not verified**。現有同影片 pairwise ranking 試驗未穩定奏效，不應當作已驗證收益。 |
| **JTT**，[*Just Train Twice: Improving Group Robustness without Training Group Information*](https://arxiv.org/abs/2107.09044)，ICML 2021 | 第一個模型找出難例；第二次訓練提高其權重；**資料重加權**。可特別檢查高分的事件前 negative。 | **可**；window 標籤；訓練不需 group 標籤，調參最好有按場景分組的 validation。 | **兩者**；重訓現有 Head 或 LoRA。Code：[anniesch/jtt](https://github.com/anniesch/jtt)。 | 原論文 **Table 1**：Waterbirds worst-group accuracy，ERM **72.6%**、JTT **86.0%**；這**不是影片定位**結果。[來源](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf) |
| **Group DRO**，[*Distributionally Robust Neural Networks for Group Shifts*](https://arxiv.org/abs/1911.08731)，ICLR 2020 | 優化表現最差的預先定義群組；**loss**。群組宜包含事件狀態與場景，不能只用影片 ID。 | **可**；window 標籤及可信的訓練群組標籤。 | **兩者**；Head 成本較低。Code：[kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)。 | 對 UCF-Crime within-video 或跨場景改善：**not verified**。 |
| **SDN**，[*Why Can’t I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition*](https://arxiv.org/abs/1912.05534)，NeurIPS 2019 | 用 scene-adversarial 等訓練訊號降低背景可預測性；**loss＋architecture**。 | **可因果**，以當前 clip 運作；需要 action/window 標籤及可供去偏的場景訊號。 | **兩者，需改造**：Head 加 scene classifier 與 gradient reversal；LoRA 須有可訓練的相關表示。Code：[vt-vl-lab/SDN](https://github.com/vt-vl-lab/SDN)。 | **Table 1**：Diving48 top-1 **18.0%→20.5%**；屬跨資料集 transfer，**不是**同影片或明確 held-out-scene 測試。[來源](https://arxiv.org/pdf/1912.05534) |
| **BE／FAME／ActorCutMix／StillMix**：[BE](https://arxiv.org/abs/2009.05769)，CVPR 2021；[FAME](https://arxiv.org/abs/2109.15130)，CVPR 2022；[ActorCutMix](https://arxiv.org/abs/2103.16565)，CVIU 2023；[StillMix](https://arxiv.org/abs/2211.12883)，ICCV 2023 | 分別以背景擾動、foreground–background merging、actor mixing、still-frame mixing 減少靜態線索；主要是**資料方法**，FAME 另有 contrastive loss。 | **可因果**，augmentation 僅在訓練時使用；BE／StillMix 需原始影格，FAME／ActorCutMix 另需可靠的 foreground 或 actor mask。 | **主要適用 LoRA**；固定的儲存特徵不能忠實重現影格混合。若 vision encoder 凍結，仍須重新抽取混合 clip 的特徵。Code：[BE](https://github.com/FingerRec/BE)、[FAME](https://github.com/Mark12Ding/FAME)、[ActorCutMix](https://github.com/vt-vl-lab/video-data-aug)、[StillMix](https://github.com/lihaoxin05/StillMix)。 | StillMix 原論文 **Table 2**：Kinetics-400 訓練、ARAS rare-scene 測試；TSM top-1 **57.86%→59.69%**，Swin-T **60.17%→62.49%**。這是跨場景 action recognition，**不是 VAD**。[來源](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Mitigating_and_Evaluating_Static_Bias_of_Action_Representations_in_the_ICCV_2023_paper.pdf) |
| **HSC**，[*Hierarchical Semantic Contrast for Scene-aware Video Anomaly Detection*](https://arxiv.org/abs/2303.13051)，CVPR 2023 | 同時建模 scene 與 motion，判斷事件是否違反該場景的常態；**architecture＋contrastive loss**。它**使用**場景，不是移除場景。 | 原方法的因果性：**not verified**；串流版須限制 motion reference 為過去資料。需要 normal 訓練影片；其增強方法另需 motion 訊號。 | **Head 需較大改造；LoRA 不直接適用**。Code：[shengyangsun/HSC_VAD](https://github.com/shengyangsun/HSC_VAD)。 | 原論文 **Table 5**：ShanghaiTech 兩場景混合、frame AUC；場景 10／12，所列 baseline HF²-VAD **66.8%**、HSC **87.3%**。這**不是未見場景 transfer**。[來源](https://arxiv.org/abs/2303.13051) |
| **DSS＋ASR**，[*Discriminative Score Suppression for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/WACV2025/html/Xu_Discriminative_Score_Suppression_for_Weakly_Supervised_Video_Anomaly_Detection_WACV_2025_paper.html)，WACV 2025；arXiv：**not verified** | 抑制容易辨識的高分片段，並利用 normal-pattern 表示；**loss＋architecture**。原方法不是專門的同影片 negative 方法。 | 原方法的因果性：**not verified**；推論時須限制 reference 為過去資料。原方法使用 video 標籤。 | **Head 可試簡化的 hard-window mining**；LoRA 須另設訓練 loss。Code：**no code found**。 | 原論文 **Table 3**：UCF-Crime abnormal-video-subset **AUCₐ 64.01%→69.38%**；baseline 對 DSS＋ASR。AUCₐ 把異常影片的影格合併計算，**不是 macro per-video AUC**。[來源](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Discriminative_Score_Suppression_for_Weakly_Supervised_Video_Anomaly_Detection_WACV_2025_paper.pdf) |

## 四個問題的判讀

**哪些訊號能迫使模型看事件？**同影片、相反 window 標籤的比較最直接控制了固定的攝影機與場景。JTT 可提高難辨事件前視窗的權重。Group DRO 與 scene-adversarial loss 需要正確的群組或 domain 定義；若場景本身決定何為異常，完全移除場景反而可能有害。影格混合測試的是靜態線索，但可能產生不真實的事件邊界。

**哪些已有對題的數字？**StillMix 提供 rare-scene action-recognition 結果；DSS＋ASR 提供 abnormal-video-subset AUCₐ。兩者都沒有直接證明本任務的 **macro per-video AUC** 上升。另有 [2026 年的 UCF-Crime 評估研究](https://arxiv.org/html/2608.11985v3)，其 **Table 2** 分別報告各模型的 within-video score AUC；例如 CLIP-based VadCLIP **67.64%**、DSANet **69.81%**。這是**不同方法的比較，不是控制變因的去偏效果**。其結果也說明不能把 pooled AUCₐ 當作逐影片定位指標。

**LoRA 與 Head 怎樣用？**優先在儲存特徵上試同影片抽樣、JTT 及 Group DRO；它們不需要重新跑 encoder。LoRA 可沿用同一批 window 標籤與抽樣規則。BE、FAME、ActorCutMix、StillMix 必須回到原始影格；後兩種尤其依賴 mask 品質。任何使用「影片平均特徵」的訓練或測試方案，都必須確認推論沒有讀取未來視窗。

**怎樣測試 shortcut？**Action recognition 使用 ARAS 等罕見場景、Mimetics 的脫離慣常場景動作，以及 background-only／still-frame 輸入。VAD 可用 [UCF-HN](https://arxiv.org/abs/2505.19022) 一類 hard-normal 測試；該論文 **Table V** 報告 false-alarm rate，而非事件定位。對現有 UCF-Crime 資料，最便宜且最貼近部署的測試是：按影片計算 AUC；將每支異常影片的**事件前視窗**與事件視窗比較；另依可取得的 camera／scene 群組報告結果。可加入單一靜態影格重複輸入或 background-only 輸入，但應把它們標為診斷性擾動，不當作真實負例。

## 優先嘗試的三個實驗

1. **同影片 event-versus-before 分類。**先用 Head，對每個 event window 抽同影片的事件前 negative，並保留原本跨影片樣本。比較 **AUCₐ、macro per-video AUC、during-minus-before gap**；三者都須報告，不能只看 pooled AUC。這是新實驗，預期收益 **not verified**。
2. **JTT 重加權。**用現有 Head 找出高分的事件前 negative，再訓練第二個 Head。檢查 **macro per-video AUC** 與 **during-minus-before gap**，並確認一般 normal-video 的誤報沒有增加。
3. **場景分組 Group DRO。**只在能建立可信 scene／camera 群組、且每組有足夠正負 window 時試用 Head。主要看**最差場景的 per-video AUC**；同時報告整體 AUCₐ，避免以犧牲某些場景換取平均改善。