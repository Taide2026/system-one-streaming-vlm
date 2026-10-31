# Video anomaly detection 的「異常影片限定」定位指標

**結論：有論文測量此量，但名稱和評估單位必須分清。** 下表的 \(AUC_A\) 是將**所有異常測試影片的影格合併**後，計算異常影格對正常影格的 ROC AUC。它排除全正常影片，但仍可能獎勵影片之間的分數差異；它**不是**逐影片計算後取平均的 AUC。表內數值均為百分比。

| 排名與訓練條件 | 論文、來源表格 | UCF-Crime \(AUC_A\) | 同模型整體 AUC |
|---|---|---:|---:|
| 另列：額外 *glance* 影格標註 | [*GlanceVAD: Exploring Glance Supervision for Label-efficient Video Anomaly Detection*，ICME 2025／arXiv:2403.06154，表 1](https://arxiv.org/html/2403.06154) | **84.94** | 91.96 |
| **1．僅影片級標籤** | [*Mixture of Experts Guided by Gaussian Splatters Matters*（GS-MoE），ICCV 2025／arXiv:2508.06318，表 1](https://arxiv.org/html/2508.06318) | **83.86** | 91.58 |
| 2．僅影片級標籤 | [*Just Dance with π! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection*（π-VAD），CVPR 2025，表 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf) | 77.77 | 90.33 |
| 3．使用 UCA 文字描述 | [*Learning Suspected Anomalies from Event Prompts for Video Anomaly Detection*（LAP-UCA），ACM TOMM／arXiv:2403.01169，刊出版表 1](https://doi.org/10.1145/3801554) | 76.07 | 90.45 |
| 4．使用 SwinBERT 描述 | [同一 LAP 論文，arXiv 版表 I](https://arxiv.org/html/2403.01169) | 73.0 | 88.9 |
| 5．僅影片級標籤 | [*Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection*（UMIL），CVPR 2023／arXiv:2303.12369，UCF-Crime 結果表](https://arxiv.org/html/2303.12369) | 68.68 | 86.75 |

表中**最佳已核對的影片級弱監督結果是 GS-MoE 的 83.86**；這不是完整領域的 SOTA 證明。GlanceVAD 的 84.94 使用額外影格標註，不能列入相同監督條件的排名。對**完全 tuning-free／zero-shot VLM**，本次未核對到論文同時報告 UCF-Crime \(AUC_A\) 與整體 AUC；其 \(AUC_A\) 最佳值為 **not verified**。GlanceVAD 作者的[程式庫結果表](https://github.com/pipixin321/GlanceVAD)另列 82.18／91.76；它與論文表 1 的 84.94／91.96 不同，不能混用。

| 其他資料集或定位評估 | 已報告的結果與適用範圍 |
|---|---|
| XD-Violence | [π-VAD，CVPR 2025，表 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf) 報告異常影片限定 **\(AP_A=85.79\)**，整體 **AP＝85.37**。此資料集使用 AP；不要與 UCF-Crime 的 ROC AUC 比大小。 |
| XD-Violence | [GlanceVAD，表 2](https://arxiv.org/html/2403.06154) 報告 **\(AP_A=89.85\)**、整體 **AP＝89.40**；同樣使用額外 glance 標註。 |
| UCF-Crime 的事件區間 | [*Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning*，CVPR 2026，表 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Fine-VAD_Towards_Fine-Grained_Video_Anomaly_Detection_via_Progressive_Cross-Granularity_Learning_CVPR_2026_paper.pdf) 報告 temporal IoU 門檻 0.1／0.3／0.5 的 mAP 為 **21.43／16.25／7.26**，所列門檻平均 **14.99**。這是區間偵測，不是影格 AUC。 |
| UCA 的 TIoU | [*Towards Surveillance Video-and-Language Understanding*，CVPR 2024／arXiv:2309.13925](https://arxiv.org/html/2309.13925) 使用 TIoU 評估**文字查詢的時間定位**；其結果不是無文字查詢的 VAD 定位分數。 |

**偏差的證據與修正。** [*Look Inside Each Video: Rethinking How Video Anomaly Detection Is Evaluated*，arXiv:2608.21854，表 17](https://arxiv.org/html/2608.21854) 顯示：UCF-Crime 即使只保留異常影片，給每支影片一個**不隨時間變化**的分數，理論上仍可達 **80.87 \(AUC_A\)**。因此，\(AUC_A\) 不能單獨證明事件已在影片內定位。該文建議增加逐影片、事件級與固定誤報操作點的評估。

[*Auditing Frame-Level AUC in Weakly Supervised Video Anomaly Detection*，arXiv:2608.11985，表 2](https://arxiv.org/html/2608.11985) 對同一批預測比較 pooled 與 within-video AUC：DSANet 為 **89.44／69.81**；GS-MoE（I3D）為 **91.58／86.79**。後一欄是**另一種指標**，不是上表的 \(AUC_A\)。[UMIL，CVPR 2023，UCF-Crime 結果表](https://arxiv.org/html/2303.12369) 則指出 MIL 偏重最高分、容易辨識的 snippet，可能學到場景 context shortcut；它加入較模糊 snippet 的訓練訊號，但其 **68.68 \(AUC_A\)** 不代表問題已消失。

**建議的對照方式：** 保留標準 pooled AUC，另報 pooled \(AUC_A\) 與「每支同時含正負影格的異常影片先算 AUC、再取平均」。若要衡量事件邊界，再報固定區間產生規則下的 mAP@tIoU。對 UCF-Crime 同時報 detection delay 與每小時誤報數的可核對論文結果：**not verified**。