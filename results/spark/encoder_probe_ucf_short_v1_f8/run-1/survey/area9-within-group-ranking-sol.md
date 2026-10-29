# 影片內排序損失調查

| 方法與論文 | 核心想法；因果性 | 標籤；輸入；儲存特徵適用性 | 程式碼；論文結果 |
|---|---|---|---|
| **PDAOM** — [*Enhancing Personalized Ranking With Differentiable Group AUC Optimization*](https://arxiv.org/abs/2304.09176)，ICDM NeuRec workshop，2022；arXiv:2304.09176 | 在同一使用者的項目中，挑選最低分正例與最高分負例，優化困難配對。套用到影片時，逐窗評分即可保持因果。 | **窗標籤**；項目特徵。可直接將使用者換成影片，並以儲存特徵訓練小型 head。 | 無已核實程式碼。Meituan CTR 的 DeepFM **group AUC 67.41 → 67.81**；[PDAOM，Table II](https://arxiv.org/html/2304.09176)。這是使用者內 AUC，不是影片內結果。 |
| **ListNet** — [*Learning to Rank: From Pairwise Approach to Listwise Approach*](https://doi.org/10.1145/1273496.1273513)，ICML 2007；arXiv ID 未找到 | 在每個 query 內，讓預測分數與相關性標籤的 softmax 分布接近。逐窗 head 因果；整組 softmax **只用於訓練**。 | **窗標籤**；項目特徵。可把影片當 query，但稀疏正例可能使完整影片的 softmax 目標過於分散。 | [第三方實作](https://github.com/valeriobasile/listnet)。論文報告 query 排序結果；可核實的影片內 AUC 數字：無。 |
| **局部 partial-AUC** — [*When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee*](https://proceedings.mlr.press/v162/zhu22g.html)，ICML 2022；arXiv:2203.00176 | 著重難負例，以優化指定區間的 partial AUC。**原方法不會自動限制配對屬於同一組**。逐窗 head 可保持因果。 | **窗標籤**；項目特徵。須自行實作影片遮罩或影片內取樣；不能直接把 LibAUC 的全資料配對當作影片內損失。 | [LibAUC](https://github.com/Optimization-AI/LibAUC)；[損失文件](https://docs.libauc.org/api/libauc.losses.html)。可核實的影片內 AUC 數字：無。 |
| **Sultani MIL ranking** — [*Real-world Anomaly Detection in Surveillance Videos*](https://arxiv.org/abs/1801.04264)，CVPR 2018；arXiv:1801.04264 | 比較異常影片最高分片段與正常影片最高分片段。異常影片的 sparsity 項抑制全片高分，但**排序項本身不比較同片段的正負窗**。逐窗 head 可保持因果。 | **影片標籤**；片段特徵。可套用儲存特徵；勿將論文的非因果特徵抽取方式帶入部署。 | [作者程式碼](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)。UCF-Crime pooled frame AUC **74.44%（無 sparsity／smoothness）→ 75.41%（兩者皆有）**；[Sultani 等，Table 3](https://arxiv.org/html/1801.04264v3)。 |
| **RTFM** — [*Weakly-Supervised Video Anomaly Detection With Robust Temporal Feature Magnitude Learning*](https://arxiv.org/abs/2101.10030)，ICCV 2021；arXiv:2101.10030 | 用特徵幅度輔助 MIL 挑選異常片段。原架構的時序上下文須改成僅看過去；幅度仍可能攜帶場景訊號。 | **影片標籤**；片段特徵。可訓練 head，但需修改原網路與取樣流程。 | [作者程式碼](https://github.com/tianyu0207/RTFM)。UCF-Crime pooled frame AUC **84.30%**；[RTFM，Table 2](https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf)。 |
| **UMIL** — [*Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection*](https://arxiv.org/abs/2303.12369)，CVPR 2023；arXiv:2303.12369 | 區分可信與模糊片段，再用配對及自訓練訊號降低情境偏差。**不是明確的同影片正負窗排序損失**。部署時須替換任何未來時序上下文。 | **影片標籤**；片段特徵。可移植到儲存特徵，但比簡單排序 head 複雜。 | [作者程式碼](https://github.com/ktr-hubrt/UMIL)。UCF-Crime「僅異常影片」pooled frame AUC **60.57%（MIL baseline）→ 68.68%（UMIL）**；[UMIL，Table 1](https://arxiv.org/html/2303.12369)。**此指標不是各影片 AUC 的平均。** |
| **DSCIL** — [*DSCIL: Dynamic selected contrastive instance learning for weakly supervised video anomaly detection*](https://www.sciencedirect.com/science/article/pii/S0031320325013196)，*Pattern Recognition*，2025 線上發表；arXiv ID 未找到 | 從異常影片挑選疑似異常及正常片段，加入 contrastive 學習。選錯的偽負例會直接傷害定位；時序模組須檢查並改為因果。 | **影片標籤**；片段特徵。可研究其選樣策略並套用儲存特徵，不宜假定原架構可直接部署。 | [作者程式碼](https://github.com/zengkun301/DSCIL)。影片內 AUC 或延遲結果：**not verified**。 |

## 實驗判斷

- **有窗標籤時，先只配對同一影片的事件窗與非事件窗。** 同場景配對直接對準部署目標，也避免以影片身分解決排序任務。用每片平均損失，避免長影片主導訓練。
- **跨影片配對應作獨立 ablation，而非混入主要影片內損失。** 可另加正常影片窗作校準或抑制誤報，並分別測試同片配對、跨片配對、兩者並用。上述論文沒有提供能核實此項取捨的同設定影片內 AUC 比較。
- **只有影片標籤時，sparsity 與片內高低分片段對比可抑制「全片高分」，但不能證明找到事件。** Sultani 的 sparsity 提供明確約束；DSCIL 的片內選樣則依賴偽標籤。單獨的跨影片 MIL ranking 仍容許場景捷徑。
- **成本主要在配對，不在小型 head。** 全配對成本隨每片正、負窗數乘積增加；每個正窗抽取少量同片負窗即可控制成本。事件窗稀少時，保留每個正窗、抽取難負例與隨機負例，並檢查標註邊界。線性或 MLP head 可逐窗推論；小型時序 head 必須使用 causal mask，且訓練與推論須採相同歷史範圍。
- **評估需同時列出 pooled AUC、僅異常影片 pooled AUC、各有效影片 AUC 的 macro 平均與誤報。** 只有同時含正負窗的影片才有影片內 AUC。找到的 VAD 論文未提供可核實的「改用影片內損失」前後、**各影片 AUC macro 平均**對照；UMIL 的異常影片 pooled AUC 不能代替它。

## 優先嘗試

1. **同片 pairwise logistic loss，加窗級 BCE。** 它直接針對影片內正負排序；先以線性 head 建立因果基線，再比較是否需要 MLP。此為建議實驗，不是上述論文已報告的影片結果。
2. **同片困難配對 PDAOM，加窗級 BCE。** 它與 macro 影片內 AUC 的分組方式最接近；比較只用最難配對與混合隨機配對，避免少數錯標窗控制梯度。
3. **同片 partial-AUC 難負例損失。** 若部署重視低誤報率，測試影片內難負例遮罩；與完整影片內 AUC、誤報及事件開始時的警報表現一起比較。