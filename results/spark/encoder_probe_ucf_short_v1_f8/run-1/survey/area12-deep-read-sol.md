# 四篇可轉用的論文：因果式影片內事件定位

**判讀界線。** 四篇論文都沒有證明可提高 UCF-Crime 的 *macro per-video AUC*。下列論文結果與本專案實驗建議分開陳述。本文的主要指標是各有效異常影片的 AUC 平均值；跨影片合併計算的 frame AUC 不能替代它。所有建議的閾值與權重都應在 validation 影片設定，不得使用 test 標籤。

## 1. NOVA：讓「正常」參照貼近當前影片

**論文。** [*NOVA: Normal-Side Modeling for Training-Free Zero-Shot Video Anomaly Detection*](https://arxiv.org/abs/2609.06360)，arXiv preprint，2026，arXiv:2609.06360。

**領域與對應。** 論文研究 training-free 影片異常偵測。其每部測試影片相當於我們的一條 camera stream；影片的視覺外觀是「場景」，短暫異常是「事件」。

**論文機制。** NOVA 先以 frozen vision-language encoder 取得單位化的視覺向量。它比較向量與異常、正常文字描述的 cosine similarity，產生初步異常分數。Visual Normality Anchor（VNA）另取影片開頭的向量，以初步分數決定權重：
\[
w_i=\frac{\exp(-\hat s_i/\gamma)}{\sum_j\exp(-\hat s_j/\gamma)},\qquad
a=\operatorname{Normalize}\!\left(\sum_iw_ix_i\right).
\]
模型把 \(a\) 加入正常參照池；較像異常的開頭畫面因此較少影響參照。論文使用 \(N_{\rm warm}=32\)、\(\gamma=0.3\)（NOVA，§3.4、§4.1）。其 Motion-Aware Stabilization（MAS）還包含置中的向量平均、以相鄰向量差控制分數的 motion gate，以及 Gaussian smoothing（NOVA，§3.5）。

**論文證據。** 在相同 scoring backbone 下，加入 VNA 使 UCF-Crime **pooled frame AUC** 從 **85.62%** 到 **86.99%**；加入正常文字建模後，VNA 使它從 **88.79%** 到 **89.54%**（NOVA，**Table 2**）。這不是影片內 AUC。MAS 單獨使用時，XD-Violence AP 反而從 **80.36%** 降至 **72.88%**（NOVA，**Table 2**）；不能把 motion gate 視為已證實的獨立收益。

**我們可取之處。** 在已存的 frozen encoder window features 上，建立「低初步分數視窗權重較高」的場景參照。先用既有原始 head 分數或 training-free motion 分數作權重，再以目前 feature 與參照的距離產生**並列分數**；不要立即硬性扣除參照。部署時只能從已到達的視窗建立參照，並須提供 warm-up 期間的原始分數。這是對 VNA 的**因果改寫**，不是原論文已測試的串流設定。

**風險。** 原流程會用開頭整段建立參照，回頭評分開頭視窗，且 MAS 使用未來向量；整套報告結果**不符合零等待因果部署**。若事件一開始就出現，或原始分數把異常場景判作正常，參照仍會受污染。既有 causal past-mean 不穩定，也表示不能只憑 VNA 的 pooled AUC 預期本專案收益。

## 2. Anomize：保留場景線索，但限制它決定事件時間

**論文。** [*Anomize: Better Open Vocabulary Video Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.pdf)，CVPR，2025；[arXiv:2503.18094](https://arxiv.org/abs/2503.18094)。

**領域與對應。** 論文偵測已見與未見類別的影片異常。靜態分支保留場景及物件線索；動態分支表示隨時間改變的線索。兩者對應我們的 frozen-feature 場景分數與 training-free motion 分數。

**論文機制。** Frozen CLIP 產生逐影格向量 \(x_t\)。LSTM 產生 temporal vector；靜態分支直接使用 \(x_t\)。兩個分支各以相關文字向量做 attention augmentation，各自預測 \(s_{\rm sta}\)、\(s_{\rm dyn}\)，最後以
\[
s_t=\beta s_{{\rm dyn},t}+(1-\beta)s_{{\rm sta},t}
\]
融合（Anomize，§3.2、§3.4–3.5）。偵測訓練對**每個分支**取影片中最高分影格的平均，再以影片標籤施加加權 MIL binary cross-entropy；分類與偵測分階段訓練（Anomize，§3.6）。論文在 UCF-Crime 使用 \(\beta=0.5\)（Anomize，§4.1）。

**論文證據。** 完整 ablation 中，靜態單分支的 UCF-Crime **pooled frame AUC 為 84.26%**，動態單分支為 **83.21%**，融合後為 **84.49%**；XD-Violence AP 分別為 **59.26%**、**54.16%**、**69.31%**（Anomize，**Table 3**）。因此，融合在該表的 XD-Violence 指標較明顯；UCF-Crime 的靜態分支增益很小。Table 3 沒有測試影片內排序或串流因果性。

**我們可取之處。** 先直接融合已存的原始 head 分數與 training-free pixel-motion 分數，不訓練自由的 concat head。用 validation 影片固定兩個分數的校準方式及融合權重。另測「只有場景分數高且 motion 也高時才升級警報」的受限 gate；這是**我們的推論**，不是 Anomize 的加權和公式。兩種融合都只讀目前分數，因此可保持因果。

**風險。** 動作少的事件可能被 motion gate 壓低。攝影機移動也可能提高 motion。論文的 temporal encoder 未提供可直接證實零等待的對照；我們不能把其已發表結果當作因果 head 的證據。更重要的是，既有 motion head 會重新學到場景，因此不應先讓融合器自由學習跨影片捷徑。

## 3. First-shot ASD：把不同正常狀態當作可選參照

**論文。** [*First-Shot Anomaly Sound Detection for Machine Condition Monitoring: A Domain Generalization Baseline*](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000191.pdf)，EUSIPCO，2023；[arXiv:2303.00455](https://arxiv.org/abs/2303.00455)。

**領域與對應。** 論文處理機器聲音中的正常運轉與故障，並測試 domain shift。一台機器的運轉狀態相當於一個 camera 的正常場景；故障聲相當於事件。這不是同一段錄音內的起點定位。

**論文機制。** 作者只用正常聲音訓練小型 autoencoder，並以輸入與重建向量的誤差評分。Selective Mahalanobis 版本分別從 source、target 正常資料估計殘差 covariance，再取兩種 Mahalanobis distance 的**較小值**：
\[
A(x)=\min\{D_{\rm source}(x,\hat x),D_{\rm target}(x,\hat x)\}.
\]
「任何一種已知正常狀態都能解釋目前輸入」時，分數便保持低值（Harada 等，§III、Eqs. 8–10）。原研究每段聲音使用五個連續 log-mel frame 作 autoencoder 輸入；訓練只用正常樣本（Harada 等，§III–IV）。

**論文證據。** 在相同 random seed 的 DCASE 2022 Task 2 evaluation set，簡單 autoencoder 的 target-domain AUC harmonic mean 為 **0.5143**，Selective Mahalanobis 版本為 **0.5610**；其 source/target pAUC harmonic mean 為 **0.5335** 與 **0.5660**（Harada 等，**Table I**）。這是按機器 section、domain 評估的**跨狀態錄音 AUC**，不是同一 stream 的事件排序；也不是 unseen-camera 測試。

**我們可取之處。** 用已存正常影片 features 建立一個固定正常參照，再從當前影片已觀察、低警報的歷史建立第二個參照。最便宜的版本不訓練 autoencoder：在低維、經訓練集決定的 feature 空間估計兩組 regularized covariance，取兩個距離的較小值。**先評分，後更新**影片參照，並凍結高警報視窗的更新。此改寫只使用目前及過去 features。

**風險。** 對「正常參照」取最小距離也可能把持續事件當成新的正常狀態。影片起點沒有可靠的 camera 參照；短影片也不足以估計高維 covariance。原論文有提供 target-domain 正常訓練錄音，不能等同於本專案部署時從未知影片自行取得的正常前綴。

## 4. Contextual Matrix Profile：比較同一人的歷史形態，而非全體人的水準

**論文。** [*An Unsupervised Data-Driven Anomaly Detection Approach for Adverse Health Conditions in People Living With Dementia: Cohort Study*](https://aging.jmir.org/2022/3/e38211/)，*JMIR Aging*，2022；無 arXiv ID。

**領域與對應。** 論文用家庭 motion sensors 監測個別患者的活動變化。家庭及患者是持續的「場景」；感染或住院相關活動變化是「事件」；每日感測資料是 stream。

**論文機制。** Contextual Matrix Profile（CMP）把連續資料切成 subsequences，再把相鄰 subsequences 組成 context。它以 z-normalized Euclidean distance 比較 context 中的片段與**同一患者過去**的片段，形成異常分數。多維版本先按各 feature 的距離排序，再平均最小的數個距離；其 \(k=1\) 設定選兩個距離最小的 feature。論文以三個 subsequence 組成 context，每個 subsequence 長三天；表現較平衡的警報設定用七天歷史分數窗及 robust-z 閾值 **1.65**（Bijlani 等，方法；[作者稿 Table 2：參數](https://s3.ca-central-1.amazonaws.com/assets.jmir.org/assets/preprints/preprint-38211-accepted.pdf)）。論文說明評估時每一時間點只使用既有歷史。

**論文證據。** Equal-weighted multidimensional CMP 的患者平均 event recall 為 **84.3%**，每位患者平均產生 **32.1** 次警報；LODA 的對應值為 **85.7%** 與 **37.8** 次（Bijlani 等，[作者稿 **Table 3：結果**](https://s3.ca-central-1.amazonaws.com/assets.jmir.org/assets/preprints/preprint-38211-accepted.pdf)；[期刊版 **Table 2**](https://aging.jmir.org/2022/3/e38211/)）。結果以患者為單位彙整，較接近我們的 per-video 評估。不過標籤只涵蓋部分健康事件，因此警報次數**不是已核實的誤報次數**；寬鬆的事件日期配對也不是 window AUC。

**我們可取之處。** 在已存 window features 上，僅比較當前短序列與**同影片已到達的正常候選短序列**。先以 pixel motion、少量訓練集選定的 feature 維度或原始 score 作輸入，避免高維 cosine nearest neighbor 再度只辨認場景。每個 window 結束後才輸出其 context 分數；只把此前的候選序列加入 reference。這個簡化版可因果執行，但必須另測事件開始時的延遲。

**風險。** z-normalization 可能刪除「motion 幅度突然升高」這個有用訊號。重複事件也可能互相匹配，導致低 novelty。長 context 會把約數個 window 的短事件與正常內容混合；論文以日為尺度選出的參數不能直接換算成影片 window。

## 綜合判斷與每篇的一個最便宜實驗

共同想法不是再加強跨影片分類，而是**保留全域語義，同時加入受約束的本 stream 正常參照或局部變化訊號**。NOVA 與機器聲論文顯示：正常參照若來自當前對象，必須防止事件污染。Anomize 顯示：靜態與動態訊號可互補，但融合的收益依資料集而變。CMP 顯示：可直接對每條 stream 的歷史形態評分。這些機制也解釋本專案的兩項失敗：只換 ranking loss 仍允許相同 frozen features 表達場景；直接扣除不可靠的 past mean 則可能同時扣除事件，或把事件吸收入參照。**這是本專案的推論，不是四篇論文的實驗結論。**

| 論文 | 在既存資料上的一個實驗 | 主要指標與放棄條件 |
|---|---|---|
| NOVA | 用既有分數加權已到達的低分 feature，建立固定 warm-up anchor；比較「原始分數」與「原始分數＋anchor distance」。 | 比較未見影片的 macro per-video AUC、事件前分數及 warm-up 期間結果。若只改善 pooled AUC，或初始事件污染 anchor，便放棄。 |
| Anomize | 在 validation 固定原始 head 分數與 training-free pixel motion 的加權和；不訓練融合 head。 | 比較 macro per-video AUC，以及固定正常影片警報率下的事件偵測率。若收益只來自 pooled AUC，或靜態事件漏報增加，便放棄。 |
| First-shot ASD | 比較「全域正常參照距離」與「全域、因果影片參照距離取最小值」；使用低維 regularized covariance。 | 比較 macro per-video AUC、事件前高分比例與持續事件分數。若事件被影片參照吸收，或結果對初始視窗敏感，便放棄。 |
| CMP | 用 motion 與少量既存 score 建立短 context，查詢同影片過去的候選 context；不使用全片 nearest neighbor。 | 比較 macro per-video AUC、起點延遲及正常影片每小時警報數。若距離只在事件起點短暫升高，卻降低整段事件排序，便停止把它當 window score；可另評估 onset alarm。 |

四個實驗都須按影片切分 train、validation、test。另報各影片 AUC 分布，而不是只報平均值。首次警報延遲與正常 stream 警報率須使用相同的因果更新規則評估。