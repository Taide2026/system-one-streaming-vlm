# 因果式影片內事件定位：診斷與優先實驗

**判斷：目前的主要問題不是分類 head 不夠強，而是固定場景訊號比事件訊號更容易學。** 事件訊號仍存在：training-free 的特徵變化與 pixel motion 能在事件前維持接近正常影片的分數，同時提供影片內區分力。但現有證據不能判定，剩餘限制主要來自場景數不足，還是局部、短暫事件已在特徵池化時流失。

以下專案數字均來自**本 brief 的「Evidence」對應項目**，不是已發表論文。文獻數字則逐項註明論文及表格。`AUC_A` 是將異常影片影格合併計算的 AUC；它**不是**各影片 AUC 的平均。

## 診斷

| 可能原因 | 我們的證據 | 判斷與最便宜的區分檢查 |
|---|---|---|
| **場景主導特徵，事件只是較小的殘差** | 事件前視窗已明顯高於正常影片；整片均值扣除改善影片內 AUC，但不能因果部署。Training-free 的特徵變化與 pixel motion 則使事件前分數接近正常影片。（本 brief，Evidence 2、4、6、7） | **最強的直接證據。** 在每部測試影片內，對原始分數、motion、變化分數分別畫事件前、起點及事件期間曲線；再檢查收益是否來自同一批影片。 |
| **訓練場景不足，無法學到跨攝影機事件規則** | Linear head 在 train 表現高，在 val/test 明顯下降；增加 head 容量未改善。這符合場景相關規則過度擬合，但不是場景數不足的獨立證明。（本 brief，Evidence 3） | **仍可能。** 按可辨識的 scene/camera 群組切分 train/validation；比較相近場景與未見場景的影片內 AUC。不要把同場景片段分到兩側。 |
| **監督獎勵「這是一部異常影片」** | Window-label LoRA 使事件前分數更高、事件期間相對事件前的差距更小；同影片排名、事件前 negative 與 JTT 均未可靠改善定位。（本 brief，Evidence 2、5） | **確定存在，但只改 loss 不足以解決。** 檢查訓練集是否也出現事件前高分；若是，再比較 scene-only、motion-only 與完整模型的錯誤影片，而非再單獨調整配對權重。 |
| **取樣與 patch-mean pooling 遺失局部或短事件** | Motion 能提供與場景分數不同的訊號；多種 pooled encoder 和 head 都有相似的影片內限制。這與資訊流失相容，卻不證明每個事件都可由更細特徵找回。（本 brief，Evidence 1、3、6、7） | **關鍵未決原因。** 從相同的事件、事件前與 hard-normal 視窗抽取未池化 patch／person 特徵；對照 pooled 特徵做影片內正負配對。已儲存特徵只能選取測例，不能證明未儲存的空間資訊不存在。 |
| **標註邊界或事件定義限制；部分事件幾乎靜止** | 測試 GT 與 HOLMES-VAU train window labels 的粒度不同；motion 不可能偵測所有靜態事件。但以測試 GT 做 video-fold oracle 仍未得到一致的高定位表現。（本 brief，Setting；Evidence 3、7） | **部分限制，不能作為總體失敗的充分解釋。** 人工複核少量最差影片的起點、可見事件位置及靜態／動態類型；分別報告邊界容差內外的結果。勿用測試 GT 訓練正式模型。 |

## 方法家族：能解決甚麼，不能解決甚麼

| 家族 | 對應原因與部署判斷 | 相關已發表證據及界限 |
|---|---|---|
| **局部 patch、object/person track、pose；較密集的短片段取樣** | 直接測試 pooled 特徵是否漏掉小型或短暫事件。可逐窗因果運作；但 tracks 與 pose 也可能漏掉無人事件。先做局部特徵 ablation，再決定是否重跑全部影片。 | 本次所讀報告沒有可核實、且同時符合未見場景與逐影片定位的對照數字：**not verified**。 |
| **Motion-centric／two-stream，加場景分數與變化分數融合** | Motion 分支對準動作事件；原始分支保留火災、物品或持續狀態等可能低動態的事件。固定或受限權重的融合，比再訓練一個會學場景的自由 head 更值得先試。 | 本專案的直接依據是 training-free motion 與變化分數（本 brief，Evidence 6、7）。[StillMix，ICCV 2023，Table 2](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Mitigating_and_Evaluating_Static_Bias_of_Action_Representations_in_the_ICCV_2023_paper.pdf) 的 TSM rare-scene action accuracy 為 **57.86%→59.69%**；這是跨場景 action recognition，**不是** VAD 定位。 |
| **僅依攝影機過去建立的 novelty、prediction error、場景條件化** | 對準持續場景捷徑。以全域正常先驗處理串流起點；先評分，再以未警報的過去視窗更新基線。不要以整片均值或未來視窗正規化。 | `area2-prediction-error-sol.md`、`area6-seizure-eeg-sol.md` 與 `area7-speech-online-normalization-sol.md` 提供可轉用設計，**未驗證 UCF-Crime 逐影片收益**。Novelty 會對 camera motion 誤報，也可能在持續事件後失效。 |
| **Causal sequence head、onset head、change-point alarm** | 能整合短期證據或改善起點決策；**不能還原已被 pooling 丟失的物件**，也不會自動移除場景捷徑。應在有用的分支分數確立後測試。 | [StartNet，ICCV 2019，Table 1](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_StartNet_Online_Detection_of_Action_Start_in_Untrimmed_Videos_ICCV_2019_paper.pdf)：THUMOS’14、容許起點偏差 **2 s** 時，**27.2% p-mAP**；容差不是處理延遲。`area1-online-action-detection-sol.md` 與 `area4-change-point-sol.md` 的方法沒有對題的逐影片 VAD 數字。 |
| **不同監督：可信 frame/window 標籤、片內 negative、困難片段抑制** | 可減少異常影片身分這個訓練捷徑；但現有片內排名失敗，表示不能只換 loss 而保留相同資訊瓶頸。MIL 尤須防止「挑到最像異常場景的片段」。 | [UMIL，CVPR 2023，Table 1](https://arxiv.org/html/2303.12369)：其 baseline 與 UMIL 的 UCF-Crime 異常影片合併 `AUC_A` 為 **60.57%→68.68%**。[DSS＋ASR，WACV 2025，Table 3](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Discriminative_Score_Suppression_for_Weakly_Supervised_Video_Anomaly_Detection_WACV_2025_paper.pdf)：**64.01%→69.38%**；**單用 ASR 為 61.05%**。兩者均未證明 macro per-video、因果推論或未見攝影機收益。參見 `area9-within-group-ranking-sol.md`、`area10-event-vs-scene-shortcut-sol.md`。 |
| **增加場景多樣性、合成事件移植、VLM change prompt／多窗 context／LoRA** | 多樣性可降低場景與標籤的關聯；VLM 可看局部事件及前後變化。合成邊界可能成為新捷徑。Prompt 與 LoRA 必須只讀當前及過去視窗，並以同影片 negative 評測。 | StillMix 的跨場景證據如上。對本設定的合成事件、change prompt 或多窗 LoRA **逐影片收益：not verified**；現有 LoRA 結果反而要求先檢查捷徑（本 brief，Evidence 2）。 |

## 排名前三的實驗

**共同決策規則。** 使用未見影片及可取得的未見 scene/camera 分組。主要報告各有效異常影片 AUC 的 macro 平均、起點後首次警報延遲、事件偵測率與正常串流的每小時誤報；另報合併 `AUC_A`。用 validation 正常串流選定閾值、融合權重及 alarm reset 規則，不用 test GT 調參。

1. **先查局部資訊是否已被 pooling 丟失，再決定是否投資新 encoder。**  
   **原因：**若這是瓶頸，增加 loss、GRU 或攝影機基線都無法重建事件物件。**最便宜的決定性實驗：**用已儲存的分數與 motion 選取成功、失敗及 hard-normal 視窗；只對這些視窗重抽未池化 patch 特徵或 person crops，做同影片事件／事件前測試，並保留 pooled 對照。**支持結果：**多個未見影片在事件所在區域出現穩定增益，且 hard-normal 不同步升分。**放棄條件：**人工可見的局部事件仍不可分，或增益只出現在 train／少數影片。這項決定性檢查**不能只靠已儲存的 patch mean**完成。

2. **以受限的 scene-score＋motion/change-score 融合，測試互補訊號。**  
   **原因：**現有 training-free 動態分數沒有明顯事件前捷徑；場景分支則可能補足靜態或持續事件。**最便宜的決定性實驗：**直接使用已存的 head 分數、特徵差與 pixel-motion 值；在 validation 固定融合與閾值，逐窗因果評估，並按事件類型及影片檢查增益。先試固定權重或明確的雙分支 alarm，勿先訓練自由 concat head。**支持結果：**影片內排序、及固定誤報率下的及早警報同時改善；改善分布於不同影片。**放棄條件：**收益只提高合併 frame AUC、增加正常影片誤報，或只對高動態事件有效。

3. **在保留原始特徵下，測試有先驗與更新保護的因果攝影機參照。**  
   **原因：**它直接處理固定場景，同時避免非因果的整片均值扣除。**最便宜的決定性實驗：**在已存特徵及 motion 分數上比較「無參照」、「僅過去的收縮場景摘要」與「先評分、只用未警報窗更新」；使用相同 head、訓練／推論歷史及正常串流閾值。**支持結果：**事件前高分下降，事件期間相對事件前的差距增加，且串流起點與持續事件不失效。**放棄條件：**跨 seed 或更新規則仍像既有 causal past-mean 一樣不穩定，或基線吸收短事件。若失敗，應優先回到方向一；不要把更複雜的 change-point 模型當作資訊流失的修補。