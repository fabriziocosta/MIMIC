# MIMIC Oversampling Experiment Plan

## Aim

Run MIMIC on the same dataset configurations and evaluation protocols used in major oversampling papers, then compare MIMIC's reported numbers directly against the published tables. The goal is to avoid rerunning every competitor method when the original papers already report comparable baselines.

Rule for every experiment: MIMIC is trained and sampled only inside each paper's training fold or training split. Published competitor numbers are copied as reference targets, and MIMIC is evaluated with the same classifier family and performance metric wherever practical.

Implementation pattern for notebooks and reusable experiment modules is captured in `manuscript/experiments/notebook_pattern.md`. Use that pattern when adding Q2 and later experiment notebooks.

## Question 1: Can MIMIC Match Classical SMOTE-Style ROC Utility?

### Competitor Paper

- Chawla, Bowyer, Hall, and Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique", JAIR, 2002. https://www.jair.org/index.php/jair/article/view/10302

### Replication Target

Use this as the historical ROC/AUC benchmark. The paper evaluates SMOTE plus majority under-sampling against plain under-sampling and cost-sensitive alternatives using C4.5, RIPPER, and Naive Bayes. It reports ROC curves, ROC convex hull membership, and AUC-style summaries rather than modern train/test tables.

### Dataset Configurations

| Dataset | Majority | Minority | Notes for MIMIC run |
|---|---:|---:|---|
| Pima | 500 | 268 | UCI Pima Indian Diabetes; minority is diabetes-positive. |
| Phoneme | 3818 | 1586 | ELENA phoneme; distinguish nasal vs oral sounds. |
| Adult | 37155 | 11687 | UCI Adult; paper evaluates SMOTE-NC and continuous-only SMOTE. Run both MIMIC mixed-type and numeric-only variants. |
| E-state | 46869 | 6351 | NCI yeast anti-cancer screen descriptors; likely harder to source, mark as optional. |
| Satimage | 5809 | 626 | Collapse original multiclass problem to smallest class vs rest. |
| Forest Cover | 35754 | 2747 | Use Ponderosa Pine vs Cottonwood/Willow only. |
| Oil | 896 | 41 | Oil-spill detection. |
| Mammography | 10923 | 260 | Mammography abnormality detection. |
| Can | 435512 | 8360 | Optional; very large and source may need reconstruction. |

### Protocol To Mirror

- Use 10-fold cross-validation.
- Compute percent true positive and percent false positive for each fold, then average.
- Generate ROC points by varying the amount of minority oversampling and majority under-sampling.
- SMOTE used five nearest neighbors; MIMIC should use the closest comparable `GenerationPolicy(n_neighbors=5)`.
- Main MIMIC run: `MIMIC-factorised` with class-conditioned generation to match the minority count created by each sampling point.
- MIMIC ablation: `MIMIC-identity` to check whether learned embedding generation beats feature-space interpolation.

### Performance Measures

- Primary: AUC from ROC points using the trapezoidal rule.
- Secondary: ROC convex hull membership compared with published SMOTE/under-sampling curves.
- Adult-specific: report mixed-type MIMIC against the SMOTE-NC motivation, and numeric-only MIMIC against the paper's continuous-feature SMOTE setup.

### Published C4.5 AUC Reference Values

The SMOTE paper reports AUC values for C4.5 in Table 3. The table prints AUCs as four digits without a decimal point; for this plan they are recorded as decimals. These are the direct AUC values to compare against MIMIC's C4.5-compatible decision-tree approximation.

| Dataset | Under-sampling AUC | 50 SMOTE | 100 SMOTE | 200 SMOTE | 300 SMOTE | 400 SMOTE | 500 SMOTE | Best reported C4.5 AUC | Best reported setting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pima | 0.7242 |  | 0.7307 |  |  |  |  | 0.7307 | 100 SMOTE |
| Phoneme | 0.8622 |  | 0.8644 | 0.8661 |  |  |  | 0.8661 | 200 SMOTE |
| Satimage | 0.8900 |  | 0.8957 | 0.8979 | 0.8963 | 0.8975 | 0.8960 | 0.8979 | 200 SMOTE |
| Forest Cover | 0.9807 |  | 0.9832 | 0.9834 | 0.9849 | 0.9841 | 0.9842 | 0.9849 | 300 SMOTE |
| Oil | 0.8524 |  | 0.8523 | 0.8368 | 0.8161 | 0.8339 | 0.8537 | 0.8537 | 500 SMOTE |
| Mammography | 0.9260 |  | 0.9250 | 0.9265 | 0.9311 | 0.9330 | 0.9304 | 0.9330 | 400 SMOTE |
| E-state | 0.6811 |  | 0.6792 | 0.6828 | 0.6784 | 0.6788 | 0.6779 | 0.6828 | 200 SMOTE |
| Can | 0.9535 | 0.9560 | 0.9505 | 0.9505 | 0.9494 | 0.9472 | 0.9470 | 0.9560 | 50 SMOTE |

Notes:

- Table 3 is C4.5-only; the paper does not provide corresponding AUC tables for Ripper or Naive Bayes.
- Adult is not included in Table 3. The paper discusses Adult separately for SMOTE-NC and continuous-only SMOTE, where the reported comparison is primarily graphical and described as not improving over plain under-sampling by AUC.
- MIMIC should be compared both to the under-sampling AUC and to the best reported SMOTE C4.5 AUC for each available dataset.

### Comparable Number To Produce

For each available dataset, produce:

| Dataset | Classifier | Under-sampling AUC | Best published SMOTE C4.5 AUC | MIMIC AUC | MIMIC vs best SMOTE | MIMIC ROC hull status |
|---|---|---:|---:|---:|---:|---|
| Pima | C4.5-compatible tree | 0.7242 | 0.7307 | 0.7095 | -0.0212 | pending hull comparison |
| Phoneme | C4.5-compatible tree | 0.8622 | 0.8661 | 0.8597 | -0.0064 | pending hull comparison |
| Satimage | C4.5-compatible tree | 0.8900 | 0.8979 | | | |
| Forest Cover | C4.5-compatible tree | 0.9807 | 0.9849 | | | |
| Oil | C4.5-compatible tree | 0.8524 | 0.8537 | | | |
| Mammography | C4.5-compatible tree | 0.9260 | 0.9330 | | | |
| E-state | C4.5-compatible tree | 0.6811 | 0.6828 | | | |
| Can | C4.5-compatible tree | 0.9535 | 0.9560 | | | |
| Adult | C4.5-compatible tree | not tabulated | not tabulated | | | SMOTE-NC / continuous-only comparison is graphical/descriptive |

Filled MIMIC values currently come from:

- `manuscript/artifacts/q1_smote_roc/tables/pima__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv`
- `manuscript/artifacts/q1_smote_roc/tables/phoneme__run_full__factorised__cap-0.25__smote-normal-k5__seed-0__summary.csv`

Implementation note: if exact C4.5/RIPPER tooling is not practical, use scikit-learn decision tree as a transparent approximation and label the table "protocol-aligned, classifier-substituted".

## Question 2: Does MIMIC Improve Adaptive Hard-Region Performance Compared With ADASYN?

### Competitor Paper

- He, Bai, Garcia, and Li, "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning", IJCNN, 2008. https://ieeexplore.ieee.org/document/4633969

### Replication Target

Use Table II from the ADASYN paper as a compact direct-comparison target. The paper compares decision tree, SMOTE, and ADASYN on five datasets and reports overall accuracy, precision, recall, F-measure, and G-mean.

### Dataset Configurations

| Dataset | Specific setup from paper |
|---|---|
| Vehicle | Binary imbalanced vehicle configuration used by the paper. |
| Pima Indian Diabetes | UCI Pima; minority is diabetes-positive. |
| Vowel recognition | Binary imbalanced vowel setup used by the paper. |
| Ionosphere | UCI Ionosphere. |
| Abalone | Use class 18 as minority and class 9 as majority; remove the discrete `sex` feature, leaving 42 minority, 689 majority, and 7 numeric attributes. |

### Protocol To Mirror

- Base classifier: decision tree.
- Repeat 100 runs.
- In each run, randomly select half of minority and half of majority examples for training; use the remaining half for testing.
- For SMOTE and ADASYN reference settings: `K=5`; SMOTE `N=200`; ADASYN `beta=1`, `dth=0.75`.
- MIMIC should train only on the half-split training set and generate enough minority rows to match the ADASYN/SMOTE balancing intent.

### Published Reference Table

| Dataset | Method | OA | Precision | Recall | F-measure | G-mean |
|---|---|---:|---:|---:|---:|---:|
| Vehicle | Decision tree | 0.9220 | 0.8454 | 0.8199 | 0.8308 | 0.8834 |
| Vehicle | SMOTE | 0.9239 | 0.8236 | 0.8638 | 0.8418 | 0.9018 |
| Vehicle | ADASYN | 0.9257 | 0.8067 | 0.9015 | 0.8505 | 0.9168 |
| Pima Indian Diabetes | Decision tree | 0.6831 | 0.5460 | 0.5500 | 0.5469 | 0.6430 |
| Pima Indian Diabetes | SMOTE | 0.6557 | 0.5049 | 0.6201 | 0.5556 | 0.6454 |
| Pima Indian Diabetes | ADASYN | 0.6837 | 0.5412 | 0.6097 | 0.5726 | 0.6625 |
| Vowel recognition | Decision tree | 0.9760 | 0.8710 | 0.8700 | 0.8681 | 0.9256 |
| Vowel recognition | SMOTE | 0.9753 | 0.8365 | 0.9147 | 0.8717 | 0.9470 |
| Vowel recognition | ADASYN | 0.9678 | 0.7603 | 0.9560 | 0.8453 | 0.9622 |
| Ionosphere | Decision tree | 0.8617 | 0.8403 | 0.7698 | 0.8003 | 0.8371 |
| Ionosphere | SMOTE | 0.8646 | 0.8211 | 0.8032 | 0.8101 | 0.8489 |
| Ionosphere | ADASYN | 0.8686 | 0.8298 | 0.8095 | 0.8162 | 0.8530 |
| Abalone | Decision tree | 0.9307 | 0.3877 | 0.2929 | 0.3249 | 0.5227 |
| Abalone | SMOTE | 0.9121 | 0.2876 | 0.3414 | 0.3060 | 0.5588 |
| Abalone | ADASYN | 0.8659 | 0.2073 | 0.4538 | 0.2805 | 0.6291 |

### Comparable Number To Produce

Add MIMIC rows to the same table:

| Dataset | Method | OA | Precision | Recall | F-measure | G-mean |
|---|---|---:|---:|---:|---:|---:|
| Vehicle | MIMIC-factorised | | | | | |
| Pima Indian Diabetes | MIMIC-factorised | | | | | |
| Vowel recognition | MIMIC-factorised | | | | | |
| Ionosphere | MIMIC-factorised | | | | | |
| Abalone | MIMIC-factorised | | | | | |

## Question 3: Does MIMIC Compete With Geometric SMOTE on Region-Based Oversampling?

### Competitor Paper

- Douzas and Bacao, "Geometric SMOTE: Effective oversampling for imbalanced learning through a geometric extension of SMOTE", 2017 arXiv / later Information Sciences version. https://arxiv.org/abs/1709.07377

### Replication Target

Use the Geometric SMOTE Table 2 as the most convenient modern benchmark for SMOTE, Borderline-SMOTE1, Borderline-SMOTE2, ADASYN, and G-SMOTE. The table reports F-measure, G-mean, and AUC for two classifiers.

### Dataset Configurations

| Dataset | Features | Instances | Minority | Majority | IR |
|---|---:|---:|---:|---:|---:|
| Breast | 9 | 106 | 36 | 70 | 1.94 |
| Ecoli | 7 | 336 | 52 | 284 | 5.46 |
| Eucalyptus | 8 | 642 | 98 | 544 | 5.55 |
| Glass | 9 | 214 | 70 | 144 | 2.06 |
| Haberman | 3 | 306 | 81 | 225 | 2.78 |
| Heart | 13 | 270 | 120 | 150 | 1.25 |
| Iris | 4 | 150 | 50 | 100 | 2.00 |
| Libra | 90 | 360 | 72 | 288 | 4.00 |
| Liver | 6 | 345 | 145 | 200 | 1.38 |
| Pima | 8 | 768 | 268 | 500 | 1.87 |
| Segment | 16 | 2310 | 330 | 1980 | 6.00 |
| Vehicle | 18 | 846 | 199 | 647 | 3.25 |
| Wine | 13 | 178 | 71 | 107 | 1.51 |

### Protocol To Mirror

- Classifiers: Gradient Boosting Classifier and Logistic Regression.
- Cross-validation: fivefold CV.
- Oversample each training fold until the resulting training set is perfectly balanced.
- Metrics: F-measure, G-mean, AUC.
- Hyperparameter search: use the paper's reported classifier grids where practical: GBC `max_depth in {5, 8}` and `n_estimators in {50, 100}`.
- MIMIC sampling target: generate `majority_count - minority_count` minority rows inside each training fold.

### Published Reference Methods

The published table includes these columns:

- No oversampling.
- SMOTE.
- Borderline SMOTE1.
- Borderline SMOTE2.
- ADASYN.
- Geometric SMOTE.

### Comparable Number To Produce

For every dataset, classifier, and metric in the paper, append:

| Dataset | Classifier | Metric | Published best method/value | MIMIC-identity | MIMIC-factorised |
|---|---|---|---:|---:|---:|
| Breast | GBC | F | from Geometric SMOTE Table 2 | | |
| Breast | GBC | G | from Geometric SMOTE Table 2 | | |
| Breast | GBC | AUC | from Geometric SMOTE Table 2 | | |
| Breast | LR | F | from Geometric SMOTE Table 2 | | |
| ... | ... | ... | ... | | |

Keep the full row-level result table in experiment outputs; the manuscript can summarize average rank against the published methods.

## Question 4: Does MIMIC Compete With Local-Manifold Oversampling on High-Imbalance and High-Dimensional Tasks?

### Competitor Paper

- Bej, Davtyan, Wolfien, Nassar, and Wolkenhauer, "LoRAS: an oversampling approach for imbalanced datasets", Machine Learning, 2021. https://link.springer.com/article/10.1007/s10994-020-05913-4

### Replication Target

Use LoRAS Table 4 and Table 5. This is the most relevant benchmark because LoRAS explicitly motivates local approximation of a minority-class data manifold.

### Dataset Configurations

| Dataset | IR | Samples | Features | Category |
|---|---:|---:|---:|---|
| Abalone_19 | 130:1 | 4177 | 10 | high imbalance |
| Arrythmia | 17:1 | 452 | 278 | high dimensional / small minority |
| Isolet | 12:1 | 7797 | 617 | high dimensional |
| Letter-img | 26:1 | 20000 | 16 | high imbalance |
| Mammography | 42:1 | 11183 | 6 | high imbalance |
| Scene | 13:1 | 2407 | 294 | high dimensional |
| Ozone_level | 34:1 | 2536 | 72 | high imbalance |
| Webpage | 33:1 | 34780 | 300 | high imbalance and high dimensional |
| Wine-quality | 26:1 | 4898 | 11 | high imbalance |
| Yeast-me2 | 28:1 | 1484 | 8 | high imbalance |
| Yeast-ml8 | 13:1 | 2417 | 103 | high dimensional |
| Credit fraud | 577:1 | 284807 | 28 | extreme imbalance |
| ar1 | 12.44:1 | 121 | 30 | small minority |
| ar3 | 6.8:1 | 63 | 30 | small minority |

### Protocol To Mirror

- Cross-validation: five repetitions of stratified 10-fold CV.
- Exceptions: `ar1` and `ar3` use five repetitions of stratified threefold CV.
- Oversample only the training fold; leave test fold untouched.
- Balance minority count as close as possible to the majority count.
- Neighborhood size: 5 if minority count is below 100, otherwise 30.
- Classifiers:
  - kNN, SVM with linear kernel, and logistic regression for the main datasets.
  - Credit fraud uses logistic regression and random forest.
  - Small datasets use only kNN and logistic regression.
- kNN classifier settings:
  - 10 neighbors if minority count is below 100.
  - 30 neighbors otherwise.
  - 5 neighbors for arrhythmia, abalone-19, ar1, and ar3.
- Metrics: balanced accuracy and F1-score.
- Selection rule: for each dataset, select the classifier that gives the highest average F1-score over all oversampling methods and baseline, then report balanced accuracy/F1 for that classifier.

### Published Reference Table

Values are balanced accuracy / F1-score:

| Dataset | ML | Baseline | SMOTE | Bl-1 | Bl-2 | SVM | ADASYN | LoRAS |
|---|---|---|---|---|---|---|---|---|
| Abalone19 | knn | .534/.000 | .644/.054 | .552/.044 | .552/.044 | .556/.045 | .571/.055 | .675/.059 |
| Arrythmia | lr | .679/.370 | .666/.345 | .672/.352 | .709/.307 | .679/.350 | .667/.362 | .694/.380 |
| Isolet | lr | .900/.826 | .898/.806 | .899/.802 | .906/.693 | .911/.799 | .898/.806 | .904/.809 |
| Letter-img | knn | .927/.915 | .988/.781 | .984/.768 | .977/.687 | .986/.724 | .985/.732 | .989/.833 |
| Mammography | knn | .703/.549 | .911/.413 | .909/.414 | .899/.326 | .909/.467 | .905/.353 | .896/.511 |
| Scene | lr | .551/.168 | .616/.222 | .619/.230 | .620/.223 | .616/.235 | .620/.224 | .616/.226 |
| Ozone_level | lr | .517/.062 | .800/.190 | .777/.212 | .781/.183 | .738/.215 | .803/.192 | .809/.207 |
| Webpage | knn | .805/.711 | .906/.267 | .901/.274 | .903/.287 | .904/.267 | .903/.264 | .923/.613 |
| Wine-quality | lr | .517/.067 | .718/.179 | .715/.182 | .711/.171 | .712/.216 | .721/.180 | .734/.197 |
| Yeast-ml8 | knn | .500/.000 | .558/.152 | .561/.153 | .563/.153 | .572/.158 | .558/.151 | .559/.152 |
| Yeast-me2 | knn | .523/.074 | .834/.331 | .797/.373 | .790/.304 | .785/.388 | .825/.315 | .842/.354 |
| Credit fraud | rf | .669/.775 | .922/.359 | .919/.645 | .919/.556 | .913/.741 | .923/.350 | .904/.820 |
| ar1 | knn | .340/.071 | .561/.306 | .549/.298 | .594/.338 | .550/.324 | .583/.320 | .563/.349 |
| ar3 | rf | .634/.259 | .810/.531 | .809/.584 | .819/.582 | .755/.479 | .781/.457 | .823/.563 |
| Average | - | .636/.338 | .775/.352 | .764/.380 | .771/.346 | .759/.386 | .777/.340 | .783/.433 |
| Average rank | - | 6.53/4.64 | 3.57/4.75 | 4.35/3.46 | 3.39/5.10 | 4.07/3.17 | 3.50/4.71 | 2.57/2.14 |

### Comparable Number To Produce

Append MIMIC columns to the same table:

| Dataset | ML | MIMIC-identity | MIMIC-factorised | MIMIC-factorised-private |
|---|---|---|---|---|
| Abalone19 | knn | | | |
| Arrythmia | lr | | | |
| Isolet | lr | | | |
| Letter-img | knn | | | |
| Mammography | knn | | | |
| Scene | lr | | | |
| Ozone_level | lr | | | |
| Webpage | knn | | | |
| Wine-quality | lr | | | |
| Yeast-ml8 | knn | | | |
| Yeast-me2 | knn | | | |
| Credit fraud | rf | | | |
| ar1 | knn | | | |
| ar3 | rf | | | |

Also reproduce LoRAS Table 5 groups:

| Group | Published LoRAS balanced accuracy/F1 | MIMIC-factorised balanced accuracy/F1 |
|---|---|---|
| Highly imbalanced datasets | .846/.449 | |
| High dimensional datasets | .739/.436 | |

## Question 5: Does MIMIC's Mixed-Type Decoder Add Value Beyond Numeric Oversampling?

### Competitor Papers

- SMOTE paper section on Adult and SMOTE-NC.
- Geometric SMOTE nominal/continuous extensions, if added later.

### Replication Target

Use Adult as the first mixed-type benchmark because the original SMOTE paper explicitly separates continuous-only SMOTE from SMOTE-NC. MIMIC should be evaluated in both ways:

- Numeric-only MIMIC: train on continuous Adult features only.
- Mixed-type MIMIC: train on the full Adult table with categorical features retained.

### Dataset Configuration

| Dataset | Samples | Minority | Feature setup |
|---|---:|---:|---|
| Adult | 48842 | 11687 | 6 continuous and 8 nominal features in the SMOTE paper. |

### Protocol To Mirror

- Use the SMOTE paper's ROC/AUC framing where possible.
- Also report modern metrics: balanced accuracy, F1-score, AUROC, AUPRC.
- Use the same train/test folds for numeric-only and mixed-type MIMIC.

### Comparable Number To Produce

| Adult setup | Generator | Classifier | AUC | Balanced accuracy | F1 | AUPRC |
|---|---|---|---:|---:|---:|---:|
| Continuous only | MIMIC-identity | tree | | | | |
| Continuous only | MIMIC-factorised | tree | | | | |
| Mixed numeric/categorical | MIMIC-factorised | tree | | | | |

## Cross-Cutting MIMIC Ablations

Run these only after the paper-aligned baseline rows are working:

- `MIMIC-identity`: feature-space generation baseline.
- `MIMIC-direct`: learned embedding with deterministic decoding.
- `MIMIC-factorised`: learned embedding with stochastic feature-wise decoding.
- `MIMIC-factorised-private`: factorised decoding plus nearest-neighbor ambiguity filtering.
- Policy: SMOTE-style interpolation vs mutual-neighbor displacement.
- Capacity: `0.0`, `0.25`, `0.5` for feasibility/quality trade-off.

## Reproducibility Checklist

- Store exact dataset source, version, class mapping, and preprocessing choices.
- Store every train/test fold index so MIMIC runs are repeatable.
- Save the fitted MIMIC model with `MIMIC.save()` for each dataset/fold where runtime is high.
- Save generated rows and trace records for every MIMIC run.
- Report when a paper's exact classifier is substituted by a modern approximate classifier.
- Keep published competitor numbers in separate immutable reference tables.
- Keep MIMIC outputs in generated experiment artifacts, not in `manuscript/`.

## Sources Checked

- Chawla et al. SMOTE paper: dataset distribution, 10-fold ROC/AUC protocol, K=5 synthetic generation, C4.5/RIPPER/Naive Bayes comparisons.
- He et al. ADASYN paper: five datasets, decision-tree protocol, 100 half-split runs, exact Table II metrics.
- Douzas and Bacao Geometric SMOTE paper: 13 UCI datasets, GBC/LR classifiers, fivefold CV, F/G/AUC metrics, published baseline methods.
- Bej et al. LoRAS paper: 14 datasets, five repeated stratified CV protocol, classifier choices, balanced accuracy/F1 Table 4 and group Table 5.
