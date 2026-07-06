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

### Published Table 2 Reference Values

These values are extracted from Table 2 of Douzas and Bacao. They are the direct comparison targets for Q3. Fill the MIMIC columns from experiment outputs, then compute rank against the six published methods.

| Dataset | Classifier | Metric | No oversampling | SMOTE | Borderline SMOTE1 | Borderline SMOTE2 | ADASYN | Geometric SMOTE | Published best method | Published best value | MIMIC-identity | MIMIC-factorised |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Breast | GBC | F | 0.700 | 0.715 | 0.727 | 0.705 | 0.673 | 0.722 | Borderline SMOTE1 | 0.727 | | |
| Breast | GBC | G | 0.770 | 0.780 | 0.795 | 0.774 | 0.748 | 0.789 | Borderline SMOTE1 | 0.795 | | |
| Breast | GBC | AUC | 0.867 | 0.867 | 0.867 | 0.869 | 0.858 | 0.874 | Geometric SMOTE | 0.874 | | |
| Breast | LR | F | 0.687 | 0.732 | 0.747 | 0.734 | 0.730 | 0.748 | Geometric SMOTE | 0.748 | | |
| Breast | LR | G | 0.752 | 0.799 | 0.812 | 0.803 | 0.798 | 0.812 | Borderline SMOTE1 / Geometric SMOTE | 0.812 | | |
| Breast | LR | AUC | 0.884 | 0.890 | 0.888 | 0.889 | 0.889 | 0.896 | Geometric SMOTE | 0.896 | | |
| Ecoli | GBC | F | 0.781 | 0.777 | 0.773 | 0.770 | 0.662 | 0.821 | Geometric SMOTE | 0.821 | | |
| Ecoli | GBC | G | 0.868 | 0.872 | 0.868 | 0.862 | 0.861 | 0.904 | Geometric SMOTE | 0.904 | | |
| Ecoli | GBC | AUC | 0.948 | 0.945 | 0.945 | 0.940 | 0.937 | 0.960 | Geometric SMOTE | 0.960 | | |
| Ecoli | LR | F | 0.249 | 0.716 | 0.682 | 0.641 | 0.473 | 0.718 | Geometric SMOTE | 0.718 | | |
| Ecoli | LR | G | 0.378 | 0.894 | 0.885 | 0.870 | 0.759 | 0.900 | Geometric SMOTE | 0.900 | | |
| Ecoli | LR | AUC | 0.934 | 0.935 | 0.929 | 0.924 | 0.896 | 0.936 | Geometric SMOTE | 0.936 | | |
| Eucalyptus | GBC | F | 0.551 | 0.526 | 0.540 | 0.535 | 0.506 | 0.549 | No oversampling | 0.551 | | |
| Eucalyptus | GBC | G | 0.682 | 0.699 | 0.703 | 0.714 | 0.688 | 0.730 | Geometric SMOTE | 0.730 | | |
| Eucalyptus | GBC | AUC | 0.878 | 0.869 | 0.872 | 0.870 | 0.841 | 0.874 | No oversampling | 0.878 | | |
| Eucalyptus | LR | F | 0.207 | 0.529 | 0.530 | 0.513 | 0.468 | 0.540 | Geometric SMOTE | 0.540 | | |
| Eucalyptus | LR | G | 0.399 | 0.790 | 0.787 | 0.785 | 0.757 | 0.797 | Geometric SMOTE | 0.797 | | |
| Eucalyptus | LR | AUC | 0.857 | 0.886 | 0.883 | 0.879 | 0.870 | 0.890 | Geometric SMOTE | 0.890 | | |
| Glass | GBC | F | 0.779 | 0.774 | 0.774 | 0.774 | 0.758 | 0.792 | Geometric SMOTE | 0.792 | | |
| Glass | GBC | G | 0.831 | 0.830 | 0.833 | 0.836 | 0.822 | 0.846 | Geometric SMOTE | 0.846 | | |
| Glass | GBC | AUC | 0.924 | 0.920 | 0.918 | 0.919 | 0.911 | 0.929 | Geometric SMOTE | 0.929 | | |
| Glass | LR | F | 0.505 | 0.657 | 0.648 | 0.651 | 0.650 | 0.661 | Geometric SMOTE | 0.661 | | |
| Glass | LR | G | 0.613 | 0.733 | 0.715 | 0.714 | 0.719 | 0.734 | Geometric SMOTE | 0.734 | | |
| Glass | LR | AUC | 0.824 | 0.825 | 0.814 | 0.818 | 0.819 | 0.825 | SMOTE / Geometric SMOTE | 0.825 | | |
| Haberman | GBC | F | 0.339 | 0.393 | 0.388 | 0.385 | 0.376 | 0.426 | Geometric SMOTE | 0.426 | | |
| Haberman | GBC | G | 0.505 | 0.558 | 0.551 | 0.549 | 0.542 | 0.584 | Geometric SMOTE | 0.584 | | |
| Haberman | GBC | AUC | 0.627 | 0.637 | 0.641 | 0.636 | 0.619 | 0.668 | Geometric SMOTE | 0.668 | | |
| Haberman | LR | F | 0.239 | 0.477 | 0.485 | 0.481 | 0.418 | 0.484 | Borderline SMOTE1 | 0.485 | | |
| Haberman | LR | G | 0.376 | 0.622 | 0.632 | 0.628 | 0.578 | 0.628 | Borderline SMOTE1 | 0.632 | | |
| Haberman | LR | AUC | 0.686 | 0.693 | 0.688 | 0.684 | 0.650 | 0.694 | Geometric SMOTE | 0.694 | | |
| Heart | GBC | F | 0.745 | 0.750 | 0.753 | 0.754 | 0.739 | 0.761 | Geometric SMOTE | 0.761 | | |
| Heart | GBC | G | 0.771 | 0.775 | 0.776 | 0.779 | 0.765 | 0.781 | Geometric SMOTE | 0.781 | | |
| Heart | GBC | AUC | 0.859 | 0.862 | 0.864 | 0.864 | 0.856 | 0.868 | Geometric SMOTE | 0.868 | | |
| Heart | LR | F | 0.822 | 0.822 | 0.823 | 0.824 | 0.820 | 0.826 | Geometric SMOTE | 0.826 | | |
| Heart | LR | G | 0.840 | 0.840 | 0.841 | 0.841 | 0.837 | 0.843 | Geometric SMOTE | 0.843 | | |
| Heart | LR | AUC | 0.902 | 0.903 | 0.901 | 0.900 | 0.901 | 0.903 | SMOTE / Geometric SMOTE | 0.903 | | |
| Iris | GBC | F | 0.923 | 0.937 | 0.915 | 0.922 | 0.911 | 0.939 | Geometric SMOTE | 0.939 | | |
| Iris | GBC | G | 0.944 | 0.954 | 0.939 | 0.945 | 0.940 | 0.956 | Geometric SMOTE | 0.956 | | |
| Iris | GBC | AUC | 0.980 | 0.985 | 0.978 | 0.984 | 0.972 | 0.988 | Geometric SMOTE | 0.988 | | |
| Iris | LR | F | 0.334 | 0.648 | 0.637 | 0.636 | 0.674 | 0.657 | ADASYN | 0.674 | | |
| Iris | LR | G | 0.453 | 0.731 | 0.715 | 0.712 | 0.752 | 0.739 | ADASYN | 0.752 | | |
| Iris | LR | AUC | 0.793 | 0.792 | 0.747 | 0.745 | 0.803 | 0.801 | ADASYN | 0.803 | | |
| Libra | GBC | F | 0.780 | 0.850 | 0.839 | 0.876 | 0.786 | 0.924 | Geometric SMOTE | 0.924 | | |
| Libra | GBC | G | 0.821 | 0.890 | 0.875 | 0.915 | 0.847 | 0.946 | Geometric SMOTE | 0.946 | | |
| Libra | GBC | AUC | 0.943 | 0.969 | 0.962 | 0.976 | 0.931 | 0.987 | Geometric SMOTE | 0.987 | | |
| Libra | LR | F | 0.316 | 0.575 | 0.512 | 0.475 | 0.631 | 0.565 | ADASYN | 0.631 | | |
| Libra | LR | G | 0.432 | 0.745 | 0.686 | 0.659 | 0.779 | 0.738 | ADASYN | 0.779 | | |
| Libra | LR | AUC | 0.737 | 0.766 | 0.733 | 0.708 | 0.776 | 0.765 | ADASYN | 0.776 | | |
| Liver | GBC | F | 0.636 | 0.630 | 0.639 | 0.644 | 0.642 | 0.657 | Geometric SMOTE | 0.657 | | |
| Liver | GBC | G | 0.688 | 0.681 | 0.687 | 0.693 | 0.686 | 0.701 | Geometric SMOTE | 0.701 | | |
| Liver | GBC | AUC | 0.753 | 0.752 | 0.755 | 0.754 | 0.756 | 0.758 | Geometric SMOTE | 0.758 | | |
| Liver | LR | F | 0.579 | 0.633 | 0.622 | 0.629 | 0.626 | 0.640 | Geometric SMOTE | 0.640 | | |
| Liver | LR | G | 0.644 | 0.669 | 0.658 | 0.661 | 0.650 | 0.669 | SMOTE / Geometric SMOTE | 0.669 | | |
| Liver | LR | AUC | 0.713 | 0.712 | 0.707 | 0.709 | 0.710 | 0.715 | Geometric SMOTE | 0.715 | | |
| Pima | GBC | F | 0.623 | 0.651 | 0.654 | 0.652 | 0.644 | 0.659 | Geometric SMOTE | 0.659 | | |
| Pima | GBC | G | 0.704 | 0.728 | 0.731 | 0.729 | 0.722 | 0.734 | Geometric SMOTE | 0.734 | | |
| Pima | GBC | AUC | 0.812 | 0.811 | 0.806 | 0.806 | 0.800 | 0.822 | Geometric SMOTE | 0.822 | | |
| Pima | LR | F | 0.617 | 0.675 | 0.675 | 0.674 | 0.676 | 0.677 | Geometric SMOTE | 0.677 | | |
| Pima | LR | G | 0.692 | 0.748 | 0.747 | 0.746 | 0.747 | 0.749 | Geometric SMOTE | 0.749 | | |
| Pima | LR | AUC | 0.825 | 0.827 | 0.825 | 0.824 | 0.824 | 0.830 | Geometric SMOTE | 0.830 | | |
| Segment | GBC | F | 0.925 | 0.927 | 0.912 | 0.887 | 0.856 | 0.934 | Geometric SMOTE | 0.934 | | |
| Segment | GBC | G | 0.941 | 0.967 | 0.961 | 0.960 | 0.956 | 0.970 | Geometric SMOTE | 0.970 | | |
| Segment | GBC | AUC | 0.996 | 0.997 | 0.995 | 0.994 | 0.990 | 0.997 | SMOTE / Geometric SMOTE | 0.997 | | |
| Segment | LR | F | 0.648 | 0.645 | 0.634 | 0.622 | 0.552 | 0.640 | No oversampling | 0.648 | | |
| Segment | LR | G | 0.749 | 0.881 | 0.881 | 0.888 | 0.850 | 0.881 | Borderline SMOTE2 | 0.888 | | |
| Segment | LR | AUC | 0.942 | 0.942 | 0.930 | 0.923 | 0.922 | 0.944 | Geometric SMOTE | 0.944 | | |
| Vehicle | GBC | F | 0.923 | 0.927 | 0.929 | 0.925 | 0.923 | 0.935 | Geometric SMOTE | 0.935 | | |
| Vehicle | GBC | G | 0.951 | 0.958 | 0.958 | 0.961 | 0.962 | 0.969 | Geometric SMOTE | 0.969 | | |
| Vehicle | GBC | AUC | 0.994 | 0.994 | 0.995 | 0.994 | 0.993 | 0.995 | Borderline SMOTE1 / Geometric SMOTE | 0.995 | | |
| Vehicle | LR | F | 0.940 | 0.941 | 0.940 | 0.895 | 0.816 | 0.940 | SMOTE | 0.941 | | |
| Vehicle | LR | G | 0.961 | 0.968 | 0.967 | 0.958 | 0.924 | 0.969 | Geometric SMOTE | 0.969 | | |
| Vehicle | LR | AUC | 0.995 | 0.995 | 0.995 | 0.992 | 0.983 | 0.995 | No oversampling / SMOTE / Borderline SMOTE1 / Geometric SMOTE | 0.995 | | |
| Wine | GBC | F | 0.911 | 0.908 | 0.912 | 0.907 | 0.875 | 0.933 | Geometric SMOTE | 0.933 | | |
| Wine | GBC | G | 0.925 | 0.923 | 0.927 | 0.922 | 0.898 | 0.943 | Geometric SMOTE | 0.943 | | |
| Wine | GBC | AUC | 0.979 | 0.978 | 0.978 | 0.975 | 0.965 | 0.986 | Geometric SMOTE | 0.986 | | |
| Wine | LR | F | 0.928 | 0.935 | 0.939 | 0.937 | 0.898 | 0.938 | Borderline SMOTE1 | 0.939 | | |
| Wine | LR | G | 0.941 | 0.947 | 0.951 | 0.950 | 0.918 | 0.951 | Borderline SMOTE1 / Geometric SMOTE | 0.951 | | |
| Wine | LR | AUC | 0.992 | 0.992 | 0.991 | 0.992 | 0.985 | 0.993 | Geometric SMOTE | 0.993 | | |

Keep the full row-level MIMIC result table in experiment outputs; the manuscript can summarize average rank against the published methods and report win/tie/loss counts against the published best value.

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
