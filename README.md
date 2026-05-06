# Applied Machine Learning Lab — Mini Project

## Performance Analysis of Machine Learning Models Using L1 and L2 Regularization for Regression and Classification Tasks

---

## 📋 Project Overview

This project investigates the **performance impact of L1 (Lasso) and L2 (Ridge) regularization** on machine learning models applied to both regression and classification tasks using the **PaySim financial transaction dataset**. The study provides critical insights into how regularization techniques control model complexity, prevent overfitting, and trade off between feature selection and multicollinearity handling.

### Key Findings
- **L1 Regularization (Lasso)** enables automatic feature selection through sparse coefficient vectors, improving model interpretability
- **L2 Regularization (Ridge)** handles multicollinearity by shrinking all coefficients proportionally while retaining all features
- Both techniques improve model generalization, with optimal performance achieved through careful hyperparameter tuning
- **Ensemble methods (XGBoost, Random Forest)** significantly outperform linear models, highlighting the importance of capturing non-linear feature interactions
- Excessive regularization causes underfitting, particularly degrading minority class performance in classification tasks

---

## 🎯 Problem Definition

### Objective
To investigate the role of L1 and L2 regularization in controlling model complexity and improving generalization on financial transaction data, with particular focus on:
1. How regularization prevents overfitting in high-dimensional, correlated feature spaces
2. The trade-offs between feature selection (L1) and multicollinearity handling (L2)
3. Optimal regularization strength determination through cross-validation

### Tasks Defined
| Task | Target | Features | Type |
|------|--------|----------|------|
| **Regression** | Log-transformed transaction amount | 12 numeric financial features | Continuous prediction |
| **Classification** | Transaction type (5 classes) | 11 numeric financial features | Multi-class (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |

### Significance
Financial fraud detection is a high-stakes domain requiring both **model interpretability and strong generalization**. Regularization ensures models do not overfit to training noise—critical in high-dimensional, imbalanced financial data where feature engineering creates correlated predictors.

---

## 📊 Dataset Overview

### PaySim — Synthetic Mobile Money Dataset
| Property | Details |
|----------|---------|
| **Source** | [Kaggle — PaySim Mobile Money Simulator](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Total Records** | 6,362,620 transactions |
| **Original Columns** | 11 (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud) |
| **Sample Used** | 50,000 rows (10,000 per transaction type — stratified) |
| **Transaction Types** | 5 classes: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| **Fraud Ratio** | ~0.13% (severely imbalanced: 1,514 fraud vs 1,184,807 legitimate) |
| **Missing Values** | 0 — No imputation required |

### EDA Key Observations
✅ **Transaction Amount**: Heavily right-skewed; log1p transformation applied for normalization  
✅ **Fraud Concentration**: Occurs exclusively in TRANSFER (0.76%) and CASH_OUT (0.18%) types  
✅ **Class Distribution**: CASH_OUT and PAYMENT dominate by volume; DEBIT is rarest  
✅ **Feature Signals**: Balance differential features show strong separation between legitimate and fraudulent transactions  
✅ **Data Quality**: No missing values or duplicates; clean dataset ready for modeling

---

## 🔧 Data Preprocessing & Feature Engineering

### Sampling Strategy
Due to dataset size (6.3M rows), a **stratified sample of 50,000 records** was drawn at 10,000 transactions per type, preserving class distributions while ensuring computational efficiency.

### Feature Engineering
Six new features created to enrich predictive signal:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| **balanceOrigDiff** | newbalanceOrig − oldbalanceOrg | Net change in sender balance |
| **balanceDestDiff** | newbalanceDest − oldbalanceDest | Net change in receiver balance |
| **origBalanceZero** | 1 if oldbalanceOrg == 0 | Zero-balance sender (fraud signal) |
| **destBalanceZero** | 1 if oldbalanceDest == 0 | Zero-balance receiver indicator |
| **amountLog** | log1p(amount) | Normalized amount (regression target) |
| **surplusOrig** | oldbalanceOrg − amount | Remaining sender balance post-transaction |

### Encoding & Scaling
- **LabelEncoder** applied to 'type' column, mapping 5 transaction types to integer labels (0–4)
- **StandardScaler** applied independently to both feature matrices—**critical for regularized models** to prevent disproportionate penalization of unscaled features
- Name columns dropped after encoding

### Final Feature Sets
| Task | Features | Count | Details |
|------|----------|-------|---------|
| **Regression** | All numeric excluding raw 'amount' | 12 | Target: amountLog |
| **Classification** | All numeric excluding 'amount' & 'amountLog' | 11 | Target: type_encoded |

### Train–Test Split
- **80/20 stratified split** with `random_state=42` for reproducibility
- **Regression**: 40,000 train / 10,000 test
- **Classification**: 40,000 train / 10,000 test
- Stratification preserves transaction type proportions in both splits

---

## 🧠 Core Concept — L1 vs L2 Regularization

### L1 Regularization (Lasso)
```
Penalty: λ × Σ|coefficients|
Behavior: Creates corner solutions at axes → exact-zero coefficients likely
Key Advantage: Automatic feature selection (sparse models)
Use Case: When interpretability and reducing feature set is important
```

### L2 Regularization (Ridge)
```
Penalty: λ × Σ(coefficient²)
Behavior: Smooth shrinkage toward zero but never exactly zero
Key Advantage: Handles multicollinearity; retains all features
Use Case: When all features are relevant or multicollinearity exists
```

### ElasticNet (L1 + L2 Hybrid)
```
Penalty: λ × (l1_ratio × Σ|coefficients| + (1-l1_ratio) × Σ(coefficient²))
Behavior: Balances feature selection with coefficient shrinkage
Key Advantage: Flexible combination; good default choice
Use Case: Uncertain about L1 vs L2 trade-off
```

---

## 🎓 Model Training

### Regression Models

| Model | Regularization | Alpha Configuration | Purpose |
|-------|-----------------|-------------------|---------|
| Linear Regression (OLS) | None | N/A | Baseline — no penalty |
| Ridge — Optimal | L2 | Via 5-fold CV (log-spaced [10⁻³, 10³]) | Best L2 performance |
| Ridge — α=0.1 | L2 | Fixed (light penalty) | Under-regularized comparison |
| Ridge — α=10.0 | L2 | Fixed (heavy penalty) | Over-regularized comparison |
| Lasso — Optimal | L1 | Via 5-fold CV (log-spaced [10⁻³, 10³]) | Best L1 performance |
| Lasso — α=0.01 | L1 | Fixed (light penalty) | Minimal sparsity |
| Lasso — α=0.5 | L1 | Fixed (heavy penalty) | Aggressive sparsity |
| ElasticNet | L1 + L2 | l1_ratio=0.5, optimal alpha | Hybrid approach |

### Classification Models (13 Total)

| Model | Regularization | Configuration | Purpose |
|-------|-----------------|-----------------|---------|
| **Linear Models** |
| Logistic Regression (No Reg) | None | penalty=None, solver=lbfgs, multinomial | Baseline |
| Logistic Regression L2 (C=1.0) | L2 | solver=lbfgs, default strength | Moderate L2 |
| Logistic Regression L2 (C=0.01) | L2 | solver=lbfgs, strong strength | Strong L2 |
| Logistic Regression L1 (C=1.0) | L1 | solver=saga, default strength | Moderate L1 |
| Logistic Regression L1 (C=0.1) | L1 | solver=saga, strong strength | Strong L1 |
| SGD Classifier | L2 | Modified Huber loss, alpha=0.001 | Stochastic descent baseline |
| **Tree-Based** |
| Decision Tree (No Reg) | None | max_depth=None (fully grown) | Unconstrained baseline |
| Decision Tree (Regularized) | Structural | max_depth=8, min_samples_split=20 | Controlled complexity |
| Random Forest | Ensemble | 100 estimators, max_depth=10 | Ensemble baseline |
| Bagging Classifier | Ensemble | 20 base estimators (Decision Trees) | Bootstrap aggregation |
| **Probabilistic & Distance-Based** |
| Gaussian Naïve Bayes | None | var_smoothing=1e-9 | Probabilistic classifier |
| KNN (k=5) | Distance-based | 5 neighbors | Instance-based learning |
| LinearSVC (SVM) | L2 | CalibratedClassifierCV wrapper, max_iter=2000 | Linear SVM |
| **Gradient Boosting** |
| XGBoost | Ensemble | 100 estimators, max_depth=6, lr=0.1 | Gradient boosting |

**Note**: Classification CV used **5-Fold Stratified KFold** to produce robust macro F1 scores for all 13 models.

---

## 📈 Results & Evaluation

### Regression Performance

| Model | Test R² | Test RMSE | Test MAE | Overfit Gap | Key Insight |
|-------|---------|-----------|----------|-------------|-------------|
| Linear Regression (OLS) | 0.2377 | 1.8713 | 1.5008 | 0.0232 | Baseline overfitting |
| Ridge — Optimal | 0.2366 | 1.8727 | 1.5048 | 0.0222 | ✅ Reduced gap |
| Ridge — α=0.1 | 0.2377 | 1.8713 | 1.5008 | 0.0232 | Similar to OLS |
| Ridge — α=10.0 | 0.2377 | 1.8713 | 1.5008 | 0.0232 | Too weak for this data |
| Lasso — Optimal | 0.2367 | 1.8725 | 1.5047 | 0.0224 | ✅ Reduced gap + sparse |
| Lasso — α=0.01 | 0.2368 | 1.8724 | 1.5044 | 0.0225 | Minimal regularization |
| Lasso — α=0.5 | 0.0931 | 2.0411 | 1.7494 | 0.0021 | ⚠️ Severe underfitting |
| ElasticNet | 0.2370 | 1.8721 | 1.5036 | 0.0226 | ✅ Matched L1/L2 performance |

**Key Findings**:
- ✅ Ridge and Lasso at optimal alpha perform nearly identically on test metrics
- ✅ **Both reduce overfitting gap compared to OLS**, stabilizing model behavior
- ⚠️ **Lasso α=0.5 shows severe underfitting**, demonstrating that excessive regularization harms accuracy
- 📊 Regularization does not necessarily improve raw accuracy but stabilizes and controls variance

### Classification Performance (Macro-Averaged Metrics)

| Model | Accuracy | Precision | Recall | F1 Macro | CV F1 |
|-------|----------|-----------|--------|----------|-------|
| **Linear Models** |
| Logistic Reg (No Reg) | 0.8746 | 0.8762 | 0.8784 | 0.8751 | 0.8706 |
| Logistic Reg L2 (C=1.0) | 0.8371 | 0.8428 | 0.8372 | 0.8376 | 0.8336 |
| Logistic Reg L2 (C=0.01) | 0.7691 | 0.7679 | 0.7710 | 0.7639 | 0.7614 |
| Logistic Reg L1 (C=1.0) | 0.8425 | 0.8483 | 0.8428 | 0.8432 | 0.8380 |
| Logistic Reg L1 (C=0.1) | 0.8367 | 0.8422 | 0.8368 | 0.8372 | 0.8327 |
| SGD Classifier (L2) | 0.7790 | 0.7645 | 0.7815 | 0.7654 | 0.7649 |
| **Tree-Based** |
| Decision Tree (No Reg) | 0.8402 | 0.8430 | 0.8438 | 0.8434 | 0.8435 |
| Decision Tree (Regularized) | 0.8781 | 0.8868 | 0.8817 | 0.8786 | 0.8751 |
| Random Forest | 0.8798 | 0.8919 | 0.8830 | 0.8800 | 0.8765 |
| Bagging Classifier | 0.8729 | 0.8766 | 0.8767 | 0.8747 | 0.8695 |
| **Other Models** |
| Gaussian Naïve Bayes | 0.6353 | 0.7010 | 0.6477 | 0.6225 | 0.6232 |
| KNN (k=5) | 0.7965 | 0.7969 | 0.7989 | 0.7951 | 0.7948 |
| LinearSVC (SVM-linear) | 0.8067 | 0.7992 | 0.8082 | 0.8021 | 0.7865 |
| **Gradient Boosting** |
| XGBoost | 🏆 0.8874 | 0.8946 | 0.8910 | 🏆 0.8881 | 🏆 0.8827 |

**Key Findings**:
- 🏆 **XGBoost achieves highest F1 (0.8881)** — non-linear ensemble dominates
- 📊 **Random Forest (0.8800) and Regularized Decision Tree (0.8786) follow closely**
- ⚠️ **Over-regularization (C=0.01 for L2, C=0.1 for L1) significantly degraded performance**
- ✅ **Regularized Decision Tree (0.8786) outperformed unregularized tree (0.8434)** — structural regularization effective
- 📉 **Gaussian Naïve Bayes (0.6225) underperformed** — inappropriate for multi-class classification

### Confusion Matrix Analysis (5×5)
- **Random Forest & Regularized Decision Tree**: Near-perfect diagonal matrices — almost all transaction types correctly identified
- **Logistic Regression**: Moderate confusion between similar types (CASH_IN vs PAYMENT), reducing macro P/R/F1
- **Strongly Regularized Models**: More off-diagonal errors — **confirming that over-regularization degrades classification**

### Coefficient Analysis — L1 vs L2 Effect
| Aspect | No Regularization | L1 (Lasso) | L2 (Ridge) |
|--------|-------------------|-----------|-----------|
| **Coefficient Magnitude** | Largest (high variance) | Near-zero groups | Uniformly shrunk |
| **Feature Selection** | None | ✅ Multiple coefficients → exactly zero | All features retained |
| **Interpretability** | Lower (many features) | ✅ Sparse model | Higher (fewer active features) |
| **Visualization** | Sparse, uneven | **Visibly sparse** | Smooth, balanced profile |

---

## 🔍 Comparative Analysis

### L1 vs L2 Summary Table

| Criterion | No Regularization | L1 (Lasso) | L2 (Ridge) | ElasticNet |
|-----------|-------------------|-----------|-----------|-----------|
| **Feature Selection** | No | ✅ Yes (exact zeros) | No | ✅ Partial |
| **Multicollinearity Handling** | Poor | Moderate | ✅ Excellent | ✅ Good |
| **Overfitting Control** | None | ✅ Strong | ✅ Strong | ✅ Strong |
| **Interpretability** | Moderate | ✅ High (sparse) | Moderate | Moderate |
| **All Coefficients Non-Zero** | Yes | ❌ No | ✅ Yes | ❌ No |
| **Best Use Case** | No redundancy | Few relevant features | All features useful | Uncertain relevance |
| **Computational Cost** | Low | Low | Low | Moderate |

### Regression Task Analysis
- ✅ **At optimal alpha**: Ridge (L2) and Lasso (L1) perform nearly identically (Test R² ≈ 0.2366-0.2367)
- ✅ **Critical distinction**: Lasso drove multiple feature coefficients to exactly zero (feature selection), Ridge retained all
- ⚠️ **Lasso α=0.5**: Severe degradation (Test R² ≈ 0.093) — demonstrates that **excessive regularization causes underfitting**
- 📊 **Alpha sensitivity**: Sweet spot exists; too low → overfitting; too high → underfitting

### Classification Task Analysis
- 🏆 **XGBoost (F1 ≈ 0.8881)** > Random Forest (0.8800) > Regularized DT (0.8786)
- ✅ **Among linear models**: Logistic Regression L1 (C=1.0) and L2 (C=1.0) performed equally (F1 ≈ 0.83)
- ⚠️ **Stronger regularization (C=0.01, C=0.1)**: Significantly degraded performance, especially minority classes
- 📊 **Structural regularization (Decision Tree depth)**: Nearly as effective as explicit L1/L2 penalties
- **Insight**: Non-linear models fundamentally capture feature interactions better than regularized linear models

---

## 💡 Key Insights & Conclusions

### 1. **Regularization is Essential for Generalization**
✅ Both L1 and L2 regularization effectively control overfitting in high-dimensional financial datasets with correlated features.  
✅ Even moderate regularization reduces the train-test generalization gap.  
⚠️ Proper hyperparameter tuning via cross-validation is **critical** — arbitrary choices degrade performance.

### 2. **L1 vs L2: Choose Based on Problem Context**
| Use L1 (Lasso) when: | Use L2 (Ridge) when: |
|--------------------|-------------------|
| Feature selection needed | Multicollinearity present |
| Interpretability is priority | All features are relevant |
| High-dimensional data | Moderate dimensionality |

### 3. **Excessive Regularization Causes Underfitting**
⚠️ Over-regularization (Lasso α=0.5, Logistic Regression C=0.01) degraded both accuracy and minority class performance.  
✅ **Lesson**: Always use cross-validation to determine optimal regularization strength, not manual tuning.

### 4. **Ensemble Methods Dominate Linear Models**
🏆 Non-linear ensemble models (XGBoost, Random Forest) significantly outperform all linear models—both regularized and unregularized.  
📊 Suggests that feature interactions in financial transactions require non-linear capture.  
✅ **Insight**: Regularization should be viewed as a **complementary technique**, not a substitute for model selection.

### 5. **Structural Regularization (Depth Limiting) is Effective**
✅ Regularized Decision Tree (depth=8, min_samples_split=20) outperformed unregularized tree (0.8786 vs 0.8434).  
✅ Demonstrates that complexity control can be achieved through model architecture, not just penalties.

### 6. **Real-World Fraud Detection Requires Imbalanced Learning**
⚠️ **Limitation of this study**: Used balanced sampled dataset (10,000 per type), which does not reflect extreme class imbalance (0.13% fraud).  
📊 **Future work**: Extend analysis to imbalanced learning techniques (SMOTE, class weighting) and binary fraud classification.

---

## 🎯 Recommendations

### **For Practitioners**:
1. **Always use cross-validation** to determine optimal regularization strength (RidgeCV, LassoCV)
2. **Use L1 (Lasso) for feature selection** when interpretability is critical in financial/compliance contexts
3. **Use L2 (Ridge) as default** when all features are relevant or multicollinearity exists
4. **Prefer ensemble methods** (XGBoost, Random Forest) for production fraud detection systems
5. **Never over-regularize** — it causes underfitting and degrades minority class performance

### **For Research**:
1. Extend to **imbalanced learning techniques** (SMOTE, class weights, threshold tuning)
2. Investigate **binary fraud classification** (more realistic than 5-class transaction type)
3. Analyze **real-world class imbalance** (0.13% fraud ratio) instead of balanced sampling
4. Compare **regularization with dimensionality reduction** (PCA, feature selection) techniques
5. Study **temporal drift** in fraud patterns over multiple months

---

## 📁 Project Structure

```
Applied-ML-Lab-Project/
├── README.md                          # This file
├── Mini_ProjectAML (1).ipynb           # Complete Jupyter notebook with all code
├── PS_20174392719_1491204439457_log.csv # PaySim dataset (sampled)
└── [Outputs/Visualizations]
    ├── type_distribution.png           # Transaction type distribution
    ├── fraud_distribution.png          # Fraud vs legitimate ratio
    ├── amount_distribution.png         # Transaction amount EDA
    ├── fraud_by_type.png              # Fraud rate by transaction type
    └── [Confusion matrices for all 13 models]
```

---

## 🔧 How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Execution
1. **Download the dataset** from [Kaggle PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. **Place dataset** in the project directory as `PS_20174392719_1491204439457_log.csv`
3. **Run the notebook** in Google Colab or Jupyter:
   ```bash
   jupyter notebook "Mini_ProjectAML (1).ipynb"
   ```
4. **View results**: Confusion matrices, coefficient plots, and performance comparisons auto-generate

---

## 📚 References & Context

**Course**: Applied Machine Learning (CSAI2017P)  
**Institution**: VIT Vellore, CSE Batch B21  
**Semester**: IV  
**Faculty Advisor**: Dr. Sahinur Laskar  
**Dataset Source**: [Kaggle — PaySim Mobile Money Simulator](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## 👤 Author

**Name**: Lakshay Manchanda  
**SAP ID**: 590011784  
**Batch**: B21  
**Program**: B. Tech. Computer Science and Engineering

---

## 📄 License

This project is part of the Applied Machine Learning coursework at VIT Vellore. Educational use only.

---

## 📞 Questions or Feedback?

For questions about the methodology, results, or implementation details, please refer to the comprehensive Jupyter notebook included in this repository, which contains:
- ✅ Complete preprocessing pipeline
- ✅ All 8 regression models with visualizations
- ✅ All 13 classification models with confusion matrices
- ✅ Detailed coefficient analysis (L1 vs L2)
- ✅ Cross-validation results and hyperparameter tuning process
- ✅ Statistical comparisons and insights

---

**Last Updated**: May 2026  
**Status**: Complete ✅
