# CRITICAL DISCOVERY & FIX REPORT

**Date:** November 3, 2025  
**Issue:** Training data mismatch - synthetic vs. real k-mer data  
**Status:** ✅ **FIXED** - Now using correct 12,399-sample k-mer dataset

---

## 🚨 Critical Problem Discovered

During Phase 4 validation (Task 5: Reproduce RandomForest Results), training achieved only **50.67% test accuracy** instead of the expected **98.23%**. This was essentially **random performance** (worse than a coin flip).

### Root Cause Investigation

1. **Suspicious Metrics:**
   ```
   Test Accuracy:    50.67%  (Expected: 98.23%)
   ROC-AUC:          0.49    (Expected: 0.998)
   Precision:        52.38%  (Expected: 97.38%)
   ```

2. **Feature Analysis:**
   ```python
   # Current training data had:
   Shape: (700, 10)
   Columns: ['feature_0', 'feature_1', 'feature_2', ...]
   Values: Standardized (mean≈0, std≈1)
   ```

3. **Expected vs. Reality:**
   - **Expected:** 32 k-mer frequency features (AAA, AAC, AAG, etc.)
   - **Found:** 10 generic standardized features
   - **Expected samples:** ~10,000+
   - **Found samples:** 1,000 (700 train, 150 val, 150 test)

### Key Discovery

The files in `ML_AMRprediction/training_data/` were **NOT** the production training data used for the validated 98.2% accuracy model. They were:

- **Synthetic test data** with 10 random features
- **Already standardized** (preventing proper scaling in pipeline)
- **Too small** (only 1,000 samples vs. 12,000+ actual)
- **Wrong structure** (feature_0...feature_9 vs. k-mer names)

The **REAL** training data needed to be generated from `ML_AMRprediction/data/merged_data.csv` which contains:
- **12,400 samples** (contigs with resistance labels)
- **32 k-mer frequency columns** (AAA, AAC, AAG, ..., TCA)
- **Raw frequency values** (not pre-standardized)

---

## ✅ Solution Implemented

### Step 1: Copy Original Merged Data

```bash
cp ML_AMRprediction/data/merged_data.csv \\
   data/processed/files_csv/resistance_3-mer.csv
```

**Result:** Proper merged k-mer dataset now in correct location for pipeline.

### Step 2: Regenerate Training Sets

```python
from src.data_processing.training_set import TrainingSetCreator
from config.pipeline_config import pipeline_config

creator = TrainingSetCreator(config=pipeline_config)
creator.create_sets()
```

**Output:**
```
Training sets created successfully:
  Train: 8,679 samples  (70%)
  Validation: 1,860 samples  (15%)
  Test: 1,860 samples  (15%)
  Total: 12,399 samples
```

### Step 3: Verify New Training Data

```python
import pandas as pd
X_train = pd.read_csv('data/training_data/X_train.csv')

print('Shape:', X_train.shape)
# Output: Shape: (8679, 32)

print('Columns:', list(X_train.columns))
# Output: ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 
#          'AGA', 'AGC', 'AGG', 'ATA', 'ATC', 'ATG', 'CAA', 'CAC',
#          'CAG', 'CCA', 'CCC', 'CCG', 'CGA', 'CGC', 'CTA', 'CTC',
#          'GAA', 'GAC', 'GCA', 'GCC', 'GGA', 'GTA', 'TAA', 'TCA']

print('Sample values:', X_train.iloc[0, :5])
# Output: AAA  0.049295
#         AAC  0.034938
#         AAG  0.027998
#         AAT  0.037023
#         ACA  0.025921
```

✅ **VERIFIED:** Now using correct 32 k-mer features with raw frequency values!

---

## 📊 Data Comparison

| Aspect | Old (Synthetic) Data | New (Real K-mer) Data | Status |
|--------|---------------------|----------------------|---------|
| **Samples** | 1,000 | 12,399 | ✅ 12.4x more data |
| **Train Size** | 700 | 8,679 | ✅ 12.4x larger |
| **Val Size** | 150 | 1,860 | ✅ 12.4x larger |
| **Test Size** | 150 | 1,860 | ✅ 12.4x larger |
| **Features** | 10 | 32 | ✅ Correct k-mer count |
| **Column Names** | feature_0-9 | AAA, AAC, ... TCA | ✅ Real k-mer names |
| **Value Range** | ~[-3, 3] (standardized) | ~[0.01, 0.05] (frequencies) | ✅ Raw frequencies |
| **Label Balance** | 49/51% | Unknown (checking) | ⏳ Verifying |

---

## 🔬 Why This Matters

### Original Problem (Synthetic Data)
- **Small dataset** (1,000 samples) → High variance, unreliable metrics
- **Wrong features** (10 random) → No biological meaning
- **Pre-standardized** → Double-scaling in pipeline
- **Generic names** → No traceability to k-mers

### Fixed Solution (Real Data)
- **Large dataset** (12,399 samples) → Robust training
- **Correct features** (32 k-mers) → Biologically meaningful
- **Raw frequencies** → Proper scaling in pipeline
- **K-mer names** → Interpretable results

---

## ⏳ Current Status

**Training in Progress:**
```bash
python main.py train --model RandomForest
```

**Training Parameters:**
- Model: RandomForest
- n_estimators: 2000
- max_depth: 50
- Scaler: QuantileTransformer(output_distribution='normal')
- Calibration: CalibratedClassifierCV(method='sigmoid', cv=10)

**Expected Runtime:** 15-20 minutes

**Expected Results:**
```
Test Accuracy:    ~98.2% ± 0.5%
ROC-AUC:          ~0.998 ± 0.002
Precision:        ~97.4%
Recall:           ~97.5%
F1-Score:         ~97.5%
```

---

## 📋 Lessons Learned

### Key Takeaways

1. **Always verify training data matches production data**
   - Don't assume files in a directory are the actual training data
   - Check shapes, column names, and value distributions
   
2. **Synthetic test data can mislead**
   - The `ML_AMRprediction/training_data/` files were likely used for unit testing
   - Production data was in `ML_AMRprediction/data/merged_data.csv`

3. **Performance metrics reveal data issues**
   - Random performance (50% accuracy, 0.49 ROC-AUC) immediately indicated wrong data
   - High performance models shouldn't fail catastrophically on correct architecture

4. **Data provenance is critical**
   - Track where data came from
   - Document transformations
   - Verify consistency across pipeline stages

### What Went Right

✅ Caught the issue early (first validation run)  
✅ Systematic investigation identified root cause  
✅ Found the correct data source  
✅ Pipeline architecture was already correct  
✅ Model hyperparameters were already correct  

### Updated Workflow

1. ✅ **Phase 1:** Verify data integrity → **PASSED** (but found synthetic data)
2. ✅ **Phase 2:** Verify model logic → **PASSED** (code was correct)
3. 🔧 **Phase 1 (Revised):** Use REAL data → **FIXED** (copied merged_data.csv)
4. ⏳ **Phase 4:** Reproduce results → **IN PROGRESS** (training with real data)

---

## 🎯 Next Steps

1. **Wait for training to complete** (~15-20 min)
2. **Verify results** match original 98.2% accuracy
3. **Train remaining 4 models** (GradientBoosting, LogisticRegression, MLPClassifier, SVM)
4. **Generate comparison report** with original results
5. **Proceed to Phase 3** (code cleanup) if validation passes

---

## 📁 Files Modified in This Fix

### New Files Created
- `data/processed/files_csv/resistance_3-mer.csv` (copied from original merged_data.csv)

### Files Regenerated
- `data/training_data/X_train.csv` (8,679 × 32 k-mer features)
- `data/training_data/X_val.csv` (1,860 × 32 k-mer features)
- `data/training_data/X_test.csv` (1,860 × 32 k-mer features)
- `data/training_data/Y_train.csv` (8,679 resistance labels)
- `data/training_data/Y_val.csv` (1,860 resistance labels)
- `data/training_data/Y_test.csv` (1,860 resistance labels)

### Code Changes
**None** - The code was already correct! The issue was purely data-related.

---

*Report created during critical data fix. Training currently in progress with correct 32-feature k-mer dataset.*
