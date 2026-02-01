# Model Training

## Using standard scaler

### With smote and without scale_pos_weight (best combination)

- XGBoost Validation ROC-AUC: 0.7915
- Accuracy: 71.75035493502938
- Precision: 70.95138536089726
- Recall: 83.22408379363296
- F1: 76.59926800198294

### without smote and with scale_pos_weight

- XGBoost Validation ROC-AUC: 0.7236
- Accuracy: 65.53214285714286
- Precision: 77.36341442882397
- Recall: 63.18974112146319
- F1: 69.56192638849465

## Using Robust scaler

### With smote and without scale_pos_weight

- XGBoost Validation ROC-AUC: 0.7915
- Accuracy: 71.73252860171515
- Precision: 70.95367527718571
- Recall: 83.16334716142192
- F1: 76.57486546375435

### without smote and with scale_pos_weight

- XGBoost Validation ROC-AUC: 0.7236
- Accuracy: 65.53214285714286
- Precision: 77.36341442882397
- Recall: 63.18974112146319
- F1: 69.56192638849465

## Using MinMax scaler

### With smote and without scale_pos_weight

- XGBoost Validation ROC-AUC: 0.7902
- Accuracy: 71.59946775661962
- Precision: 70.98589886147822
- Recall: 82.6682863101923
- F1: 76.38298435556025

### without smote and with scale_pos_weight

- XGBoost Validation ROC-AUC: 0.7236
- Accuracy: 65.53214285714286
- Precision: 77.36341442882397
- Recall: 63.18974112146319
- F1: 69.56192638849465
