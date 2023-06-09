

compare the 2 ensembling methods:
- average all probs across all models, then search for the best thresholds (my current approach)
- tune each threshold to be the best for each model, then do majority vote (your approach), maybe additionally tune the vote threshold?