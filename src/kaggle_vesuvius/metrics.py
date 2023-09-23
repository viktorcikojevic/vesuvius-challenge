import torch
import torch.nn as nn

# ref - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
def fbeta_score(preds, targets, threshold, apply_sigmoid=False, beta=0.5, smooth=1e-5):
    if apply_sigmoid:
        preds = torch.sigmoid(preds)
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()
    
    ctp = preds_t[targets==1].sum()
    cfp = preds_t[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def compute_fbeta_score(p):
    predictions, labels = p
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    
    thresholds = np.linspace(0., 1.0, 0.01)
    
    # compute fbeta score for each threshold
    fbeta = []
    for threshold in thresholds:
        fbeta.append(fbeta_score(predictions, labels, threshold=threshold, apply_sigmoid=True))
    
    # return the best fbeta score and the corresponding threshold
    best_fbeta = np.max(fbeta)
    best_threshold = thresholds[np.argmax(fbeta)]
    
    return {
        'fbeta': best_fbeta,
        'threshold': best_threshold
    }