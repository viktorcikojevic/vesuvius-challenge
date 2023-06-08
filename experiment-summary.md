Summary of the experiments.
Fold 1 means that I train on images 2 and 3, and validate on image 1. Same logic for fold 2 and 3.
I'm performing all the experiments with the 3D resnet model. 

- Losses exploration

    BCE Loss:

    - Fold 1: 0.5454 CV, 0.6 LB
        - exp-1-reproduce-training-with-augm
    - Fold 2: 0.4302 CV
        - exp-1-reproduce-training-with-augm-fold-2
    - Fold 3: 0.5982 CV
        - exp-1-reproduce-training-with-augm-fold-3

    BCE + Dice Loss:

    - Fold 1: 0.5526  CV,  **0.65 LB (with size 256)**
        - exp-1-reproduce-training-with-augm-bce-dice-loss
    - Fold 2: 0.4460 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-2
    - Fold 3: 0.6380 CV, **0.4 public LB !!!**
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-3

    Ensemble of all three gives public 0.52 LB !? 
    This means that you should take care on how you do the ensembling. 
    So Fold 3 gives bad LB result, while fold 1 gives the best possible. Could it be that images 1 and 2 are worse for training than 2 and 3? Maybe it’s best to investigate what is the LB when you train only on image 1, 2 or 3.



    Focal Loss (gamma=2)

    - Fold 1: 0.5512 CV, 0.58 public LB
        - exp-1-reproduce-training-with-augm-focal-loss
    - Fold 2: 0.6837 CV
        - exp-1-reproduce-training-with-augm-focal-loss-fold-2
    - Fold 3: 0.7020 CV, 0.56 public LB
        - exp-1-reproduce-training-with-augm-focal-loss-fold-2



    Ensemble of all three give 0.59 public LB.

    Conclusion: focal loss works the best on local CV, but BCE+Dice is better on Kaggle public LB. 
    Maybe this is because current Kaggle data look like image 1. 
    It could be better if focal loss is used in the end, since the private data can end up being more like image 2 and 3.



- ResNet Depth summary
    
    
    Depth 18: 
    
    - Loss is BCE + Dice, Fold 1, 0.5454 CV
        - exp-1-reproduce-training-with-augm
    - Loss is BCE + Dice, Fold 2,  0.4460  CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-2
    - Loss is BCE + Dice, Fold 3,  0.6380  CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-3
    
    Depth 34: 
    
    - Loss is BCE + Dice, Fold 1, 0.5599 CV
        - exp-1-reproduce-training-with-augm-2-fold-1-depth-34
    - Loss is BCE + Dice, Fold 2,  0.4865 CV
        - exp-1-reproduce-training-with-augm-2-fold-2-depth-34
    - Loss is BCE + Dice, Fold 3,  0.6326  CV
        - exp-1-reproduce-training-with-augm-2-fold-3-depth-34

    Conclusion: depth 34 is better than depth 18 on all three folds. 


- Image size analysis

    Image 256 x 256:

    - Loss is BCE + Dice, Fold 1, 0.5454 CV
        - exp-1-reproduce-training-with-augm
    - Loss is BCE + Dice, Fold 2,  0.4460  CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-2
    - Loss is BCE + Dice, Fold 3,  0.6380  CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-3

    Image 384 x 384:

    - Loss is BCE + Dice, Fold 1, 0.5354 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-384
    - Loss is BCE + Dice, Fold 2,  0.4096   CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-384-fold-2
    - Loss is BCE + Dice, Fold 3,  0.6412   CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-384-fold-3

    Image 192 x 192:

    - Fold 1: 0.5345  CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-192-fold-1
    - Fold 2: 0.4398   CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-192-fold-2
    - Fold 3: 0.6194   CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-192-fold-3

    Image 768 x 768:
    - Fold 1: 0.5369 CV, **0.66 LB**
        - exp-1-reproduce-training-with-augm-bce-dice-loss-size-768-fold-1



    Conclusion: 256 x 256 is the best size. Maybe you can train on larger images, but you need to do something special. 



- Image depth analysis


    Z-start=24 → 40 

    - Fold 1: 0.5526  CV, 0.64 LB
        - exp-1-reproduce-training-with-augm-bce-dice-loss
    - Fold 2: 0.4460 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-2
    - Fold 3: 0.6380 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-fold-3

    Z-start=16 → 40 

    - Fold 1: 0.5652  CV,
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-16-40
    - Fold 2: 0.4915 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-16-40-fold-2
    - Fold 3: 0.6232 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-16-40-fold-3

    Z-start=8 → 40 

    - Fold 1:  0.5821 CV,
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-8-40-fold-1
    - Fold 2: 0.4794 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-8-40-fold-2
    - Fold 3: 0.6620 CV
        - exp-1-reproduce-training-with-augm-bce-dice-loss-channels-8-40-fold-3


    Conclusion: 8 → 40 is actually the best?




- Image denoising

    Idea: perform denoising on the images for pretraining, as explored in this paper: https://arxiv.org/abs/2205.11423
    Fold 1: 0.4 CV (am I doing something wrong here????)
        exp-1-reproduce-training-with-augm-bce-dice-loss-size-256-denoising-pretraining


- CutMix

    Idea: perform cutmix augmentation.
    Fold 1:
        exp-1-reproduce-training-with-augm-bce-dice-loss-size-256-cutmix