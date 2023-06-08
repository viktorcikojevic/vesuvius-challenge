

The purpose of this study is to explore how well a denoiser can be trained. 

- [random-guess](random-guess) - A denoiser that doesn't have inputs changed, so it just "learns" to predict the average of the training data.
- [pretraining](pretraining) - A denoiser that is trained on the same data as the other denoisers, but with the noise added to the inputs. The MSE should be compared with [random-guess](random-guess) to see if it is better.