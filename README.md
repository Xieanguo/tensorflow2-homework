# tensorflow2-homework
Here is some simply code about my tensorflow2 homeworks,I will manage to learn more to be better!

According to the requirements, select the first 100 numbers of each number in the mnist data set to form a training set consisting of 1000 numbers, and then take 80% as the training set and the rest as the test set, and train for 50 epochs respectively to obtain the following results.

The small data set is extracted by the above method.

Big dataset refers to the entire MNIST dataset.

We used big data and small data to train the model separately. During training, the epoch is 50 and the batch_size is the same. There is only the difference in the data used for training. Next, each model is evaluated on the big data and small data validation sets. , and visualized using tensorbroad, observing the results, we found the following conclusions:

At the beginning of training, the model trained on a small dataset has poor performance, and the model trained on a large dataset has always been better.

During the training process, the small data set converges quickly, and the accuracy quickly reaches a high level. The model trained on the large data set belongs to "steady and steady", and the effect has been slowly improving, and finally gradually reached a better level.

When the training is about to end, both models have obtained good results in the training set and test set, and even reached 1.000000 at one point, but on the validation set, we found that the model trained on the small data set was less effective, while the large data set The trained model results are significantly better.

In fact, we found that the value of the loss function of the model on the large dataset was much smaller than that of the model trained on the small dataset, indicating that the model trained on the small dataset "didn't learn enough".

Explanation of the problem: The amount of data in the small data set is too small, the model is easily affected by accident, the convergence is fast, it is easy to overfit, and the generalization ability is weak. After the amount of data increases, the convergence speed is significantly slower and the learning curve is more gentle, but it gradually approaches the final target result, the generalization ability is strong, and the model trained with large amount of data is generally more reliable.
