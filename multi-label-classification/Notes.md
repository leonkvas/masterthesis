Validation Loss: 0.1311, Validation Accuracy: 0.9609, Validation Full Sequence Accuracy: 0.7573

mit img size 50 250:

Validation Loss: 0.0737, Validation Accuracy: 0.9802, Validation Full Sequence Accuracy: 0.8645

model dropoutandbatchn.keras 70 train acc but v good validate acc 96 - generalizes very good apparently

robustCNNwith1channel (even though no greyscale yet) best performance:
Validation Loss: 0.1367, Validation Accuracy: 0.9803, Validation Full Sequence Accuracy: 0.8752

15%on65samples:
Test Accuracy: 15.38% on 65 samples.
Validation Loss: 0.1445, Validation Accuracy: 0.9674, Validation Full Sequence Accuracy: 0.8047

robust model with 32 epoches 32 batch size MUCH BETTER!!

Epoch 32/32
24/24 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - accuracy: 0.9990 - full_sequence_accuracy: 0.9929 - loss: 0.0043
Validation Loss: 0.0036, Validation Accuracy: 0.9993, Validation Full Sequence Accuracy: 0.9948
Test Accuracy: 30.77% on 65 samples.

Most Frequent Misclassifications:
  'O' misclassified as '0': 17 times.

----

kk