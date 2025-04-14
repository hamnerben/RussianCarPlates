### Dalton Report page 
I succesfully got a RNN with 1Epoch to submit with very basic layersize and processing.


Decided to switch too LSTM very early in the process using the huberloss loss function, and adam optimizer.



Here's the scores for the lstm without year0: 
* 1 Epoch: 194.46
* 5 Epoch: 174.86
* 10 Epoch: 154.50
* 15 Epoch: 137.4139
* 75 Epoch 81.1514
* 75 + 50 Epoch 78.0753

The score of lstm with year:
* 50 Epoch: ~77
* 300 Epoch: 75.7014


Now I'm gonna try to edit learning rate, loss function, optimizer and other parts to find what works the best with the year plus changing paramters
