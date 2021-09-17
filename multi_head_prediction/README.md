Multi-task learning code for time window prediction
Tested on pytorch 1.5.1, Python 3.6.8
test_classification.py :predict if the onset time is below or above the fixed time
1. put the test data in toy_data
2. prepare the csv file in doc/toy, e.g. test_cls.csv
3. change the arguments in test_classification.py if needed(mainly the model,test and save path)
4. run the test_classification.py
test_regression.py:regression of  the onset time
1. put the test data in toy_data
2. prepare the csv file in doc/toy, e.g. test_reg.csv
3. change the arguments in test_regression.py if needed(mainly the model,test and save path)
4. run the test_classification.py

model:
pretrained models:
6h.pth: for prediction if the onset time is below or above 6h
8h.pth: for prediction if the onset time is below or above 8h
12h.pth: for prediction if the onset time is below or above 12h
reg.pth: for regression of the onset time