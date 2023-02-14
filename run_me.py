import pandas as pd
from matplotlib import pyplot as plt

from RNN import RNN     # import RNN class


def scale_0_1(x, min_x, max_x):
    if abs(max_x - min_x) > 1e6:    # prevent dividing by 0
        return (x - min_x)/(max_x - min_x)
    else:
        print('Data hardly varies.')
        return x


# Load .csv file and sort it by date
df = pd.read_csv("data/stock_market_data-AAL.csv")
df = df.sort_values('Date')
#print(df.head())


# Choose price type
data = df['Close'].to_numpy() # options: 'Low', 'High', 'Close', 'Open'
# data = (df['High'].to_numpy() + df['Low].to_numpy()) / 2.0 # mid price


# The first 80% of the data is the training data, the remaining 20% is the test data
length = len(data)
train_length = int(length * 0.8)

train_data = data[:train_length]
test_data = data[train_length:]


# Scale the data to be between 0 and 1 (only depending on the training data set!)
min_x = min(train_data)
max_x = max(train_data)
train_data = scale_0_1(train_data, min_x, max_x)
test_data = scale_0_1(test_data, min_x, max_x)


# Bring the data in shape fot fitting
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)


# Training
my_rnn = RNN(train_length, len(test_data))
my_rnn.train_network(train_data, 20, test_data)    # TODO: maybe as a for loop as in your version Uttam?


# Can we predict some other prices?
data_to_predict = scale_0_1(df['Open'].to_numpy()).reshape(-1,1)
prediction = my_rnn.predict(data_to_predict)

plt.figure()    # figsize = (15,9) this fixed size is quite annoying on my laptop
plt.plot(range(data_to_predict.shape[0]), data_to_predict, label='real')
plt.plot(range(prediction.shape[0]), prediction, label='prediction')
plt.xlabel('Date')
plt.ylabel('American Airlines stock price')
plt.legend()
plt.show()