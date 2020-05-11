# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# use these functions
from utils import look_back_dataset, \
    load_data, \
    normalize, \
    split, \
    reshape, \
    transform, \
    transform_predict, \
    rmse, \
    plot \
 \
# set the lookback (int)
look_back = 10

# load the dataset
dataset = load_data()

# normalize data
dataset = normalize(dataset)

# split to train/test data
train, test = split(dataset)

# reshape looking back for ...
trainX, trainY = look_back_dataset(train, look_back)
testX, testY = look_back_dataset(test, look_back)

#reshape to fit RNN's input shape
trainX, testX = reshape(trainX=trainX,testX=testX)


# build, compile and train RNN
model = Sequential()

model.add(LSTM(
    units=4,
    input_shape=(1,look_back)
))

model.add(Dense(
    units=1
))

# compile
model.compile(
    loss='mean_squared_error',
    optimizer="adam"
)

# train
model.fit(
    trainX, trainY,
    epochs=100,
    batch_size=1,
    verbose=0
)

# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# transform
train_predict = transform_predict(train_predict)
test_predict = transform_predict(test_predict)
trainY = transform(trainY)
testY = transform(testY)

# calculate root mean square error for train and test data
# real Y vs. predicted Y
train_score = rmse(trainY, train_predict)
test_score = rmse(testY, test_predict)
print('Train Score: %.2f RMSE' % (train_score))
print('Test Score: %.2f RMSE' % (test_score))

# plot data on a graph
plot(dataset,train_predict,test_predict,look_back)

# give the mode's summary
model.summary()

