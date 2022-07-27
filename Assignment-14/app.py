import streamlit as st
import numpy as np
from nsepy import get_history
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time


def compile_model(TimeSteps, TotalFeatures):

    regressor = Sequential()
    regressor.add(LSTM(units=10, activation='relu', input_shape=(
        TimeSteps, TotalFeatures), return_sequences=True))
    regressor.add(LSTM(units=5, activation='relu', input_shape=(
        TimeSteps, TotalFeatures), return_sequences=True))
    regressor.add(LSTM(units=5, activation='relu', return_sequences=False))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    StartTime = time.time()
    regressor.fit(X_train, Y_train, batch_size=5, epochs=100)
    EndTime = time.time()
    regressor.fit(X_train, Y_train, batch_size=5, epochs=100)
    EndTime = time.time()
    st.write("### Total Time Taken: " +
             str(round((EndTime-StartTime)/60))+'Minutes ##')
    return regressor


np.set_printoptions(suppress=True)
st.title('Stock Market Prediction using LSTM')
col1, col2 = st.columns(2)
startDate = (col1.date_input('Enter Start Date'))
endDate = (col2.date_input('Enter End Date'))

symbol = st.text_input('Enter Stock Symbol')

if st.button('Get Data'):
    StockData = get_history(symbol=symbol, start=startDate, end=endDate)
    print(StockData.shape)
    print(StockData.columns)
    StockData['TradeDate'] = StockData.index
    fig = plt.figure(figsize=(20, 6))
    plt.plot(StockData['TradeDate'], StockData['Close'])
    plt.title('Stock Prices Vs Date')
    plt.xlabel('TradeDate')
    plt.ylabel('Stock Price')
    st.pyplot(fig)

    FullData = StockData[['Close']].values
    st.header('Before Normalization')
    st.write(FullData[0:5])

    sc = MinMaxScaler()
    DataScaler = sc.fit(FullData)
    X = DataScaler.transform(FullData)
    st.header('After Normalization')
    st.write(X[0:5])

    X_samples = list()
    Y_samples = list()
    NumberOfRows = len(X)
    TimeSteps = 10

    for i in range(TimeSteps, NumberOfRows, 1):
        X_sample = X[i-TimeSteps:i]
        Y_sample = X[i]
        X_samples.append(X_sample)
        Y_samples.append(Y_sample)

    X_data = np.array(X_samples)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

    Y_data = np.array(Y_samples)
    Y_data = Y_data.reshape(Y_data.shape[0], 1)

    st.header('Data Shapes for LSTM')
    col1, col2 = st.columns(2)
    col1.write(X_data.shape)
    col2.write(Y_data.shape)

    TestingRecords = 5
    X_train = X_data[:-TestingRecords]
    X_test = X_data[-TestingRecords:]
    Y_train = Y_data[:-TestingRecords]
    Y_test = Y_data[-TestingRecords:]

    st.header('Training and Testing Data Shapes')
    col1, col2 = st.columns(2)
    col1.write(X_train.shape)
    col2.write(Y_train.shape)
    col1.write(X_test.shape)
    col2.write(Y_test.shape)

    TimeSteps = X_train.shape[1]
    TotalFeatures = X_train.shape[2]

    st.header('Creating LSTM Model')
    st.write("Number of TimeSteps: " + str(TimeSteps))
    st.write('Number of Features: ' + str(TotalFeatures))
    regressor = compile_model(TimeSteps, TotalFeatures)

    predicted_price = regressor.predict(X_test)
    predicted_price = DataScaler.inverse_transform(predicted_price)

    orig = Y_test
    orig = DataScaler.inverse_transform(Y_test)

    st.header('Visualising the Test Records')
    st.write('Accuracy: ' + str(100-(100*(abs(orig-predicted_price)/orig)).mean()))
    fig = plt.figure(figsize=(20, 6))
    plt.plot(predicted_price, color='blue', label='Predicted Volume')
    plt.plot(orig, color='red', label='Original Volume')
    plt.title('Stock Price Predictions')
    plt.xlabel('Trading Date')
    plt.ylabel('Stock Price')
    st.pyplot(fig)

    st.header('Visualising for Full Data')
    fig = plt.figure(figsize=(20, 6))
    TrainPredictions = DataScaler.inverse_transform(regressor.predict(X_train))
    TestPredictions = DataScaler.inverse_transform(regressor.predict(X_test))

    FullDataPredictions = np.append(TrainPredictions, TestPredictions)
    FullDataOrig = FullData[TimeSteps:]

    plt.plot(FullDataPredictions, color='blue', label='Predicted Price')
    plt.plot(FullDataOrig, color='red', label='Original Price')
    plt.title('Stock Price Predictions')
    plt.xlabel('Trading Date')
    plt.ylabel('Stock Price')
    st.pyplot(plt)

    Last10Days = np.array(StockData['Close'][-10:])
    Last10DaysPrices = Last10Days.reshape(-1, 1)
    X_test = DataScaler.transform(Last10DaysPrices)

    NumberofSamples = 1
    TimeSteps = X_test.shape[0]
    NumberofFeatures = X_test.shape[1]
    X_test = X_test.reshape(NumberofSamples, TimeSteps, NumberofFeatures)
    Next5DaysPrice = regressor.predict(X_test)
    Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)
    st.header('Prediction of Stock Market Price')
    st.write(Next5DaysPrice)
