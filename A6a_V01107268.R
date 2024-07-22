# Install necessary packages if not already installed
if (!require(quantmod)) install.packages("quantmod")
if (!require(forecast)) install.packages("forecast")
if (!require(caret)) install.packages("caret")
if (!require(e1071)) install.packages("e1071")
if (!require(keras)) install.packages("keras")
if (!require(tensorflow)) install.packages("tensorflow")
if (!require(dplyr)) install.packages("dplyr")
if (!require(ggplot2)) install.packages("ggplot2")

# Load necessary libraries
library(quantmod)
library(forecast)
library(caret)
library(e1071)
library(keras)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(zoo)

# Download Microsoft Corporation's historical stock data
getSymbols('MSFT', from='2010-01-01', to='2024-07-19')
data <- na.omit(MSFT)

# Cleaning the data
data <- data.frame(date=index(data), coredata(data))
data <- data %>% filter(!is.na(MSFT.Close))

# Checking for outliers
Q1 <- quantile(data$MSFT.Close, 0.25)
Q3 <- quantile(data$MSFT.Close, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
outliers <- data %>% filter(MSFT.Close < lower_bound | MSFT.Close > upper_bound)
data <- data %>% filter(MSFT.Close >= lower_bound & MSFT.Close <= upper_bound)

# Interpolate missing values
data <- data %>% mutate(MSFT.Close = zoo::na.approx(MSFT.Close))

# Plotting line graph
ggplot(data, aes(x=date, y=MSFT.Close)) +
  geom_line() +
  ggtitle('Microsoft Stock Price') +
  xlab('Date') +
  ylab('Price')

# Creating train and test datasets
set.seed(42)
trainIndex <- createDataPartition(data$MSFT.Close, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- data[trainIndex,]
data_test <- data[-trainIndex,]

# Convert data to monthly
data$date <- as.Date(data$date)
data_monthly <- data %>% group_by(month = as.yearmon(date)) %>% 
  summarise(MSFT.Close = mean(MSFT.Close, na.rm = TRUE))

# Decompose time series into components using additive and multiplicative models
ts_data <- ts(data_monthly$MSFT.Close, frequency = 12)
decomposition_additive <- decompose(ts_data, type = "additive")
decomposition_multiplicative <- decompose(ts_data, type = "multiplicative")

# Plot decompositions
plot(decomposition_additive)
plot(decomposition_multiplicative)

# Univariate forecasting - conventional models / statistical models
# Holt-Winters model
model_hw <- HoltWinters(ts_data, seasonal = "additive")
forecast_hw <- forecast(model_hw, h = 12)
plot(forecast_hw)

# ARIMA model
model_arima <- auto.arima(ts_data)
summary(model_arima)
forecast_arima <- forecast(model_arima, h = 90)
plot(forecast_arima)

# Seasonal ARIMA (SARIMA) model
model_sarima <- auto.arima(ts_data, seasonal = TRUE)
summary(model_sarima)
forecast_sarima <- forecast(model_sarima, h = 90)
plot(forecast_sarima)

# Fit the ARIMA to the monthly series
model_arima_monthly <- auto.arima(ts_data)
summary(model_arima_monthly)
forecast_arima_monthly <- forecast(model_arima_monthly, h = 12)
plot(forecast_arima_monthly)

# Prepare data for LSTM
data_scaled <- scale(data$MSFT.Close)
time_step <- 60
X <- list()
y <- list()
for (i in 1:(length(data_scaled) - time_step)) {
  X[[i]] <- data_scaled[i:(i + time_step - 1)]
  y[[i]] <- data_scaled[i + time_step]
}
X <- array(unlist(X), dim = c(length(X), time_step, 1))
y <- array(unlist(y), dim = c(length(y), 1))

# Split into train and test sets
train_size <- floor(0.8 * nrow(X))
X_train <- X[1:train_size,,]
X_test <- X[(train_size+1):nrow(X),,]
y_train <- y[1:train_size]
y_test <- y[(train_size+1):nrow(y)]

