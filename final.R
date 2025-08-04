# 1. Introduction and Data Preparation
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(forecast)
library(tseries)
library(seasonal)

# Load the dataset
df <- read_csv("main_data.csv")
df$DateTime <- ymd_hms(df$DateTime)
df <- df %>% mutate(Year = year(DateTime), Month = month(DateTime))

# 2. Data Exploration and Visualization
# Time Series Plot (Hourly)
ggplot(df, aes(x = DateTime, y = Global_active_power)) +
  geom_line(alpha = 0.3) +
  labs(title = "Original Time Series Plot: Hourly Global Active Power", x = "Date", y = "Global Active Power (kW)") +
  theme_minimal()

# Seasonal Plot (Hourly)
ggplot(df, aes(x = Month, y = Global_active_power, color = as.factor(Year))) +
  stat_summary(fun = mean, geom = "line") +
  labs(title = "Original Seasonal Plot (Hourly Data)", x = "Month", y = "Global Active Power (kW)", color = "Year") +
  theme_minimal()

# Seasonal Subseries Plot (Hourly)
ggplot(df, aes(x = as.factor(Month), y = Global_active_power)) +
  geom_boxplot() +
  labs(title = "Original Seasonal Subseries Plot", x = "Month", y = "Global Active Power (kW)") +
  theme_minimal()

# ACF and PACF (Hourly)
daily_ts <- ts(df$Global_active_power, frequency = 7)
tsdisplay(daily_ts, main = "Daily Global Active Power: ACF & PACF")

# 3. Aggregation to Monthly Data and Visualization
df_monthly <- df %>%
  mutate(MonthStart = make_date(Year, Month, 1)) %>%
  group_by(Year, MonthStart) %>%
  summarise(Global_active_power = mean(Global_active_power, na.rm = TRUE), .groups = "drop")

monthly_ts <- ts(df_monthly$Global_active_power, start = c(2006, 12), frequency = 12)

autoplot(monthly_ts) + labs(title="Monthly Global Active Power", y="kW", x="")
ggseasonplot(monthly_ts) + labs(title="Monthly Seasonal Plot", y="kW", x="Month")
ggsubseriesplot(monthly_ts) + labs(title="Monthly Subseries Plot", y="kW", x="Month")
tsdisplay(monthly_ts, main = "Monthly Global Active Power: ACF & PACF")

# 4. Transformation and Differencing
lambda <- BoxCox.lambda(monthly_ts)
cat("Calculated Box–Cox lambda =", lambda, "\n")
bc_monthly <- BoxCox(monthly_ts, lambda)
tsdisplay(bc_monthly, main = "Diagnostics: Box-Cox Transformed Series")

log_monthly <- log(monthly_ts)
cat("Using log-transform (lambda = 0) for modeling\n")
tsdisplay(log_monthly, main = "Diagnostics: Log-Transformed Series")

y2_log <- diff(log_monthly, lag = 12)
y3_log <- diff(y2_log, differences = 1)
y3 <- na.omit(y3_log)
tsdisplay(y3, main = "Diagnostics: Log Transformed & Differenced Series")

# 5. Train-Test Split
total_obs <- length(y3)
train_size <- floor(0.8 * total_obs)
times <- time(y3)
end_train_time <- times[train_size]
start_test_time <- times[train_size + 1]
y3_train <- window(y3, end = end_train_time)
y3_test  <- window(y3, start = start_test_time)

# 6. ETS Models
ets_AAA <- ets(y3_train, model = "AAA")
fc_AAA  <- forecast(ets_AAA, h = length(y3_test))
ets_AdA <- ets(y3_train, model = "AAA", damped = TRUE)
fc_AdA  <- forecast(ets_AdA, h = length(y3_test))

autoplot(y3_train, series = "Training Data") +
  autolayer(y3_test, series = "Test Data") +
  autolayer(fc_AAA$mean, series = "ETS(AAA) Forecast") +
  autolayer(fc_AdA$mean, series = "ETS(AdA) Forecast") +
  labs(title = "ETS Forecasts vs Actuals (Log-Differenced Series)", y = "Log(Global Active Power) diffs", x = "Time") +
  theme_minimal()

# 7. ARIMA Models
log_train <- window(log_monthly, end = end_train_time)
log_test  <- window(log_monthly, start = start_test_time)

fit_arima_011_011 <- Arima(log_train, order = c(0,1,1), seasonal = list(order = c(0,1,1), period = 12))
fc_011_011 <- forecast(fit_arima_011_011, h = length(log_test))

fit_arima_110_011 <- Arima(log_train, order = c(1,1,0), seasonal = list(order = c(0,1,1), period = 12))
fc_110_011 <- forecast(fit_arima_110_011, h = length(log_test))

fit_auto <- auto.arima(log_train, seasonal = TRUE)
fc_auto <- forecast(fit_auto, h = length(log_test))

# 8. Model Comparison and Forecast Accuracy
ess_acc_AAA <- accuracy(fc_AAA, y3_test)
ess_acc_AdA <- accuracy(fc_AdA, y3_test)
arma_acc_011_011 <- accuracy(fc_011_011, log_test)
arma_acc_110_011 <- accuracy(fc_110_011, log_test)
arma_acc_auto    <- accuracy(fc_auto,      log_test)

# 9. Final Out-of-Sample Forecast (ETS AdA)
ets_AdA <- ets(y3, model = "AAA", damped = TRUE)
fc_AdA  <- forecast(ets_AdA, h = 12)
autoplot(fc_AdA) +
  labs(title = "Out-of-Sample Forecasts: ETS(ADA) on Log Global Active Power", y = "Log(Global Active Power)", x = "Time") +
  theme_minimal()

# 10. Advanced Models (TBATS and NNAR)
tbats_model <- tbats(y3_train)
fc_tbats     <- forecast(tbats_model, h = length(y3_test))
nnar_model   <- nnetar(y3_train)
fc_nnar      <- forecast(nnar_model, h = length(y3_test))

tbats_train <- tbats(log_train)
fc_tbats_tr  <- forecast(tbats_train, h = length(log_test))
nnar_train   <- nnetar(log_train)
fc_nnar_tr   <- forecast(nnar_train, h = length(log_test))

tbats_acc <- accuracy(fc_tbats_tr, log_test)
nnar_acc  <- accuracy(fc_nnar_tr,  log_test)

# Residual Diagnostics
checkresiduals(ets_AAA)
checkresiduals(ets_AdA)
checkresiduals(fit_arima_011_011)
checkresiduals(fit_arima_110_011)
checkresiduals(fit_auto)
checkresiduals(tbats_train)
checkresiduals(nnar_train)