

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

x<-arima.sim(list(ar = c(0.8897, -0.4858, 0.1, -0.2, -0.1), sd = sqrt(1)), 30000) 
plot(x[1:100], type='l')

write.csv(x, 'ar data.csv')
