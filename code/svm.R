data <- read.csv("./data/data.csv")
size <- floor(0.80*nrow(data))
set.seed(1234)
data$HOME_TEAM_WINS <- as.factor(data$HOME_TEAM_WINS)
train_ind<- sample(seq_len(nrow(data)), size=size)
train_set <- data[train_ind, ]
test_set <- data[-train_ind, ]

require(rlang)
library(caret)
require(tidyverse)
require(e1071)
library(rminer)
library(pROC)

M0=HOME_TEAM_WINS ~ .
F0 =svm(M0, data= train_set, importance=T)
F0

w <- t(F0$coefs) %*% F0$SV                 # weight vectors
w <- apply(w, 2, function(v){sqrt(sum(v^2))})  # weight
w <- sort(w, decreasing = T)
print(w)

pred0= predict(F0, test_set)
confusionMatrix(table(pred0, test_set$HOME_TEAM_WINS))
roc0 <- roc( test_set$HOME_TEAM_WINS, predictor =as.numeric(pred0))
plot(roc0, col = "blue")

##cross validation error with 10 folds
cv.error = function(formula, learner, data, k, ...) {
  indexes = sample(nrow(data))
  errs = c(1:k) %>% map_dbl(function(i) {
    indexes.test = indexes[c((nrow(data)/k*(i-1)+1):(nrow(data)/k*i))]
    m = learner(formula, data[-indexes.test,], ...)
    predicted.y = predict(m, data[indexes.test,], type = "class")
    actual.y = data[indexes.test, as.character(f_lhs(formula))]
    confusion.matrix = table(actual.y, predicted.y)
    1-sum(diag(confusion.matrix))/sum(confusion.matrix)
  })
  names(errs) = paste0("fold", c(1:k))
  errs
}

results0 = expand_grid(kernel=c("linear","polynomial","radial","sigmoid"), cost=exp(seq(-6,8,1))) %>% rowwise() %>% mutate(error = mean(cv.error(M0, svm, train_set, 10, kernel=kernel, cost=cost, degree=2)))
results0 %>% ggplot(aes(x=cost,y=error,color=kernel)) + geom_line() + scale_x_log10() + geom_point()


