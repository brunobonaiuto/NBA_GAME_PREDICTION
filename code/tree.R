# TREE CODE FOR COMPARATION

library(rpart)
library(rpart.plot)
library(vip)
library(pROC)
library(ggthemes)
library(caret) 
library(ipred) 
library(randomForest)

#############
# LOAD DATA #
#############

data <- read.csv("./data/finalCSV.csv")

table(data$HOME_TEAM_WINS)

set.seed(1234)

size <- floor(0.80 * nrow(data))
index <- sample(seq_len(nrow(data)), size = size)
train_set <- data[index, ]
test_set <- data[-index, ]




########################
# SINGLE UNPRUNED TREE #
########################
set.seed(1234)

single.tree <- rpart(
  HOME_TEAM_WINS ~ ., 
  data = train_set,
  control = rpart.control(minsplit = 0, cp = 0)
)

single.tree.prediction <- predict(single.tree, test_set, type = 'class')
single.tree.confusion.matrix <- table(single.tree.prediction, test_set$HOME_TEAM_WINS)
single.tree.accuracy <- sum(diag(single.tree.confusion.matrix)) / sum(single.tree.confusion.matrix)
single.tree.error <- 1 - single.tree.accuracy
single.tree.sensitivity <-  single.tree.confusion.matrix[4] / (single.tree.confusion.matrix[4] + single.tree.confusion.matrix[3])

single.tree.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(single.tree.prediction))
single.tree.auc <- auc(single.tree.roc)





######################
# SINGLE PRUNED TREE #
######################
set.seed(1234)

cp_df <- data.frame(printcp(single.tree))
single.tree.pruned <- prune(single.tree, cp = cp_df$CP[which(cp_df$xerror == min(cp_df$xerror))[1]])

single.tree.pruned.prediction <- predict(single.tree.pruned, test_set, type = 'class')
single.tree.pruned.confusion.matrix <- table(single.tree.pruned.prediction, test_set$HOME_TEAM_WINS)
single.tree.pruned.accuracy <- sum(diag(single.tree.pruned.confusion.matrix)) / sum(single.tree.pruned.confusion.matrix)
single.tree.pruned.error <- 1 - single.tree.pruned.accuracy
single.tree.pruned.sensitivity <-  single.tree.pruned.confusion.matrix[4] / (single.tree.pruned.confusion.matrix[4] + single.tree.pruned.confusion.matrix[3])

single.tree.pruned.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(single.tree.pruned.prediction))
single.tree.pruned.auc <- auc(single.tree.pruned.roc)





############
# BAGGING #
###########
set.seed(1234)

train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

bagging <- bagging(
  formula = HOME_TEAM_WINS ~ .,  
  data = train_set,
  nbagg = 100,
  control = rpart.control(cp = 0),
  coob = TRUE,
)

bagging.prediction <- predict(bagging, newdata = test_set, type = "class")

bagging.confusion.matrix <- table(bagging.prediction, test_set$HOME_TEAM_WINS)
bagging.accuracy <- sum(diag(bagging.confusion.matrix)) / sum(bagging.confusion.matrix)
bagging.error <- 1 - bagging.accuracy
bagging.oob <- bagging[["err"]]
bagging.sensitivity <-  bagging.confusion.matrix[4] / (bagging.confusion.matrix[4] + bagging.confusion.matrix[3])

bagging.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(bagging.prediction))
bagging.auc <- auc(bagging.roc)




#################
# RANDOM FOREST #
#################
set.seed(1234)

train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

rf <- randomForest(HOME_TEAM_WINS ~ ., train_set, ntree = 100)

rf.prediction <- predict(rf, newdata = test_set, type = "class")

rf.confusion.matrix <- table(rf.prediction, test_set$HOME_TEAM_WINS)
rf.accuracy <- sum(diag(rf.confusion.matrix)) / sum(rf.confusion.matrix)
rf.error <- 1 - rf.accuracy
rf.oob <- rf$err.rate[,1][100]
rf.sensitivity <-  rf.confusion.matrix[4] / (rf.confusion.matrix[4] + rf.confusion.matrix[3])

rf.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(rf.prediction))
rf.auc <- auc(rf.roc)
