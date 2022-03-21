library(rpart)
library(rpart.plot)
library(pROC)
library(ggthemes)
library(caret) 
require(tidyverse)
require(e1071)
library(mlbench)
library(ipred)
library(randomForest)
library(class)

################
################
## SETUP DATA ##
################
################

# Set seed for have the same sampling in other runs
set.seed(1234)

# Load data
data <- read.csv("./data/data.csv")

table(data$HOME_TEAM_WINS)

set.seed(1234)
# Divide data in train (80%) and test (20%) set
size <- floor(0.80 * nrow(data))
index <- sample(seq_len(nrow(data)), size = size)
train_set <- data[index, ]
test_set <- data[-index, ]

# SINGLE UNPRUNED TREE
single.tree <- rpart(HOME_TEAM_WINS ~ .,  data = train_set,control = rpart.control(minsplit = 0, cp = 0))

single.tree.prediction <- predict(single.tree, test_set, type = 'class')
single.tree.confusion.matrix <- confusionMatrix(table(single.tree.prediction, test_set$HOME_TEAM_WINS))
tree.roc<- roc(test_set$HOME_TEAM_WINS, as.numeric(single.tree.prediction))

# Find the best CP for pruning the big tree
set.seed(1234)
cp_df <- data.frame(printcp(single.tree))
single.tree.pruned <- prune(single.tree, cp = cp_df$CP[which(cp_df$xerror == min(cp_df$xerror))[1]])

single.tree.pruned.prediction <- predict(single.tree.pruned, test_set, type = 'class')
single.tree.pruned.confusion.matrix <- confusionMatrix(table(single.tree.pruned.prediction, test_set$HOME_TEAM_WINS))
pruned.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(single.tree.pruned.prediction))

#bagging
set.seed(1234)
train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

bagging <- bagging(formula = HOME_TEAM_WINS ~ .,  data = train_set,nbagg = 100,control = rpart.control(cp = 0),coob = TRUE,)
bagging.prediction <- predict(bagging, newdata = test_set, type = "class")
bagging.confusion.matrix <-confusionMatrix(table(bagging.prediction, test_set$HOME_TEAM_WINS))

bagging.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(bagging.prediction))

# RANDOM FOREST 
set.seed(1234)
train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

rf <- randomForest(HOME_TEAM_WINS ~ ., train_set, ntree = 100)
rf.prediction <- predict(rf, newdata = test_set, type = "class")

rf.confusion.matrix <- confusionMatrix(table(rf.prediction, test_set$HOME_TEAM_WINS))
rf.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(rf.prediction))

#SVM
SVM_fit =svm( HOME_TEAM_WINS ~. , data= train_set)
predsvm= predict(SVM_fit, test_set)
SVM.confusion.matrix <- confusionMatrix(table(predsvm, test_set$HOME_TEAM_WINS))
SVM.roc<- roc( test_set$HOME_TEAM_WINS, predictor =as.numeric(predsvm))

#KNN
#normalized data excepted the winner variable
normalize <- function(x){ return((x - min(x)) / (max(x) - min(x)))}
data.normalized <- as.data.frame(lapply(data[,-1], normalize))
size <- floor(0.80 * nrow(data.normalized))
index <- sample(seq_len(nrow(data.normalized)), size = size)
train_set <- data.normalized[index, ]
test_set <- data.normalized[-index, ]

train_set.labels <- data[index,1] # 1 y 2 for yes or no corresponding
row_labels <- data[,1]
test_set.labels <- row_labels[-index]

knn_fit2<- knn(train = train_set,test= test_set,cl= train_set.labels, k=65,prob = TRUE,)
KNN.confusion.matrix<- confusionMatrix(table(knn_fit2, test_set.labels))
KNN.roc <- roc(test_set.labels, as.numeric(knn_fit2))

#Plots of ROC
plot(tree.roc, col = "yellow", xlim = c(1, 0))
lines(pruned.roc, col = "red")
lines(bagging.roc, col = "green")
lines(rf.roc, col = "blue")
lines(SVM.roc, col = "hot pink")
lines(KNN.roc, col = "orange")

legend(0.1, 0.5, legend=c("Tree unpruned", "Tree pruned ", "Bagging", "Random Forest", "SVM", "KNN"),
       col=c("yellow", "red", "green", "blue", "hot pink", "orange"), lty=1:2, cex=0.8,
       title="Legends", text.font=4, bg='white')

single.tree.confusion.matrix
single.tree.pruned.confusion.matrix
bagging.confusion.matrix
rf.confusion.matrix
SVM.confusion.matrix
KNN.confusion.matrix




