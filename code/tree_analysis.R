library(rpart)
library(rpart.plot)
library(vip)
library(pROC)
library(ggthemes)
library(caret) 
library(ipred) 

################
################
## SETUP DATA ##
################
################

data <- read.csv("./data/finalCSV.csv")

table(data$HOME_TEAM_WINS)

set.seed(1234)
# Divide data in train (80%) and test (20%) set
size <- floor(0.80 * nrow(data))
index <- sample(seq_len(nrow(data)), size = size)
train_set <- data[index, ]
test_set <- data[-index, ]

# Remove from train and test test the useless variables
#data <- subset(data, select = -c(1, 2, 3, 4, 23, 41, 45, 46, 50, 51))
#train_set <- subset(train_set, select = -c(1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 15, 17, 23, 41, 45, 46, 50, 51))
#test_set <- subset(test_set, select = -c(1, 2, 3, 4, 23, 41, 45, 46, 50, 51))

#######################
#######################
## SIMPLE TREE MODEL ##
#######################
#######################

#df <- data.frame("NO", 46.3, 34.6, 14.5, 35.6, 26.2, 10.2, 6.5, 45.5, 32.6, 15.7, 35.3, 24.7, 7.3, 3.7, 0.7, 0.52)
#colnames(df) <- c("HOME_TEAM_WINS", "FGP_H", "FG3P_H", "FTM_H", "DREB_H", "AST_H", "STL_H", "BLK_H", "FGP_A", "FG3P_A", "FTM_A", "DREB_A", "AST_A", "STL_A", "BLK_A", "RANKING_H", "RANKING_A")
#test_set <-rbind(test_set, df)

set.seed(1234)

# SINGLE UNPRUNED TREE
single.tree <- rpart(
  HOME_TEAM_WINS ~ ., 
  data = train_set,
  control = rpart.control(minsplit = 0, cp = 0)
)
# 
summary(single.tree)
printcp(single.tree)
plotcp(single.tree)

# PREDICT THE RESULTS WITH THE CONFUSION MATRIX
prediction <- predict(single.tree, test_set, type = 'class')
confusion_matrix <- table(prediction, test_set$HOME_TEAM_WINS)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
error <- 1 - accuracy

# Find the best CP for pruning the big tree
cp_df <- data.frame(printcp(single.tree))
model.pruned<- prune(single.tree, cp = cp_df$CP[which(cp_df$xerror == min(cp_df$xerror))[1]])

summary(model.pruned)
# Predict the results and compute the accuracy and error with the confusion matrix
prediction.pruned <- predict(model.pruned, test_set, type = 'class')
confusion_matrix.pruned <- table(prediction.pruned, test_set$HOME_TEAM_WINS)
accuracy.pruned <- sum(diag(confusion_matrix.pruned)) / sum(confusion_matrix.pruned)
error.pruned <- 1 - accuracy.pruned

# Find the most important variables
vip(model.pruned, num_features = 40, bar = FALSE)

ROC1 <- roc(test_set$HOME_TEAM_WINS, as.numeric(prediction))
plot(ROC1, col = "blue")
auc(ROC1)

ROC2 <- roc(test_set$HOME_TEAM_WINS, as.numeric(prediction.pruned))
plot(ROC2, col = "red", xlim = c(1.5, 0), ylim = c(0, 1))
auc(ROC2)

plot(ROC1, col = "blue", xlim = c(1, 0))
lines(ROC2, col = "red")

legend(1.5, 1, legend=c("ROC Tree unpruned", "ROC Tree pruned "),
       col=c("blue", "red"), lty=1:2, cex=0.8,
       title="Legends", text.font=4, bg='white')

#rpart.plot(model.pruned)

########################
########################
## BAGGING TREE MODEL ##
########################
########################

bag.df <- data.frame("X", "Y", "Z", "L", "M")
names(bag.df)[names(bag.df) == "X.X."] <- "i"
names(bag.df)[names(bag.df) == "X.Y."] <- "bagging.oob"
names(bag.df)[names(bag.df) == "X.Z."] <- "bagging.error"
names(bag.df)[names(bag.df) == "X.L."] <- "bagging.accuracy"
names(bag.df)[names(bag.df) == "X.M."] <- "bagging.auc"

data$HOME_TEAM_WINS <- as.factor(data$HOME_TEAM_WINS)
train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

set.seed(1234)
for (i in 10:200) {
  print(i)
  
  # TRAIN MODEL
  bagging <- bagging(
    formula = HOME_TEAM_WINS ~ .,  
    data = train_set,
    nbagg = i,
    control = rpart.control(cp = 0),
    coob = TRUE,
  )
  
  # PREDICTION
  bagging.pred <- predict(bagging, newdata = test_set, type = "class")
  
  # OOB - TEST ERROR - ACCURACY - AUC
  bagging.oob <- bagging[["err"]]
  confusion_matrix <- table(bagging.pred, test_set$HOME_TEAM_WINS)
  bagging.accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  bagging.error <- 1 - bagging.accuracy
  bagging.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(bagging.pred))
  bagging.auc <- auc(bagging.roc)
  
  # SAVE
  df.temp <- data.frame(i, bagging.oob, bagging.error, bagging.accuracy, bagging.auc)
  
  bag.df <- rbind(bag.df, df.temp)
}

bag.df <- bag.df[-c(1), ]
bag.df$i <- as.numeric(bag.df$i)
bag.df$bagging.oob <- as.numeric(bag.df$bagging.oob)
bag.df$bagging.error <- as.numeric(bag.df$bagging.error)
bag.df$bagging.accuracy <- as.numeric(bag.df$bagging.accuracy)
bag.df$bagging.auc <- as.numeric(bag.df$bagging.auc)

write.csv(bag.df,"./bagging.csv", row.names = FALSE)

ggplot(bag.df, aes(x = i)) +
  geom_line(aes(y = bagging.oob, colour = "OOB error"), size = 0.5) +
  geom_line(aes(y = bagging.error, colour = "Test error"), size = 0.5) +
  scale_colour_manual("", breaks = c("OOB error", "Test error"),
                      values = c("#e06666", "#6fa8dc")) +
  scale_y_continuous(breaks = c(0.19, 0.20, 0.21, 0.22, 0.23, 0.24)) +
  scale_x_continuous(breaks = c(10, 50, 100, 150, 200)) +
  xlab("n° trees") +
  ylab("Error") +
  theme_light()

# CHOOSE THE BEST TREE

set.seed(1234)
bagging <- bagging(
  formula = HOME_TEAM_WINS ~ .,  
  data = train_set,
  nbagg = 100,
  control = rpart.control(cp = 0),
  coob = TRUE,
)

# PREDICTION
bagging.pred <- predict(bagging, newdata = test_set, type = "class")

# OOB - TEST ERROR - ACCURACY - AUC
bagging.oob <- bagging[["err"]]
confusion_matrix <- table(bagging.pred, test_set$HOME_TEAM_WINS)
bagging.accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
bagging.error <- 1 - bagging.accuracy
bagging.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(bagging.pred))
bagging.auc <- auc(bagging.roc)



##############################
##############################
## RANDOM FOREST TREE MODEL ##
##############################
##############################

library(randomForest)
set.seed(1234)
data$HOME_TEAM_WINS <- as.factor(data$HOME_TEAM_WINS)
train_set$HOME_TEAM_WINS <- as.factor(train_set$HOME_TEAM_WINS)
test_set$HOME_TEAM_WINS <- as.factor(test_set$HOME_TEAM_WINS)

rf.df <- data.frame("X", "Y", "Z", "L", "M")
names(rf.df)[names(rf.df) == "X.X."] <- "i"
names(rf.df)[names(rf.df) == "X.Y."] <- "rf.oob"
names(rf.df)[names(rf.df) == "X.Z."] <- "rf.error"
names(rf.df)[names(rf.df) == "X.L."] <- "rf.accuracy"
names(rf.df)[names(rf.df) == "X.M."] <- "rf.auc"
?randomForest
set.seed(1234)
for (i in 10:200) {
    print(i)
  
    rf <- randomForest(HOME_TEAM_WINS ~ ., train_set, ntree = i)
    
    # PREDICTION
    rf.pred <- predict(rf, newdata = test_set, type = "class")
    
    # OOB - TEST ERROR - ACCURACY - AUC
    rf.oob <-  rf$err.rate[,1][i]
    confusion_matrix <- table(rf.pred, test_set$HOME_TEAM_WINS)
    rf.accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
    rf.error <- 1 - rf.accuracy
    rf.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(rf.pred))
    rf.auc <- auc(rf.roc)
    
    df.temp <- data.frame(i, rf.oob, rf.error, rf.accuracy, rf.auc)
    rf.df <- rbind(rf.df, df.temp)
    
}

rf.df <- rf.df[-c(1), ]
rf.df$i <- as.numeric(rf.df$i)
rf.df$rf.oob <- as.numeric(rf.df$rf.oob)
rf.df$rf.error <- as.numeric(rf.df$rf.error)
rf.df$rf.accuracy <- as.numeric(rf.df$rf.accuracy)
rf.df$rf.auc <- as.numeric(rf.df$rf.auc)

write.csv(rf.df,"./random_forest.csv", row.names = FALSE)

ggplot(rf.df, aes(x = i)) +
  geom_line(aes(y = rf.oob, colour = "OOB error"), size = 0.5) +
  geom_line(aes(y = rf.error, colour = "Test error"), size = 0.5) +
  scale_colour_manual("", breaks = c("OOB error", "Test error"),
                      values = c("#F1C74C", "#38b27b")) +
  scale_y_continuous(breaks = c(0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27)) +
  scale_x_continuous(breaks = c(10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)) +
  xlab("n° trees") +
  ylab("Error") +
  theme_light()

# CHOOSE THE BEST TREE

set.seed(1234)
rf <- randomForest(HOME_TEAM_WINS ~ ., train_set, ntree = 100)

# PREDICTION
rf.pred <- predict(rf, newdata = test_set, type = "class")

# OOB - TEST ERROR - ACCURACY - AUC
rf.oob <-  rf$err.rate[,1][100]
confusion_matrix <- table(rf.pred, test_set$HOME_TEAM_WINS)
rf.accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
rf.error <- 1 - rf.accuracy
rf.roc <- roc(test_set$HOME_TEAM_WINS, as.numeric(rf.pred))
rf.auc <- auc(rf.roc)

###############################
###############################
## COMPARISION OF THE MODELS ##
###############################
###############################

df <- data.frame(bag.df, rf.df)

ggplot(df, aes(x = i)) +
  geom_line(aes(y = bagging.oob, colour = "OOB: Bagging"), size = 0.5) +
  geom_line(aes(y = bagging.error, colour = "Test: Bagging"), size = 0.5) +
  geom_line(aes(y = rf.oob, colour = "OOB: Random Forest"), size = 0.5) +
  geom_line(aes(y = rf.error, colour = "Test: Random Forest"), size = 0.5) +
  geom_hline(yintercept = 0.198, linetype='dotted', col = "black") +
  scale_colour_manual("", breaks = c("OOB: Bagging", "Test: Bagging", "OOB: Random Forest", "Test: Random Forest"),
                      values = c("#e06666", "#6fa8dc", "#F1C74C", "#38b27b")) +
  scale_y_continuous(breaks = c(0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24
                                , 0.25, 0.26)) +
  scale_x_continuous(breaks = c(10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)) +
  xlab("n° trees") +
  ylab("error indexes") +
  theme_light()


