setwd('~/workspace/data-science-specialization/practical-ml-course-project')
library(caret)
library(rpart)
library(randomForest)

train_data <- read.csv('./pml-training.csv')
test_data <- read.csv('./pml-testing.csv')

#remove columns with NAs
test_data <- test_data[,colSums(is.na(test_data)) < nrow(test_data)]
test_data <- test_data[,-c(1, 3, 4, 5, 6, 7, 60)]
test_data$classe <- NA
train_data <- train_data[,colnames(test_data)]

set.seed(16)
trainIndex = createDataPartition(train_data$classe, p=0.9, times=1, list=FALSE)
train = train_data[trainIndex,]
test = train_data[-trainIndex,]

fit1 <- randomForest(classe ~ ., data = train, ntree=30, mtry=5)
Prediction1 <- predict(fit1, test)
confusionMatrix(test$classe, Prediction1)

# train on all train data and predict the test data for submission
fit2 <- randomForest(classe ~ ., data = train_data, ntree=30, mtry=5)
Prediction2 <- predict(fit2, test_data)

#write the prediction restul into files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
setwd('./answer')
pml_write_files(Prediction2)