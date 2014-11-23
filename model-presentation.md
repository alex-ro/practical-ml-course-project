# Practical Machine Learining Course Project - Human Activity Recognition

## Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

## Data Processing and Model Creation
Loading the training data and the test data:


```r
setwd('~/workspace/data-science-specialization/practical-ml-course-project')
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
train_data <- read.csv('./pml-training.csv')
test_data <- read.csv('./pml-testing.csv')
```

The training and test data have a lot of columns (160) and not all of them can be used to train the model because in the test data most columns have mission data (NAs). So, I decided to remove all the columns in test data that contain all NAs and some that are used for counting and time keeping (like num_window, raw_timestamp_part_1). Also did the same for train data. After this we have only 53 columns that will be used to train the model.

```r
test_data <- test_data[,colSums(is.na(test_data)) < nrow(test_data)]
test_data <- test_data[,-c(1, 3, 4, 5, 6, 7, 60)]
test_data$classe <- NA
train_data <- train_data[,colnames(test_data)]
```

Spliting the train data into train and test (10%) to have an estimation of in sample accuracy.

```r
set.seed(16)
trainIndex = createDataPartition(train_data$classe, p=0.9, times=1, list=FALSE)
train = train_data[trainIndex,]
test = train_data[-trainIndex,]
```

I chose the radom forest algorithm because of his good results for clasificaiton and because it does cross-validation internaly.

```r
fit1 <- randomForest(classe ~ ., data = train, ntree=30, mtry=5)
Prediction1 <- predict(fit1, test)
confusionMatrix(test$classe, Prediction1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556   2   0   0   0
##          B   1 377   1   0   0
##          C   0   1 341   0   0
##          D   0   0   3 317   1
##          E   0   0   1   0 359
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.991, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.992    0.986    1.000    0.997
## Specificity             0.999    0.999    0.999    0.998    0.999
## Pos Pred Value          0.996    0.995    0.997    0.988    0.997
## Neg Pred Value          0.999    0.998    0.997    1.000    0.999
## Prevalence              0.284    0.194    0.177    0.162    0.184
## Detection Rate          0.284    0.192    0.174    0.162    0.183
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.995    0.992    0.999    0.998
```
The accuracy is realy high (0.995), but the out of sample error (Out Of Bag estimate) is not that high, 0.79%

```r
fit1
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train, ntree = 30,      mtry = 5) 
##                Type of random forest: classification
##                      Number of trees: 30
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.79%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5010    4    3    3    2    0.002389
## B   12 3383   17    4    2    0.010240
## C    3   18 3046   13    0    0.011039
## D    4    0   35 2852    4    0.014853
## E    0    3    2   10 3232    0.004620
```

Now I create the final model, using all training data

```r
fit2 <- randomForest(classe ~ ., data = train_data, ntree=30, mtry=5)
Prediction2 <- predict(fit2, test_data)
```
The out of sample error for this model (OOB estimate) is 0.64%

```r
fit2
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train_data, ntree = 30,      mtry = 5) 
##                Type of random forest: classification
##                      Number of trees: 30
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5568    7    0    4    1    0.002151
## B   19 3757   20    1    0    0.010535
## C    0   18 3392   11    1    0.008767
## D    1    0   29 3183    3    0.010261
## E    1    0    3    7 3596    0.003050
```
