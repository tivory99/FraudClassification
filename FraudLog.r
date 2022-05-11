#import data and drop columns
CREDITFRAUDDATASET <- read.csv("C:/Users/tivory/Downloads/CREDITFRAUDDATASET.csv")
FraudData <- CREDITFRAUDDATASET[-c(1,2,4,7,11)]

#split data into testing and training subsets
fraud_split <- sample.split(Y = FraudData$isFraud, SplitRation = 0.7)
train_set <- subset(x = FraudData, fraud_split == TRUE)
test_set <- subset(x = FraudData, fraud_split == FALSE)

#create logistic regression
logistic <- glm(isFraud ~ ., data = train_set, family = "binomial")

#evaluate model
probs <- predict(logistic, newdata = test_set, type = "response")
pred <- ifelse(probs > 0.5, 1, 0)
confusionMatrix(factor(pred), factor(test_set$isFraud), positive = as.character(1))