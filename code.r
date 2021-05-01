library(forecast)
library(leaps)
library(ggplot2)
library(gplots)
library(dplyr)
library(dummies)
library(tidyverse)
require(caTools)
library(rpart.plot)	

train<-read.csv("analysisData.csv")
test<-read.csv("scoringData.csv")

nrow(train)
ncol(train)
nrow(test)
ncol(test)

#Summary stats of training and test.
summary(train)
summary(test)

#Checking for missing values
sum(is.na(train))
sum(is.na(test))


#code for inserting the mean for the missing values in most significant columns.
train = transform(train, bathrooms = ifelse(is.na(bathrooms), mean(bathrooms, na.rm=TRUE), bathrooms))
train = transform(train, host_response_rate = ifelse(is.na(host_response_rate), mean(host_response_rate, na.rm=TRUE), host_response_rate))
train = transform(train, review_scores_rating = ifelse(is.na(review_scores_rating), mean(review_scores_rating, na.rm=TRUE), review_scores_rating))
train = transform(train, bedrooms = ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms))
train = transform(train, beds = ifelse(is.na(beds), mean(beds, na.rm=TRUE), beds))

#Inserting mean in place of n/a in most significant column features of test dataset.
test = transform(test, bathrooms = ifelse(is.na(bathrooms), mean(bathrooms, na.rm=TRUE), bathrooms))
test = transform(test, host_response_rate = ifelse(is.na(host_response_rate), mean(host_response_rate, na.rm=TRUE), host_response_rate))
test = transform(test, review_scores_rating = ifelse(is.na(review_scores_rating), mean(review_scores_rating, na.rm=TRUE), review_scores_rating))
test = transform(test, bedrooms = ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms))
test = transform(test, beds = ifelse(is.na(beds), mean(beds, na.rm=TRUE), beds))
all(is.na(train$is_location_exact))

#Converting train into meaningful factors. For better learning understanding.
train$host_has_profile_pic <- factor(train$host_has_profile_pic)
train$host_identity_verified <- factor(train$host_identity_verified)
train$instant_bookable <- factor(train$instant_bookable)
train$require_guest_profile_picture <- factor(train$require_guest_profile_picture)
train$require_guest_phone_verification <- factor(train$require_guest_phone_verification)
train$is_location_exact <- factor(train$is_location_exact)

# Converting test dataset into factors. For better learning.
test$host_has_profile_pic <- factor(test$host_has_profile_pic)
test$host_identity_verified <- factor(test$host_identity_verified)
test$instant_bookable <- factor(test$instant_bookable)
test$require_guest_profile_picture <- factor(test$require_guest_profile_picture)
test$require_guest_phone_verification <- factor(test$require_guest_phone_verification)
test$is_location_exact <- factor(test$is_location_exact)

# Factoring further columns
train$bed_type <- factor(train$bed_type)
train$property_type <- factor(train$property_type)
train$instant_bookable <- factor(train$instant_bookable)

# Factoring further test features 
test$bed_type <- factor(test$bed_type)
test$property_type <- factor(test$property_type)
test$instant_bookable <- factor(test$instant_bookable)




#70-30 train test division for training of algo and testing of algo.
set.seed(70) 
sample = sample.split(train, SplitRatio = .70)
traint = subset(train, sample == TRUE)
testt  = subset(train, sample == FALSE)

#Training linear regression model for RMSE calculation.
m<-lm(price~instant_bookable+bed_type+is_location_exact+require_guest_phone_verification+require_guest_profile_picture+host_is_superhost+host_has_profile_pic+host_identity_verified+instant_bookable+bathrooms+bedrooms+beds+review_scores_rating+host_is_superhost+host_identity_verified, data=traint)
#plot(m)

#Predicting the test labels using linear regression model.
predicted_value<-predict(m,testt)
summary(predicted_value)

#Accuracy outputs all important accuracy metrices for evaluation.
accuracy(predicted_value, testt$price)

#We will use this linear regression model for submission generation.
m<-lm(price~bathrooms+bedrooms+beds+review_scores_rating+host_is_superhost+host_identity_verified, data=train)

#Predicting the test dataset using the trained linear regression.
predicted_value<-predict(m,test)
summary(predicted_value)

my_submission <- data_frame('Id' = test$id, 'price' = predicted_value)

#Save our file as the first submission for the linear regression.
write_csv(my_submission, 'submissionLinearRegression1.csv')

# Plotting and observing the price for possible outlier and noise detection.
plot(train$price)
nrow(train)

#Now, training linear regression with differnt column features.
m<-lm(price~minimum_nights+number_of_reviews+bathrooms+bedrooms+beds+review_scores_rating+host_is_superhost+host_identity_verified, data=train)
plot(m)

#Predicting the test labels
predicted_value<-predict(m,test)

#Displaying accuracy metrices of model.
accuracy(predicted_value, test$price)

my_submission <- data_frame('Id' = test$id, 'price' = predicted_value)

# save our file asthe 2nd submission of the linear regression.
write_csv(my_submission, 'submissionLinearRegression2.csv')



# Logestic regression model.
LogisticModel = glm(formula = price~minimum_nights+number_of_reviews+bathrooms+bedrooms+beds+review_scores_rating
              , data = train)

plot(LogisticModel)

# Predicting test labels using logistic regression
predicted_value<-predict(LogisticModel,test)
summary(predicted_value)

my_submission <- data_frame('Id' = test$id, 'price' = predicted_value)

# save our file asthe 2nd submission of the linear regression.
write_csv(my_submission, 'submissionLogisticRegression.csv')

#Poison Regression. Model having worst accuracy. We should'nt have used it.
output <-glm(formula = price~minimum_nights+number_of_reviews+bathrooms+bedrooms+beds+review_scores_rating, data = traint,
             family = poisson)
print(summary(output))
plot(output)

predicted_value<-predict(output,test)
summary(predicted_value)

my_submission <- data_frame('Id' = test$id, 'price' = predicted_value)
write_csv(my_submission, 'submissionPoissonRegression.csv')


#Using Random forest. The model with best accuracy. 
library(randomForest)

price <- randomForest(price ~instant_bookable+bed_type+property_type+host_is_superhost+host_identity_verified+ minimum_nights+number_of_reviews+bathrooms+bedrooms+beds+review_scores_rating, data = traint, mtry = 3,
                      importance = TRUE, na.action = na.omit)

plot(price)
price.rf

#Predicting labels of test dataset.
predict<-predict(price, testt)

#Displays all accuracy stats. 
accuracy(predict, testt$price)

# Using random forest on whole train data for test data submission.
price <- randomForest(price ~ bathrooms+bedrooms+beds+review_scores_rating, data = train, mtry = 3,
                         importance = TRUE, na.action = na.omit)
plot(price)
price.rf
predict<-predict(price, test)
my_submission <- data_frame('Id' = test$id, 'price' = predict)

# Save our file. 
write_csv(my_submission, 'submissionRandomForest1.csv')


#Random Forest 2. With enhanced features for training.
price <- randomForest(price ~ property_type+host_is_superhost+number_of_reviews+bathrooms+bedrooms+beds+review_scores_rating, data = train, mtry = 3,
                      importance = TRUE, na.action = na.omit)

predict<-predict(price, test)
my_submission <- data_frame('Id' = test$id, 'price' = predict)

# Saving our file. 
write_csv(my_submission, 'submissionRandomForest3.csv')




#Tree Prediction Decision Tree. A relatively poor model.
library(rpart)

tree = rpart(price~bathrooms+bedrooms+beds+review_scores_rating,data=train,method="anova", control=rpart.control(minbucket=15))
rpart.plot(tree)
plot(tree)

#Predicting test lables using model.
pred <- predict(object=tree,
                newdata = test)

#calculating accuracies
accuracy(pred, test$price)

#Creating dataframe for outputting predictions into csv.
my_submission <- data_frame('Id' = test$id, 'price' = pred)

# save our dataframe into csv file
write_csv(my_submission, 'submissionDecisionTree1.csv')

#Decision Tree submission 2.
tree2 = rpart(price~bathrooms+bedrooms+beds+review_scores_rating,data=train,method="anova",control=rpart.control(minbucket=5))
rpart.plot(tree2)


pred <- predict(object=tree2,
                newdata = test)

#Creating dataframe for csv file submision.
my_submission <- data_frame('Id' = test$id, 'price' = pred)

# save our predicted results in csv file
write_csv(my_submission, 'submissionDecisionTree2.csv')

