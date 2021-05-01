# Kaggle-Competition

Description

People interested in renting an apartment or home, share information about themselves and their property on Airbnb. Those who end up renting the property share their experiences through reviews. The dataset describes property, host, and reviews for over 40,000 Airbnb rentals in New York along 90 variables.*

Goal

Construct a model using the dataset supplied and use it to predict the price of a set of Airbnb rentals included in scoringData.csv.

Metric

Submissions will be evaluated based on RMSE (root mean squared error) (Wikipedia). Lower the RMSE, better the model.

Submission File

The submission file should be in text format (.csv) with only two columns, id and price. The price column must contain predicted price. Number of decimal places to use is up to you. The file should contain a header and have the following format:

"id","price"
25850,136.805271193055
26433,136.716602314255
26588,135.81605109478
30272,138.797573821079
34712,129.395798816707
An example of the sample submission file (sample_submission.csv) is shared with the set of files.

Sample Code

Here is an illustration in R of how you can create a model, apply it to scoringData.csv and prepare a submission file (sample_submission.csv).

# For the following code to work, ensure analysisData.csv and scoringData.csv are in your working directory.

# Read data and construct a simple model
data = read.csv('analysisData.csv')
model = lm(price~minimum_nights+review_scores_accuracy,data)

# Read scoring data and apply model to generate predictions
scoringData = read.csv('scoringData.csv')
pred = predict(model,newdata=scoringData)

# Construct submission from predictions
submissionFile = data.frame(id = scoringData$id, price = pred)
write.csv(submissionFile, 'sample_submission.csv',row.names = F)
* Disclaimer: The data is not supplied by Airbnb. It was scraped from Airbnb's website. We do not either implicitly or explicitly guarantee that the data is exactly what is found on Airbnb's website. This data is to be used solely for the purpose of the Kaggle Project for this course. It is not recommended for any use outside of this competition.





*GENERATING PREDICTION*
For this project, you are given a listing of over 35,000 Airbnb rentals in New York City*. The goal of this competition is to predict the price for a rental using 90 variables on the property, host, and past reviews. To arrive at the predictions, you are encouraged to apply your learning on data exploration, summarization, preparation, and analysis.

Arriving at good predictions begins with gaining a thorough understanding of the data. This could be gleaned from examining the description of predictors, learning of the types of variables, and inspecting summary characteristics of the variables. Visual exploration may yield insights missed from merely examining descriptive characteristics. Often the edge in predictive modeling comes from variable transformations such as mean centering or imputing missing values. Review the predictors to look for candidates for transformation.

Not all variables are predictive, and models with too many predictors often overfit the data they are estimated on. With the large number of predictors available for this project, it is critically important to judiciously select features for inclusion in the model.

There are a number of predictive techniques discussed in this course, some strong in one area while others strong in another. Furthermore, default model parameters seldom yield the best fit. Each problem is different, therefore deserves a model that is tuned for it.

Finally, predictive modeling is an iterative exercise. It is more than likely that after estimating the model, you will want to go back to the data preparation stage to try a different variable transformation.

Goal

Once you construct a set of predictions for Airbnb rentals in the scoring dataset, you will upload your prediction file by clicking on "Submit Predictions". Your submission will be evaluated based on RMSE (root mean squared error) and results posted on Kaggle’s Leaderboard. Lower the RMSE, better the model.
