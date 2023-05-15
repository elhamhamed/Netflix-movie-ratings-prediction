# Netflix-movie-ratings-prediction
**Overview**

The Netflix Movie Ratings Prediction project is a machine learning project that aims to predict movie ratings based on various features such as genre, cast, and runtime. The project uses a dataset of movie ratings collected from Netflix and employs various machine learning algorithms to make predictions
Recommendation system is the most profitable solution for the organization that offers its services to a number of users who have a number of options. As the name suggests, the system uses algorithms to recommend content (videos, ads, news, etc.) to an individual based on their usage.
Content based method utilizes more information about user and items to create a model that learns how a user having certain features will prefer a particular item having a set of characteristics over other items. Content based method uses available information about user and items like features of a user or characteristic or content of an item that explain the interaction between the user and item. The method doesn’t suffer from poor recommendations due to insufficient beginner data.

**Problem Statement**

The goal of this project is to find similarity within groups of people to build a movie recommendation system for users. We are going to analyze a dataset from the Netflix database to explore the characteristics that people share in movies. We have experienced it ourselves or have been in the room, the endless scrolling of selecting what to watch. Users spend more time deciding what to watch than watching their movie.

**Data Summery**

This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Fixable which is a third-party Netflix search engine. In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset. Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.
•	Show_Id: Unique ID for every Movie / Tv Show
•	Category: Identifier - A Movie or TV Show
•	Director: Director of the Movie
•	Cast: Actors involved in the movie / show
•	Country: Country where the movie / show was produced
•	Release_Date: Actual Release Year of the movie / show
•	Rating: TV Rating of the movie / show
•	Duration: Total Duration - in minutes or number of seasons
•	Type: Genere
•	Description: The Summary description

**Dataset**

The dataset used in this project is a collection of movie ratings collected from Netflix. The dataset includes various features such as genre, cast, and runtime, and is available
source :https://www.kaggle.com/datasets/sonalisingh1411/netflix-dataset

**Requirements**

•Data Preprocessing (Cleaning Data, Normalization)
• Split your data set into train data and test data (If the data is not split)
• You will use a suitable algorithm for your dataset. (Classification, Regression, Clustering, Association, etc...).
• The programming language you will be using is python and its libraries (Pandas, Numby, Matplot, Seaborn, Scikitlearn, etc...).
• After prediction, you will work on Evaluation metric.
** 

**•Data Preprocessing**

Before you can begin training and testing your machine learning model, you will need to preprocess the dataset. This includes cleaning the data, handling missing values, and normalizing the data :

  **•Data Cleaning:**
  
   Data cleaning is the process of identifying and correcting or removing inaccurate or irrelevant
   data from the dataset. This may involve handling missing orduplicated data, dealing with outliers, and correcting inconsistencies in     the data.
  Before applying the models, we preprocessed the data by performing the following steps:

--Removed missing values: We removed any rows with missing values in the dataset.

--Converted categorical variables to numerical: We converted the categorical variables such as 'type' and 'rating' to numerical values.

--Split the data into training and testing sets: We split the data into a 80/20 training and testing set to evaluate the performance of    the models.
  **•#Data Visulization :**
Visualizing the Netflix dataset can help you gain a better understanding of the data, identify patterns and trends, and communicate insights to stakeholders. Here are some reasons why you might want to visualize the Netflix dataset:

1. Exploring the data: Data visualization can help you explore the features and characteristics of the Netflix dataset, such as the distribution of the ratings, the number of movies or TV shows released each year, and the popularity of different genres.

2. Finding patterns and trends: Visualization can help you identify patterns and trends in the data that may not be apparent from looking at individual values or tables. For example, you may be able to identify trends in the popularity of certain genres over time, or patterns in the ratings based on the length of the movie or TV show.

3. Communicating insights: Visualization can help you communicate insights and findings to stakeholders in a clear and effective way. By creating visualizations that highlight important trends or patterns, you can help stakeholders understand the significance of the data and make informed decisions based on the insights.

Overall, visualizing the Netflix dataset can help you better understand the data, identify trends and patterns, and communicate insights to stakeholders.


  **•Data Normalization**:
  
   Data normalization is the process of scaling the values of the dataset so that they fall within a specific range or distribution. This is important to ensure
   that the data is comparable and that the machine learning algorithm can make accurate predictions.
        
**• Feature Selection:** 

   Feature selection involves identifying the most relevant features in the dataset that will be used to make predictions.
   This can be done by analyzing the correlation between features and the target variable, and removing any features that
   are redundant or irrelevant.

**• Model:** 
Model Building and Evaluation
We trained and evaluated four different machine learning models on the preprocessed Netflix dataset. The models we used were:

Decision Tree: A decision tree model was built using the sklearn library. We used the default hyperparameters for this model.

Random Forest: A random forest model was built using the sklearn library. We used the default hyperparameters for this model.

Support Vector Classification (SVC): A SVC model was built using the sklearn library. We used the default hyperparameters for this model.

Logistic Regression: A logistic regression model was built using the sklearn library. We used the default hyperparameters for this model.

We evaluated the performance of each model using the following metrics:

Accuracy: The proportion of correctly classified instances in the test set.
Precision: The proportion of true positive predictions among the total positive predictions.
Recall: The proportion of true positive predictions among the total actual positives.
F1-Score: A weighted average of precision and recall.
Results
The performance of the four models on the Netflix dataset is summarized in the table below:

Based on the above results, the logistic regression model had the highest accuracy and F1-score among all models. Therefore, we can conclude that the logistic regression model is the best model for predicting whether a movie or TV show on Netflix is popular or not.


**Conclusion**
In conclusion,  we compared the performance of four different machine learning models on the Netflix dataset. We found that the logistic regression model had the best performance based on the evaluation metrics used. This model can be used to predict the popularity of movies and TV shows on Netflix with a high degree of accuracy. tailored recommendations can be made based on information about movies and TV shows. In addition, similar models can be developed to provide valuable recommendations to consumers in other domains. It will solve for improved movie and TV-Show selection times with a considerable growth in satisfaction of the content being consumed leading to more user engagement and greater trust in Netflix recommendations.
