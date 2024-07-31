Application link:**premiumpredictor-sram.streamlit.app**


Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. Data science addresses these issues by analyzing variable interactions to uncover patterns and trends. Using clustering algorithms, insurers can segment customers based on similar characteristics, enabling more targeted sales strategies and accurate pricing models. By leveraging predictive machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.
The Data and insights
The dataset includes 986 samples and 11 attributes related to health and demographic factors. These attributes cover various health conditions, medical history, and basic physical measurements. The data provides a comprehensive overview that can be analyzed to understand patterns and correlations affecting health insurance premiums.

Distribution of variables
Analysis of the central tendencies of the dataset and visualized the data using histogram-q plots and kernel density estimates (KDE) were done. These visualizations helped in understanding the distribution of each attribute. Additionally, conducting normality hypothesis tests, such as the Shapiro-Wilk test, to determine if the data follows a normal distribution. The results indicated that the data is non-normal, necessitating non-parametric statistical methods for further analysis. This insight is crucial for applying appropriate data science techniques to gain accurate and meaningful insights. Decile checks and outlier analysis revealed a few outliers in weight, premium, and BMI, while all deciles seemed normal. The mean premium check of categorical variables showed that people with health issues are charged higher premiums as expected. There is a very low distribution of customers with transplants, chronic diseases, a family history of cancer, and a high number of surgeries, indicating these high-risk cases are less preferred and are charged higher premiums. The highest premiums are observed for customers with transplants, with no special cases noted otherwise.

Correlation analysis
The correlation analysis of numeric variables indicates that the premium is most strongly correlated with age, followed by BMI. Weight and height are naturally correlated with BMI. Scatter plots do not suggest any clear linear relationships. Hypothesis tests, such as Spearman correlation, confirm the Pearson correlation findings and visual insights, reinforcing the observed patterns.

Violin plots of categorical variables with respect to the premium reveal non-normal distributions. Non-parametric tests, such as the KS test, indicate that all variables influence the premium except for the number of surgeries, likely due to its rarity. Chi-square tests for categorical variable interactions show varying levels of influence between different categories. These analyses highlight the complex factors affecting premium pricing.

Clustering and Cluster analysis
K-means is a clustering algorithm that partitions data into K clusters by minimizing the variance within each cluster. K-means++ is an improved version that initializes centroids to be as far apart as possible, leading to better and more consistent clustering results.

Clusters are typically selected based on the elbow curve method, which plots the explained variance against the number of clusters. The optimal number of clusters is identified at the “elbow” point, where the rate of variance reduction slows down. For our insurance data, the elbow curve suggested four clusters. These clusters might represent distinct groups such as low-risk customers, moderate-risk customers, high-risk customers, and very high-risk customers, each with varying premium levels and health characteristics.

Plotting a pair plot segmented by clusters provides a visual representation of group attributes, helping to identify distinct patterns and relationships. Aggregating data at the group level offers insights into the average premium and other numeric attributes for each cluster, as well as the most common health conditions like diabetes and blood pressure issues. Further analysis, such as drilling down to the decile level and examining value counts, reveals more detailed characteristics of each group. This comprehensive approach helps in understanding the distribution and prevalence of attributes within each cluster, aiding in targeted strategies for insurance pricing and sales.

Modeling
Before diving into the modeling, we preprocess the data to ensure optimal model performance. This involves standardizing the numerical features (age, BMI, number of surgeries) to have a mean of 0 and a standard deviation of 1.

To find the best model for predicting insurance premiums, we evaluate several machine learning algorithms:

Linear Regression
Ridge Regression
Decision Tree Regressor
Random Forest Regressor
Bagging Regressor
AdaBoost Regressor
Gradient Boosting Regressor
Linear Regression:

Linear Regression is a fundamental and widely used model that assumes a linear relationship between the input features and the target variable. We start with this model to establish a baseline for comparison.

Ridge Regression:

Ridge Regression is a linear model that includes a regularization term to prevent overfitting. It is particularly useful when dealing with multicollinearity.

Decision Tree Regressor:

The Decision Tree Regressor is a non-linear model that splits the data into subsets based on the value of input features, creating a tree-like structure. This model can capture complex interactions between features but is prone to overfitting.

Random Forest Regressor:

The Random Forest Regressor is an ensemble learning method that combines multiple decision trees to improve performance and reduce overfitting. By averaging the predictions of individual trees, it achieves better generalization and robustness.

Bagging Regressor:

The Bagging Regressor is an ensemble method that improves the stability and accuracy of machine learning algorithms by combining the predictions of multiple base estimators.

AdaBoost Regressor:

AdaBoost Regressor is an ensemble technique that combines the predictions of multiple weak learners to create a strong predictive model. It works by assigning weights to instances based on the error of the previous model.

Gradient Boosting Regressor:

Gradient Boosting Regressor is an ensemble learning technique that builds models sequentially, with each model attempting to correct the errors of the previous one. This method is highly effective for regression tasks.

Comparison of Models:

After training and evaluating the models, we compare their performance using the Mean Average Error (MAE),Mean Squared Error (MSE) and R-squared (R²) metrics. These metrics provide insight into the accuracy and explanatory power of the models.

Why Random Forest Regressor is the Best
The Random Forest Regressor outperforms the other models, as evidenced by its lower MSE and higher R²(0.86) values. Here are the key reasons why the Random Forest Regressor provides the best results:

Robustness: By averaging the predictions of multiple decision trees, the Random Forest Regressor reduces the risk of overfitting, which is a common issue with individual decision trees.
Handling Non-linearity: Unlike Linear Regression and Ridge Regression, the Random Forest Regressor can
Deployment through streamlit
Preparing the Environment:

Before we start building the Streamlit app, ensure you have the required libraries installed. You can install Streamlit and other necessary libraries using pip, we use srteamlit,pandas,scikit-learn

Loading the Model and Scaler:

First, save the trained model(in our case we observed RF model to be the best based on MSE and R2) and the scaler used for preprocessing the data. This ensures that the same transformations are applied to the input data during prediction.

Building the Streamlit App:

Create a new Python file, app.py, and start by setting up the Streamlit application. Streamlit provides a simple and intuitive API for building interactive web applications.

Creating Input Fields:

Add input fields for the features required by the model. Use Streamlit’s widgets to create interactive input fields for user data.

Preprocessing the Input:

Calculate the BMI and standardize the numerical features before making predictions.

Making Predictions:

Use the preprocessed input data to make predictions using the loaded model. Display the prediction to the user.

Streamlit is a popular choice for creating web applications for the following reasons:

Ease of Use: Streamlit allows you to build interactive web applications with minimal effort. Its syntax is intuitive and straightforward, making it accessible even to those with limited web development experience.

Rapid Development: Streamlit applications can be developed quickly. You can create a web app in just a few lines of Python code, which is ideal for prototyping and iterative development.

Integration with Python Ecosystem: Streamlit is designed to work seamlessly with Python libraries like NumPy, Pandas, Matplotlib, and Scikit-learn. This makes it easy to integrate data analysis and machine learning models into your web application.
