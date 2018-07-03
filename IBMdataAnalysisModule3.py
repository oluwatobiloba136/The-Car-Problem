# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:17:49 2018

@author: oluwatobiloba
"""

"""
        MODULE 3
        
         EXPLORATORY DATA ANALYSIS
         
"""

"""
    In this section, we will explore several methods to see 
    if certain characteristics or features can be used to predict price.

    The Main Question is:

'What are the main characteristics which have the most impact on the car price?'

"""

# let import the clean data from module 2

import pandas as pd

import numpy as np

path = "C:/Users/oluwatobiloba/Desktop/APPZ/pythonClassIBMCo/DataAnalysisWithPython/clean_df.txt"

# let load it and store in a dataframe

df = pd.read_csv(path)
df.head()

"""

    Analyzing Individual Feature Patterns using Visualization
    
"""

"""
Import visualization packages "Matplotlib" and "Seaborn", 
don't forget about "%matplotlib inline" to plot in a Jupyter notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline (use for ploting in jupter notebook)
 
"""
  Choosing the right visualization method

When visualizing individual variables, it is important to first
 understand what type of variable you are dealing with. 
 
This will help us find the right visualisation method for that variable.
"""

# list the data types for each column
df.dtypes

# let for instance check the data type of 'peak-rpm', this we shall do as:

df["peak-rpm"].dtypes

"""
let for example, we can calculate the
 correlation between variables of type "int64" or "float64"
 
 using the method "corr":
     
     NOTE: the method "corr"  return a full correlation matrix as a DataFrame
     and the diagonal elements are always  1
"""

df.corr()

"""
Now let Find the correlation between the following columns:
    bore, stroke,compression-ratio , and horsepower. 
    
"""

df[['bore','stroke' ,'compression-ratio','horsepower']].corr()


"""
        Continuous numerical variables

"""

"""
Continuous numerical variables are variables that may contain
 any value within some range. 
 
 Continuous numerical variables can have the type "int64" or "float64". 
 
 A great way to visualize these variables is by using 
 scatterplots with fitted lines.

In order to start understanding the (linear) relationship between
 an individual variable and the price.
 
 We can do this by using "regplot", which plots the
 scatterplot plus the fitted regression line for the data.
 
"""

# Let's see several examples of different linear relationships:

"""
 Let's find the scatterplot of
         "engine-size" and "price" (a positive linear relationship)
         
"""

# Engine size as potential predictor variable of price

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

"""
We can see As the engine-size goes up, the price goes up: 
    
    this indicates a positive direct correlation between these two variables.
    Engine size seems like a pretty good predictor of price 
    since the regression line is almost a perfect diagonal line. E
    
"""

# let examine the correlation coefficient  between 'engine-size' and 'price'

df[['engine-size','price']].corr()

# we can see it's approximately 0.87


"""
    Let obtain the scatterplot of
        "highway-mpg" and "price" (a negative linear relationship)
        
"""

# highway-mpg as potential predictor variable of price

sns.regplot(x="highway-mpg", y="price", data=df)

"""
 We can see As the highway-mpg goes up, the price goes down: 
     
  this indicates an inverse/ negative relationship between these two variables.
   Highway mpg could potentially be a predictor of price.
   
"""

# let examine the correlation coefficient  between 'highway-mpg' and 'price'

df[['highway-mpg','price']].corr()

# we can see it's approximately -0.70

"""
    Weak Linear Relationship

"""

# Let's see if "Peak-rpm" as a predictor variable of "price".

sns.regplot(x="peak-rpm", y="price", data=df)

"""
    As we can see Peak rpm does not seem like a good predictor of the price
    at all since the regression line is close to horizontal.
    
    Also, the data points are very scattered and far from the fitted line,
    showing lots of variability.
    
    Therefore it is not a reliable variable.
"""

# let examine the correlation coefficient  between 'peak-rpm' and 'price'

df[['peak-rpm','price']].corr()

# we can see it's approximately -0.10

# let Find the correlation between x="stroke", y="price". 

df[["stroke","price"]].corr() 

"""
The correlation is 0.0823, the non-diagonal elements of the table.

There is a weak correlation between the variable 'stroke' and 'price.'
 as such regression will not work well. 
 
 We can see this using the "regplot" to demonstrate this.
"""

sns.regplot(x="stroke", y="price", data=df)

"""

    Categorical variables

These are variables that describe a 'characteristic' of a data unit, 
and are selected from a small group of categories. 

The categorical variables can have the type "object" or "int64".

 A good way to visualize categorical variables is by using boxplots.
"""

# Let's look at the relationship between "body-style" and "price".

sns.boxplot(x="body-style", y="price", data=df)

"""
 We see that the distributions of price between the different 
body-style categories have a significant overlap, and so body-style would not
 be a good predictor of price. 
"""
 
# Let's examine "engine-location" and "price" :

sns.boxplot(x="engine-location", y="price", data=df)

"""
Here we see that the distribution of price between these
 two engine-location categories, front and rear, are distinct enough
 to take engine-location as a potential good predictor of price.
"""

# Let's examine "drive-wheels" and "price".

sns.boxplot(x="drive-wheels", y="price", data=df)

"""
Here we see that the distribution of price between the different 
drive-wheels categories differs; 

as such drive-wheels could potentially be a predictor of price.
"""


"""
        Descriptive Statistical Analysis
        
"""

"""
Let's first take a look at the variables by utilising a description method.

The **describe** function automatically computes basic statistics
 for all continuous variables.
 
 Any NaN values are automatically skipped in these statistics.
 
 This will show:
     
- the count of that variable
- the mean
- the standard deviation (std) 
- the minimum value
- the IQR (Interquartile Range: 25%, 50% and 75%)
- the maximum value

"""
# We can apply the method "describe" as follows:

df.describe()

"""
The default setting of "describe" skips variables of type object. 

We can apply the method "describe" on the variables
 of type 'object' as follows:
"""

df.describe(include=['object'])

"""
    value_counts() method
    
Value_counts is a good way of understanding how many units of
 each characteristic/variable we have.
 
 We can apply the "value_counts" method on the column 'drive-wheels'. 
 
 Don’t forget the method "value_counts" only works on Pandas series, 
 not Pandas Dataframes.
 
 This means the output is a series not a Dataframes.

 As a result, we only include one
 bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".
"""

df['drive-wheels'].value_counts()


# Note: We can convert the series to a Dataframe as follows 
df['drive-wheels'].value_counts().to_frame()

"""
    Let's repeat the above steps 
    but save the results to the dataframe "drive_wheels_counts" 
"""
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()

# let rename the column 'drive-wheels' to 'value_counts'

"""
    If you want to create a transformed version of a dataset
 without modifying the original, a useful method is rename:

"""

drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)

# check the created dataframe
drive_wheels_counts

# Now let's rename the index to 'drive-wheels':

"""
    You can assign "name"  to "index object", modifying the DataFrame in-place:
because pandas’s Index objects are responsible for holding the axis labels
 and other metadata (like the axis name or names).
 
"""
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# We can repeat the above process for the variable 'engine-location'.

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

"""
Examining the value counts of the engine location would not
 be a good predictor variable for the price. 
 This is because we only have three cars with a rear engine 
 and 198 with an engine in the front, this result is skewed.
 
 Thus, we are not able to draw any conclusions about the engine location.
"""


""" 

    Basic of Grouping
    
"""

"""
The "groupby" method groups data by different categories. 
The data is grouped based on one or several variables and 
analysis is performed on the individual groups.

For example, 

If we want to know, on average, which type of drive wheel is most valuable,
 we can group "drive-wheels" and then average them.
 
    let's group by the variable "drive-wheels".
"""
# let check the different categories of drive wheels taht exists with the ffg code
df['drive-wheels'].unique()

"""
we can select the columns 'drive-wheels','body-style' and 'price' , 
then assign it to the variable "df_group_one".
"""

df_group_one=df[['drive-wheels','body-style','price']]

"""
we can then calculate the average price for each of 
the different categories of data.
"""

df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
df_group_one

"""
the above code is literally trandslated as
    "group variable df_group_one on the basis of 'drive-wheel' and 
    the calculate the mean for each group obtained"
"""

"""
From our data, it seems rear-wheel drive vehicles are, on average, the 
most expensive, while 4-wheel and front-wheel are approximately 
the same in price. 
"""


"""
    You can also group with multiple variables.

 For example, let's group by both 'drive-wheels' and 'body-style'.
 This groups the dataframe by the unique combinations 'drive-wheels' 
and 'body-style'. We can store the results in the variable 'grouped_test1'

"""

df_gptest=df[['drive-wheels','body-style','price']]
grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],
                                as_index= False).mean()
grouped_test1

"""
    PIVOT TABLE
    
This grouped data is much easier to visualize when it is
 made into a pivot table. 
 
 A pivot table is like an Excel spreadsheet, with one variable 
 along the column and another along the row. 
 
 We can convert the dataframe to a pivot table using the
 method "pivot " to create a pivot table from the groups.

In this case, we will leave the drive-wheel variable as the rows 
of the table, and pivot body-style to become the columns of the table:
"""

grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

"""
Often, we won't have data for some of the pivot cells. 

We can fill these missing cells with the value 0, but any other
 value could potentially be used as well. It should be mentioned that 
 missing data is quite a complex subject and is an entire course on its own.
 
"""
#fill missing values with 0
grouped_pivot=grouped_pivot.fillna(0) 
grouped_pivot

"""
let Use the "groupby" function to find the average "price"
 of each car based on "body-style" ? 
"""
df_group_two=df[['body-style','drive-wheels','price']]
df_group_two=df_group_two.groupby(['body-style'],as_index= False).mean()
df_group_two

"""
    HEAT MAP (graphical representation of pivot table)
    
"""
# If you didn't import "pyplot" let's do it again. 

import matplotlib.pyplot as plt
# % matplotlib inline (for jupter notebook)

"""
Let's use a heat map to visualize the relationship between Body Style vs Price 

Variables: Drive Wheels and Body Style vs Price
"""

#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

"""
The heatmap plots the target variable (price) proportional to colour
 with respect to the variables 'drive-wheel' and 'body-style' 
 in the vertical and horizontal axis respectively. This allows us 
 to visualize how the price is related to 'drive-wheel' and 'body-style'.
 
The default labels convey no useful information to us. Let's change that:
"""

fig, ax=plt.subplots()
im=ax.pcolor(grouped_pivot, cmap='RdBu')

"""
    Note: the column obtained from that pivot table is MultiIndex, so we 
use 'levels method' to obtained the index we want which is 'body-style'

    Here is the columns for details
In[1]:grouped_pivot.columns
Out[1]: 
MultiIndex(levels=[['price'], ['convertible', 'hardtop', 'hatchback', 'sedan',
 'wagon']],
           labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],
           names=[None, 'body-style'])

so we can label as below:
"""
#label names
row_labels=grouped_pivot.columns.levels[1]
col_labels=grouped_pivot.index

"""
 Note: the shape of pivot table is such that it has 3 row and 5 column
 
In[1]:grouped_pivot.shape
Out[1]: (3, 5)

In[2]:grouped_pivot.shape[1]
Out[2]: 5

In[3]:np.arange(grouped_pivot.shape[1])
Out[3]: array([0, 1, 2, 3, 4])

so we can code as below:
"""

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

"""
Visualization is very important in data science, and Python visualization
 packages provide great freedom.
 We will go more in-depth in a separate Python Visualizations course.
 
 The main question we want to answer in this module, is 
 
 "What are the main characteristics which have the most impact on the car price?". 
 
 To get a better measure of the important characteristics, 
 we look at the correlation of these variables with the car price,
 in other words: how is the car price dependent on this variable?
"""
 
 
"""

        CORRELATION AND CAUSATION
"""

"""
Correlation: a measure of the extent of interdependence between variables.

Causation: the relationship between cause and effect between two variables.

It is important to know the difference between these two and that
 correlation does not imply causation. Determining correlation is much 
 simpler the determining causation as causation
 may require independent experimentation 
"""

""" 
    Pearson Correlation

The Pearson Correlation measures the linear dependence between
 two variables X and Y. The resulting coefficient is a value between 
 -1 and 1 inclusive, where:
     
•1: total positive linear correlation,
•0: no linear correlation, the two variables most likely do not affect each other
•-1: total negative linear correlation.

Pearson Correlation is the default method of the function "corr". 
"""
df.corr()

# sometimes we would like to know the significant of the correlation estimate.

"""
    P-value:

 What is this P-value? The P-value is the probability value that the 
 correlation between these two variables is statistically significant. 
 Normally, we choose a significance level of 0.05, which means that
 we are 95% confident that the correlation between the variables is significant. 

By convention, when the 

    •p-value is < 0.001 
    we say there is strong evidence that the correlation is significant,
    •the p-value is < 0.05;
    there is moderate evidence that the correlation is significant,
    •the p-value is < 0.1;
    there is weak evidence that the correlation is significant, and
    •the p-value is > 0.1; 
    there is no evidence that the correlation is significant.

We can obtain this information using "stats" module in the "scipy" library.

"""

from scipy import stats

"""
 Let's calculate the Pearson Correlation Coefficient 
and P-value of 'wheel-base' and 'price'. 
"""
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is",
      pearson_coef, " with a P-value of P =", p_value)  


"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between wheel-base and price 
is statistically significant, although the linear relationship
 isn't extremely strong (~0.585)
 
"""

"""
Let's calculate the Pearson Correlation Coefficient 
and P-value of 'horsepower' and 'price'.
"""

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value)  

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between horsepower 
and price is statistically significant, and the linear relationship 
is quite strong (~0.809, close to 1)
"""

"""
Let's calculate the Pearson Correlation Coefficient 
and P-value of 'length' and 'price'.
"""

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, 
      " with a P-value of P =", p_value)

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between length 
and price is statistically significant, and the linear relationship 
is moderately strong (~0.691).
"""

"""
Let's calculate the Pearson Correlation Coefficient 
and P-value of 'width' and 'price':
"""

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value ) 

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between width and
 price is statistically significant, and the linear relationship
 is quite strong (~0.751).
"""

"""
Let's calculate the Pearson Correlation Coefficient
and P-value of 'curb-weight' and 'price':
"""

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value ) 

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between curb-weight 
and price is statistically significant, and the linear relationship
 is quite strong (~0.834).
"""

"""
Let's calculate the Pearson Correlation Coefficient
 and P-value of 'engine-size' and 'price':
"""

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value ) 

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between 
engine-size and price is statistically significant, and 
the linear relationship is very strong (~0.872).
"""

"""
Let's calculate the Pearson Correlation Coefficient
 and P-value of 'bore' and 'price':
"""
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value ) 

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between bore 
and price is statistically significant, but the linear
 relationship is only moderate (~0.521).
"""

# We can relate the process for each 'City-mpg' and 'Highway-mpg':

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value)

""" 
    Our Conclusion:

Since the p-value is < 0.001, the correlation between city-mpg
 and price is statistically significant, and the coefficient
 of ~ -0.687 shows that the relationship is negative and moderately strong.
"""

# Highway-mpg vs price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef,
      " with a P-value of P =", p_value ) 

"""
    Our Conclusion:

Since the p-value is < 0.001, the correlation between highway-mpg
 and price is statistically significant, and the coefficient
 of ~ -0.705 shows that the relationship is negative and moderately strong.
"""



"""
        ANOVA
        
ANOVA: Analysis of Variance

The Analysis of Variance (ANOVA) is a statistical method used to
 test whether there are significant differences between the means
 of two or more groups. 
 
     ANOVA returns two parameters:

F-test score: ANOVA assumes the means of all groups are the same,
 calculates how much the actual means deviate from the assumption,
 and reports it as the F-test score.
 
 A larger score means there is a larger difference between the means.

P-value: P-value tells how statistically significant is our calculated
 score value

If our price variable is strongly correlated with the variable
 we are analyzing, 
 
 expect ANOVA to return a sizeable F-test score and a small p-value.

"""

"""
    Drive Wheels

Since ANOVA analyzes the difference between different groups of
 the same variable, the groupby function will come in handy. 
 Because the ANOVA algorithm averages the data automatically,
 we do not need to take the average before hand.
 
Let's see if different types 'drive-wheels' impact 'price', we group the data.
"""

grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
grouped_test2.head(2)

# We can obtain the values of the method group using the method "get_group". 

grouped_test2.get_group('4wd')['price']

"""
we can use the function 'f_oneway' in the module 'stats'
 to obtain the F-test score and P-value.
"""

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],
                              grouped_test2.get_group('rwd')['price'], 
                              grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   

"""
This is a great result, with a large F test score showing a strong correlation
 and a P value of almost 0 implying almost certain statistical significance.
 
 But does this mean all three tested groups are all this highly correlated? 
 
     Separately: fwd and rwd
"""

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],
                              grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )

# Let's examine the other groups 

# 4wd and rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], 
                              grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)  

# 4wd and fwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'],
                              grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)  

"""
        Conclusion: Important Variables
        
    


We now have a better idea of what our data looks like and
 which variables are important to take into account when predicting 
 the car price.

 We have narrowed it down to the following variables:

Continuous numerical variables:
•Length
•Width
•Curb-weight
•Engine-size
•Horsepower
•City-mpg
•Highway-mpg
•Wheel-base
•Bore

Categorical variables:
•Drive-wheels

AS we now move into building machine learning models to automate our
 analysis, feeding the model with variables that meaningfully affect
 our target variable will improve our model's prediction performance.


About the Authors:

This notebook written by Mahdi Noorian PhD ,Joseph Santarcangelo PhD,
 Bahare Talayian, Eric Xiao, Steven Dong, Parizad , Hima Vsudevan 
 and Fiorella Wenver.

        
"""



