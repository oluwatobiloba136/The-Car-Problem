# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:51:07 2018

@author: oluwatobiloba
"""

import pandas as pd

# the first path below is the url from the data source through internet
path ="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"


df = pd.read_csv(path, header=None)
print('Done')

# showing the first five row

df.head(5)

# bottom 10 row of the dataframe

df.tail(10)

"""
our data has no header so we have to introduce
header manually, the header information is 
supplied from the url
"""
#Firstly, we create a list "headers" that include all column names in order.

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

headers

# We replace headers and recheck our data frame

df.columns = headers
df.head(10)

# we can drop missing values along the column "price" as follows

df.dropna(subset=['price'],axis=0)
"""
Now, we have successfully 
read the raw dataset and add the correct headers into the data frame.
"""

# the name of the columns of the dataframe can be obtained with the code

df.columns

# let save  save the dataframe "df" as "automobile.csv" to our local machine
df.to_csv("automobile.csv")
"""
# In order to better learn about each attribute, 
#it is always good for us to know the data type of each column.
# let check the data type of data frame "df" by .dtypes
"""

df.dtypes

# for statistical summary of each column we use

df.describe()
"""
# but This method will provide 
#various summary statistics, excluding NaN (Not a Number) values.
so to check for all columns including the object type, use
"""

df.describe(include="all")

"""
let us apply the method describe() to the length
and compression-ratio, this is achieve thus
"""

df[["length","compression-ratio"]].describe()

""" 
we can also use the info method to get concise summary of the dataframe
"""

df.info


"""
MODule 2 starts here (it is all about data cleaning)

"""


"""
Data wrangling is the process of converting data from the initial format 
to a format that may be better for analysis.
"""

"""

IDENTIFICATION AND HANDLING OF MISSING DATA

As we can see, several question marks appeared in the dataframe; 
those are missing values which may hinder our further analysis.

So, how do we identify all those missing values and deal with them?
How to work with missing data?

Steps for working with missing data:

identify missing data
deal with missing data
correct data format
"""

"""
In the car dataset, missing data comes with the question mark "?".
 We replace "?" with NaN (Not a Number), which is Python's default
 missing value marker, for reasons of computational speed and convenience.
 Here we use the function:

.replace(A, B, inplace = True) 
to replace A by B
"""
# the code to repalce ? with Nan

import numpy as np
df.replace("?", np.nan, inplace=True)
df.head(5)

"""
Evaluating for Missing Data

The missing values are converted to Python's default.
 We use Python's built-in functions to identify these missing values. 
 There are two methods to detect missing data:

.isnull()
.notnull()

The output is a boolean value indicating whether the passed in argument value
 are in fact missing data.
 """
 
missing_data = df.isnull()
missing_data.head(5)
 
 # "True" stands for missing value, while "False" stands for not missing value.
 
"""
 Count missing values in each column
 
Using a for loop in Python, we can quickly figure out the number of 
missing values in each column. As mentioned above, "True" represents 
a missing value, "False" means the value is present in the dataset.
 In the body of the for loop 
 the method ".value_counts()" counts the number of "True" values.
"""
 
for column in missing_data.columns.values.tolist():
     print(column)
     print(missing_data[column].value_counts())
     print("")
     
     
"""
Dealing with missing values
"""

"""
**How to deal with missing data?**

    
    1. drop data 
        a. drop the whole row
        b. drop the whole column
        
    2. replace data
        a. replace it by mean
        b. replace it by frequency
        c. replace it based on other functions
        
"""

"""
Whole columns should be dropped only if most entries in the column are empty.
 In our dataset, none of the columns are empty enough to drop entirely.
We have some freedom in choosing which method to replace data; however,
 some methods may seem more reasonable than others. 
 We will apply each method to many different columns:

**Replace by mean:**

    "normalized-losses": 41 missing data, replace them with mean
    "stroke": 4 missing data, replace them with mean
    "bore": 4 missing data, replace them with mean
    "horsepower": 2 missing data, replace them with mean
    "peak-rpm": 2 missing data, replace them with mean
 
**Replace by frequency:**

    "num-of-doors": 2 missing data, replace them with "four". 
        * Reason: 84% sedans is four doors. Since four doors is most frequent, 
        it is most likely to 
    

**Drop the whole row:**

    "price": 4 missing data, simply delete the whole row
        * Reason: price is what we want to predict. Any data entry without 
        price data cannot be used for prediction; therefore they are not
        useful to us
        
"""
#### Calculate the average of the column
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)

"""
Replace "NaN" by mean value in "normalized-losses" column
"""

df["normalized-losses"].replace(np.nan, avg_1, inplace = True)

### Calculate the mean value for 'bore' column

avg_2=df['bore'].astype('float').mean(axis=0)

### Replace NaN by mean value

df['bore'].replace(np.nan, avg_2, inplace= True)

"""
replace NaN in "stroke" column by mean.
"""
avg_3 = df["stroke"].astype("float").mean(axis = 0)
df["stroke"].replace(np.nan, avg_3, inplace = True)

"""
Calculate the mean value for the 'horsepower' column
Replace "NaN" by mean value

do the same for 'peak-rpm'
"""

avg_4=df['horsepower'].astype('float').mean(axis=0) 
df['horsepower'].replace(np.nan, avg_4, inplace= True)


avg_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)


"""
To see which values are present in a particular column,
 we can use the ".value_counts()" method
"""

df['num-of-doors'].value_counts()

"""
 We can also use the ".idxmax()" method 
 to calculate for us the most common type automatically
"""

df['num-of-doors'].value_counts().idxmax()

# then we replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)


"""
Finally, let's drop all rows that do not have price data

# simply drop whole row with NaN in "price" column
"""

df.dropna(subset=["price"], axis=0, inplace = True)

# reset index, because we droped two rows (the above line do it for us)

df.head()

"""
Now we have a dataset with no missing values

Then the next step (The last step) in data cleaning is checking and
making sure that all data is in the correct format (int, float, text or other).
 
In Pandas, we use

.dtype() to check the data type

.astype() to change the data type

"""

# Lets list the data types for each column

df.dtypes

"""
if we notice, some columns are not of the correct data type. 
Numerical variables should have type 'float' or 'int', and 
variables with strings such as categories should have type 'object'.
 For example, 'bore' and 'stroke' variables are numerical values 
 that describe the engines, so we should expect them to be of the
 type 'float' or 'int'; however, they are shown as type 'object'.
 We have to convert data types into a proper format for each 
 column using the "astype()" method.
 """

# Convert data types to proper format
 
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")


df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")


df[["price"]] = df[["price"]].astype("float")


df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


print("Done")



# Let us list the columns after the conversion

df.dtypes


"""
Now, we finally obtain the cleaned dataset with no missing values
 and all data in its proper format.

"""



"""
DATA STANDARDIZATION

"""


"""
Data is usually collected from different agencies with different formats.

(Data Standardization is also a term for a particular
 type of data normalization,where we subtract the mean 
 and divide by the standard deviation)


What is Standardization?

Standardization is the process of transforming data into 
a common format which allows the researcher to make the meaningful comparison.

    For Example

    Transform mpg to L/100km:
    
  In our dataset, the fuel consumption columns "city-mpg" and "highway-mpg"
 are represented by mpg (miles per gallon) unit.
 
 
 Assume we are developing an application in a country that accept 
 the fuel consumption with L/100km standard
 
 
 
We will need to apply data transformation to transform mpg into L/100km?


The formula for unit conversion is L/100km = 235 / mpg

"""


# Note:  many mathematical operations can be done directly in Pandas.

# here is the data transformation code

df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()


# let  rename column name from "highway-mpg" to "highway-L/100km"

"""
    This is done with use of the 'rename method'

    Notably, rename can be used in conjunction with a dict-like object
    providing new values for a subset of the axis labels.
        
        
"""
# here is the code:

df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# Note: Should you wish to modify a dataset in-place, (we pass inplace=True)


# check your transformed data 
df.head()


"""
       DATA NORMALIZATION
       
"""

"""
Why normalization?

Normalization is the process of transforming values of several
 variables into a similar range. Typical normalizations include 
 scaling the variable so the variable average is 0, 
 scaling the variable so the variable variance is 1,
 or scaling variable so the variable values range from 0 to 1
 
"""

"""
let normalize the column 'length', 'width' and 'height' so that their
value range from 0 t0 1
"""

# replace (origianl value) by (original value)/(maximum value)

df['length'] = df['length']/df['length'].max()

df['width'] = df['width']/df['width'].max()

df['height'] = df['height']/df['height'].max()

# show the scaled columns
df[["length","width","height"]].head()



"""
        BINNING

"""

"""
    Why binning?

Binning is a process of transforming continuous numerical variables
 into discrete categorical 'bins', for grouped analysis.
 
 Continuous data is often discretized or otherwise 
 separated into “bins” for analysis. 
 
  For Example:

In our dataset, "horsepower" is a real valued variable ranging 
from 48 to 288, it has 57 unique values.
 
What if we only care about the price difference between cars
 with high horsepower, medium horsepower, and little horsepower (3 types)?
 Can we rearrange them into three ‘bins' to simplify analysis?
 
We will use the Pandas function 'cut' to segment
 the 'horsepower' column into 3 bins
 
"""

# let Convert the data to correct format
# if you note 'horsepower' is an object type

df["horsepower"]=df["horsepower"].astype(float, copy=True)

"""

We would like four bins of equal size bandwidth,
the forth is because the function "cut" include the rightmost edge

"""

binwidth = (max(df["horsepower"]) - min(df["horsepower"]))/4

"""
We build a bin array, with a minimum value to a maximum value,
 with bandwidth calculated above. The bins will be values used 
 to determine when one bin ends and another begins.

 we shall use the 'arange' function in numpy
"""

bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)

# let view the output of the bins
bins


"""
 we can also pass your own bin names by passing a list or array 
  to the labels option:
"""

# We set group names as 


group_names = ['Low', 'Medium', 'High']

# let apply the function "cut" 


df['horsepower-binned'] = pd.cut(df['horsepower'], 
  bins, labels=group_names,include_lowest=True )

# then let determine 
# what each value of "df['horsepower']" belongs to.

df[['horsepower','horsepower-binned']].head(20)

# We successfully narrow the intervals from 57 to 3!


"""

Bins visualization

"""

"""
Normally, a histogram is used to visualize 
the distribution of bins we created above.
"""

# let visualize it:

# %matplotlib inline (this is use to plot in jupter notebook)

import matplotlib as plt

from matplotlib import pyplot

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
# and set x/y labels and plot title

plt.pyplot.hist(df["horsepower"], bins = 3)
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


"""

  INDICATOR VARIABLE

"""

"""
What is an indicator variable?

An indicator variable (or dummy variable) is a 
numerical variable used to label categories.

  Why are they called dummies?
  
They are called 'dummies' because the numbers
 themselves don't have inherent meaning.
They are only used to turn categorical variables into
quantitative variables in Python

"""


"""

Why we use indicator variables?

So that we can use categorical variables for regression
 analysis in the later modules because most statistical models
 cannot take in objects/strings as input
 
 For Example

We see the column "fuel-type" has two unique values,
 "gas" or "diesel". Regression doesn't understand words, only numbers. 
 To use this attribute in regression analysis, we convert "fuel-type" 
 into indicator variables.
 
We will use the panda's method 'get_dummies' to assign
 numerical values to different categories of fuel type.
 
"""
 
df.columns

""" 
let get indicator variables and assign it to data frame "dummy_variable_1"


"""

dummy_variable_1 = pd.get_dummies(df["fuel-type"])

dummy_variable_1.head()

# let change column names for clarity

dummy_variable_1.rename(columns={'gas' :'fuel-type-gas', 
                                 'diesel':'fuel-type-diesel'}, inplace=True)
    
# let check it again
dummy_variable_1.head()

"""
We now have the value 1 to represent "gas" and 0 to represent "diesel" in the
 column "fuel-type". 
 We will now insert this column back into our original dataset.
 
"""

"""
 let merge data frame "df" and "dummy_variable_1" 
 using one of the join and merge method

 we shall use the 'concat' method

"""

df = pd.concat([df, dummy_variable_1], axis=1)

# let drop original column "fuel-type" from "df"

df.drop("fuel-type", axis = 1, inplace=True)

df.head()

"""
let create indicator variable to the column 
of "aspiration": "std" to 0, while "turbo" to 1.
"""


# let get indicator variables of aspiration and 
# assign it to data frame "dummy_variable_2"

dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity

dummy_variable_2.rename(columns={'std':'aspiration-std', 
                                 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


"""
let Merge the new dataframe to the original dataframe then
 drop the column 'aspiration'
 
"""

#merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

"""

LET SAVE THE NEW DATAFRAME(CSV)

"""

#df.to_csv('clean_df.csv')