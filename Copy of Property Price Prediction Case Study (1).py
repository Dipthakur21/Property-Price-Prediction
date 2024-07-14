#!/usr/bin/env python
# coding: utf-8

# <table align="center" width=100%>
#     <tr>
#         <td width="15%">
#             <img src="house.jpg">
#         </td>
#         <td>
#             <div align="center">
#                 <font color="#21618C" size=24px>
#                     <b>Property Price Prediction
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# ## Problem Statement
# 
# A key challenge for property sellers is to determine the sale price of the property. The ability to predict the exact property value is beneficial for property investors as well as for buyers to plan their finances according to the price trend. The property prices depend on the number of features like the property area, basement square footage, year built, number of bedrooms, and so on. Regression analysis can be useful in predicting the price of the house.

# ## Data Definition
# 
# **Dwell_Type:** Identifies the type of dwelling involved in the sale
# 
# **Zone_Class:** Identifies the general zoning classification of the sale
# 	
# **LotFrontage:** Linear feet of street-connected to the property
# 
# **LotArea:** Lot size is the lot or parcel side where it adjoins a street, boulevard or access way
# 
# **Road_Type:** Type of road access to the property
#        	
# **Alley:** Type of alley access to the property
# 		
# **Property_Shape:** General shape of the property
# 
# **LandContour:** Flatness of the property
# 
# **LotConfig:** Lot configuration
# 	
# **LandSlope:** Slope of property
# 
# **Neighborhood:** Physical locations within Ames city limits
# 			
# **Condition1:** Proximity to various conditions
# 
# **Condition2:** Proximity to various conditions (if more than one is present)
# 	
# **Dwelling_Type:** Type of dwelling
# 	
# **HouseStyle:** Style of dwelling
# 
# **OverallQual:** Rates the overall material and finish of the house
# 	
# **OverallCond:** Rates the overall condition of the house
# 		
# **YearBuilt:** Original construction date
# 
# **YearRemodAdd:** Remodel date (same as construction date if no remodeling or additions)
# 
# **RoofStyle:** Type of roof
# 
# **RoofMatl:** Roof material
# 		
# **Exterior1st:** Exterior covering on the house
# 	
# **Exterior2nd:** Exterior covering on the house (if more than one material)
# 
# **MasVnrType:** Masonry veneer type
# 
# **MasVnrArea:** Masonry veneer area in square feet
# 
# **ExterQual:** Evaluates the quality of the material on the exterior
# 
# **ExterCond:** Evaluates the present condition of the material on the exterior
# 		
# **Foundation:** Type of foundation
# 		
# **BsmtQual:** Evaluates the height of the basement
# 		
# **BsmtCond:** Evaluates the general condition of the basement
# 	
# **BsmtExposure:** Refers to walkout or garden level walls
# 
# **BsmtFinType1:** Rating of basement finished area
# 		
# **BsmtFinSF1:** Type 1 finished square feet
# 
# **BsmtFinType2:** Rating of basement finished area (if multiple types)
# 
# **BsmtFinSF2:** Type 2 finished square feet
# 
# **BsmtUnfSF:** Unfinished square feet of the basement area
# 
# **TotalBsmtSF:** Total square feet of the basement area
# 
# **Heating:** Type of heating
# 		
# **HeatingQC:** Heating quality and condition
# 		
# **CentralAir:** Central air conditioning
# 
# **Electrical:** Electrical system
# 		
# **1stFlrSF:** First Floor square feet
#  
# **2ndFlrSF:** Second floor square feet
# 
# **LowQualFinSF:** Low quality finished square feet (all floors)
# 
# **GrLivArea:** Above grade (ground) living area square feet
# 
# **BsmtFullBath:** Basement full bathrooms
# 
# **BsmtHalfBath:** Basement half bathrooms
# 
# **FullBath:** Full bathrooms above grade
# 
# **HalfBath:** Half baths above grade
# 
# **Bedroom:** Bedrooms above grade (does NOT include basement bedrooms)
# 
# **Kitchen:** Kitchens above grade
# 
# **KitchenQual:** Kitchen quality
# 
# **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)
# 
# **Functional:** Home functionality (Assume typical unless deductions are warranted)
# 
# **Fireplaces:** Number of fireplaces
# 
# **FireplaceQu:** Fireplace quality
# 
# **GarageType:** Garage location
# 		
# **GarageYrBlt:** Year garage was built
# 		
# **GarageFinish:** Interior finish of the garage
# 
# **GarageCars:** Size of garage in car capacity
# 
# **GarageArea:** Size of garage in square feet
# 
# **GarageQual:** Garage quality
# 		
# **GarageCond:** Garage condition
# 		
# **PavedDrive:** Paved driveway
# 		
# **WoodDeckSF:** Wood deck area in square feet
# 
# **OpenPorchSF:** Open porch area in square feet
# 
# **EnclosedPorch:** Enclosed porch area in square feet
# 
# **3SsnPorch:** Three season porch area in square feet
# 
# **ScreenPorch:** Screen porch area in square feet
# 
# **PoolArea:** Pool area in square feet
# 
# **PoolQC:** Pool quality
# 		
# **Fence:** Fence quality
# 		
# **MiscFeature:** Miscellaneous feature not covered in other categories
# 		
# **MiscVal:** Value of miscellaneous feature
# 
# **MoSold:** Month Sold (MM)
# 
# **YrSold:** Year Sold (YYYY)
# 
# **SaleType:** Type of sale
# 
# **SaleCondition:** Condition of sale
#        
# **Property_Sale_Price:** Price of the house

# ## Icon Legends
# <table>
#   <tr>
#     <th width="25%"> <img src="infer.png" style="width:25%;"></th>
#     <th width="25%"> <img src="alsoreadicon.png" style="width:25%;"></th>
#     <th width="25%"> <img src="todo.png" style="width:25%;"></th>
#     <th width="25%"> <img src="quicktip.png" style="width:25%;"></th>
#   </tr>
#   <tr>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Inferences from outcome</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Additional Reads</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Lets do it</b></font></div>
#     </td>
#     <td><div align="center" style="font-size:120%">
#         <font color="#21618C"><b>Quick Tips</b></font></div>
#     </td>
# 
# </tr>
# 
# </table>

# ## Table of Contents
# 
# 1. **[Import Libraries](#import_lib)**
# 2. **[Set Options](#set_options)**
# 3. **[Read Data](#Read_Data)**
# 4. **[Prepare and Analyze the Data](#data_preparation)**
#     - 4.1 - [Understand the Data](#Data_Understanding)
#         - 4.1.1 - [Data Type](#Data_Types)
#         - 4.1.2 - [Summary Statistics](#Summary_Statistics)
#         - 4.1.3 - [Distribution of Variables](#distribution_variables)
#         - 4.1.4 - [Discover Outliers](#outlier)
#         - 4.1.5 - [Missing Values](#Missing_Values)
#         - 4.1.6 - [Correlation](#correlation)
#         - 4.1.7 - [Analyze Relationships Between Target and Categorical Variables](#cat_num)
#     - 4.2 - [Data Preparation](#Data_Preparation)
#         - 4.2.1 - [Check for Normality](#Normality)
#         - 4.2.2 - [Dummy Encode the Categorical Variables](#dummy)
# 5. **[Linear Regression (OLS)](#LinearRegression)**
#     - 5.1 - [Multiple Linear Regression Full Model with Log Transformed Dependent Variable (OLS)](#withLog)
#     - 5.2 - [Multiple Linear Regression Full Model without Log Transformed Dependent Variable (OLS)](#withoutLog)
#     - 5.3 - [Feature Engineering](#Feature_Engineering)
#       - 5.3.1 - [Multiple Linear Regression (Using New Feature) - 1](#feature1)
#       - 5.3.2 - [Multiple Linear Regression (Using New Feature) - 2](#feature2)
# 6. **[Feature Selection](#feature_selection)**
#      - 6.1 - [Variance Inflation Factor](#vif)
# 
# 7. **[Conclusion and Interpretation](#conclusion)**

# <a id='import_lib'></a>
# # 1. Import Libraries

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Import the required libraries and functions</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


# pd.api.types.is_numeric_dtype
# is_numeric_dtype


# In[ ]:


import pandas as pd 


# In[ ]:


# We use 'Numpy'for mathematical operations on large, multi-dimensional arrays and matrices
# 'Pandas' is used for data manipulation and analysis
import numpy as np
import pandas as pd 

# To check the data type we import 'is_string_dtype' and 'is_numeric_dtype'
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

#  To build and analyze various statistical models we use 'Statsmodels'
import statsmodels
import statsmodels.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing

# To perform scientific computations
from scipy.stats import shapiro
from scipy import stats

# 'Matplotlib' is a data visualization library for 2D and 3D plots
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# seaborn is used for plotting statistical graphics
import seaborn as sns


# In[ ]:


# Do not run this cell and look at the warnings for learning purpose
# suppress the warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# set the plot size using 'rcParams'
# once the plot size is set using 'rcParams', it sets the size of all the forthcoming plots in the file
# 15 and 8 are width and height in inches respectively
plt.rcParams['figure.figsize'] = [15,8]


# <a id='set_options'></a>
# # 2. Set Options

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>
# 1. Display complete data frames<br>
# 2. To avoid the exponential number<br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None

# use below code to convert the 'exponential' values to float
np.set_printoptions(suppress=True)


# <a id='Read_Data'></a>
# # 3. Read Data

# In[ ]:


#Do not copy this cell if writing code in your local machine
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# read csv file using pandas
df_property = pd.read_csv('/content/drive/MyDrive/imarticus_classrooms/PGAA-online-01/OLS/Property Price Prediction/Dataset/HousePrices.csv')

# display the top 5 rows of the dataframe
df_property.head()


# In[ ]:


df_property


# #### Lets take a glance at our dataframe and see how it looks

# #### Dimensions of the data

# In[ ]:


# 'shape' function returns a tuple that gives the total number of rows and columns in the data
df_property.shape


# In[ ]:


# Before applying any machine learning algorithm, the data should be clean and in proper format.
#handle missing values (delete/imterpolate..)
#Handle outliers
#handle categorical variables


# In[ ]:


#Preprocessing/EDA(Exploratory data analysis)
#EDA might or might not directly help us building OLS solution


# <a id='data_preparation'></a>
# # 4. Data Analysis and Preparation

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Data preparation is the process of cleaning and transforming raw data before building predictive models. <br><br>
#                         Here, we analyze and perform the following tasks:<br>
#                         1. Check data types. Ensure your data types are correct.<br>
#                         2. We need to change the data types as per requirement If they are not as per business definition <br>
#                         3. Go through the summary statistics<br>
#                         4. Distribution of variables<br>
#                         5. Study the correlation<br>
#                         6. Detect outliers from the data<br>
#                         7. Look for the missing values<br><br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Understanding'></a>
# ## 4.1 Understand the Dataset

# <a id='Data_Types'></a>
# ### 4.1.1 Data Type
# The data types in pandas dataframes are the object, float, int64, bool, and datetime64. We should know the data type of each column.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>In our dataset, we have a blend of numerical and categorical variables. The numeric variables should have data type 'int'/'float' while categorical variables should have data type 'object'.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Check for the data type**

# In[ ]:


type(df_property.dtypes)


# In[ ]:


# 'dtypes' provides the data type for each column
df_property.dtypes


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>From the above output, we can see that 'Dwell_Type', 'OverallQual' and 'OverallCond' have data type as 'int64'.<br>
# 
# But as per the data definition, 'Dwell_Type ', 'OverallQual' and 'OverallCond' are categorical variables, so we need to convert these variables data type to 'object'.</br></b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# We decide a particular column to be categorical or contineous primarily from the domain/biz knowledge and optionally from the pandas datatype of the column


# #### Let us convert  'Dwell_Type ', 'OverallQual' and 'OverallCond'  to categorical data type

# In[ ]:


df_property['Dwell_Type'] = df_property['Dwell_Type'].astype('O')
df_property['OverallQual'] = df_property['OverallQual'].astype('O')
df_property['OverallCond'] = df_property['OverallCond'].astype('O')


# #### Let us now remove the Id column as this will not be necessary for our analysis

# In[ ]:


df_property.drop(['Id'], axis=1, inplace=True)


# <a id='Summary_Statistics'></a>
# ### 4.1.2 Summary Statistics
# 
# Here we take a look at the summary of each attribute. This includes the count, mean, the minimum and maximum values as well as some percentiles for numeric variables and count, unique, top, frequency for categorical variables.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> In our dataset we have numerical as well as categorical variables. Now we check for summary statistics of all the variables.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. For getting the statistical summary of numerical variables we use the describe()**

# In[ ]:


# by default the describe function returns the summary of numerical variables
df_property.describe()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
# <b>The above output displays the summary statistics of all the numeric variables like mean, median, standard deviation, minimum, and the maximum values, the first and third quantiles.<br><br> 
#     We can see that the LotFrontage ranges from 21 feet to 313 feet, with mean 70 feet. 
#     We can see that the minimum pool area is 0 sq.ft. and this means that not all houses have pools and yet have been considered to calculate the mean pool area. Also the count for LotFrontage is less than the total number of observations which indicates the presence of missing values.
#     </b>     </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. For getting the statistical summary of categorical features we use the describe(include = object)**

# In[ ]:


# summary of categorical variables
df_property.describe(include = object)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The summary statistics for categorical variables contains information about the total number of observations, number of unique classes, the most occurring class, and its frequency.:<br><br> 
#                         Lets understand the outputs of the above table using variable 'Property_Shape' <br> 
#                         count: Number of observations = 2073 <br> 
#                         unique: Number of unique classes in the column = 4 classes<br>  
#                         top: The most occurring class = Reg<br>
#                         frequency: Frequency of the most repeated class; out of 2073 observations Reg has a frequency of 925 <br> It is visible that some of the variables have count less than total number of observations which indicates the presence of missing values.</b>  
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# As, the variable `PoolQC` has only 8 non-zero values out of 2073 observations. And also the variable `PoolArea` contains the area of these 8 pools, we will remove the variables `PoolQC` and `PoolArea`.

# In[ ]:


# use drop() to drop the redundant variables
# 'axis = 1' drops the corresponding columns
df_property = df_property.drop(['PoolQC', 'PoolArea'], axis= 1)

# re-check the shape of the dataframe
df_property.shape


# <a id='distribution_variables'></a>
# ### 4.1.3 Distribution of Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Check the distribution of all the variables.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Distribution of numeric variables**
# 
# We plot the histogram to check the distribution of the variables.

# In[ ]:


# filter the numerical features in the dataset using select_dtypes()
# include=np.number is used to select the numeric features
df_numeric_features = df_property.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[ ]:


# plot the histogram of numeric variables
# hist() by default considers the numeric variables only, 
# rotate the x-axis labels by 20 degree using the parameter, 'xrot'
df_property.hist(xrot = 20, color = "maroon")

# adjust the subplots
plt.tight_layout()

# display the plot
plt.show()  


# #### Visualize the target variable

# In[ ]:


# Sale Price Frequency Distribution
# set the xlabel and the fontsize
plt.xlabel("Sale Price", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Frequency", fontsize=15)

# set the title of the plot
plt.title("Frequency Distribution", fontsize=15)

# plot the histogram for the target variable
plt.hist(df_property["Property_Sale_Price"], color = 'maroon')
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The above plot shows that the target variable 'Property_Sale_Price' is right skewed. 
#                     </b>   
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# **2. Distribution of categorical variables**

# For the categoric variables, we plot the countplot

# In[ ]:


# create an empty list to store all the categorical variables
categorical=[]

# check the data type of each variable
for column in df_property:

    # check if the variable has the categorical type 
    if is_string_dtype(df_property[column]):
        
        # append the categorical variables to the list 'categorical'
        categorical.append(column)

# plot the count plot for each categorical variable 
# 'figsize' sets the figure size
fig, ax = plt.subplots(nrows=7, ncols=6, figsize = (50, 35))

# plot the count plot using countplot() for each categorical variable
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(df_property[variable], ax = subplot)
    
# display the plot
plt.show()


# #### Boxplot of OverallQuality and Property_Sale_Price

# In[ ]:


# draw the boxplot for OverallQuality and the Property_Sale_Price
sns.boxplot(y="Property_Sale_Price", x="OverallQual", data= df_property)

# set the title of the plot and the fontsize
plt.title("Overall Quality vs Property_Sale_Price", fontsize=15)

# set the xlabel and the fontsize
plt.xlabel("Overall Quality", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Sale Price", fontsize=15)

# display the plot
plt.show()


# #### Boxplot of Overall Condition and Property_Sale_Price

# In[ ]:


# draw the boxplot for OverallQuality and the Property_Sale_Price
sns.boxplot(y="Property_Sale_Price", x="OverallCond", data= df_property)

# set the title of the plot and the fontsize
plt.title("Overall Condition vs Property_Sale_Price", fontsize=15)

# set the xlabel and the fontsize
plt.xlabel("Overall Condition", fontsize=15)

# set the ylabel and the fontsize
plt.ylabel("Sale Price", fontsize=15)

# display the plot
plt.show()


# #### Draw the pairplot of the numeric variables

# In[ ]:


# Pairplot of numeric variables

# select the columns for the pairplot
columns= ["Property_Sale_Price", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd"]

# draw the pairplot such that the diagonal should be density plot and the other graphs should be scatter plot
sns.pairplot(df_property[columns], size=2, kind= "scatter", diag_kind="kde")

# display the plot
plt.show()


# <a id='outlier'></a>
# ### 4.1.4 Outliers Discovery

# In[ ]:


# plot a boxplot of target variable to detect the outliers
sns.boxplot(df_property['Property_Sale_Price'], color='maroon')

# set plot label
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Property_Sale_Price)', fontsize = 15)

# display the plot
plt.show()


# From the above plot we can see that there are outliers present in the target variable 'Property_Sale_Price'. Outliers badly affect the prediction of the regression model and thus, we will remove these outliers.

# In[ ]:


# remove the observations with the house price greater than or equal to 500000
df_property = df_property[df_property['Property_Sale_Price'] < 500000]

# check the dimension of the data
df_property.shape


# <a id='Missing_Values'></a>
# ### 4.1.5 Missing Values
# 
# If we do not handle the missing values properly then we may end up drawing an inaccurate inference about the data.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Look for the missing values and handle the missing values separately for numerical and categorical values.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **Look for the missing values**

# In[ ]:


# 'isnull().sum()' returns the number of missing values in each variable
# 'ascending = False': sorts values in the descending order
total_nulls = df_property.isnull().sum().sort_values(ascending = False)          

# calculate the percentage of missing values
# 'ascending = False' sorts values in the descending order
percent_null = (df_property.isnull().sum()*100/df_property.isnull().count())  
percent_null = percent_null.sort_values(ascending = False) 

# concat the 'total_nulls' and 'percent_null' columns
# 'axis = 1' stands for columns
missing_values = pd.concat([total_nulls, percent_null], axis = 1, keys = ['Total Nulls', 'Percentage of Missing Values'])    

# add the column containing data type of each variable
missing_values['Data Type'] = df_property[missing_values.index].dtypes
missing_values


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We can see that 18 variables contain the missing values.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **Handle the missing values for numerical variables**

# In[ ]:


# filter out the categorical variables and consider only the numeric variables with missing values
num_missing_values = missing_values[(missing_values['Total Nulls'] > 0) & (missing_values['Data Type'] != 'object')]
num_missing_values


# 

# For the numerical variables, we can replace the missing values by their mean, median or mode as per the requirement.

# #### The variable 'LotFrontage' is right skewed and thus we will fill the missing values with its median value

# In[ ]:


# use the function fillna() to fill the missing values
df_property['LotFrontage'] = df_property['LotFrontage'].fillna(df_property['LotFrontage'].median())


# #### We will replace the missing values in the numeric variable `GarageYrBlt` by 0. The missing values in this variable indicates that there are 81 (check it's 113)observations for which garage facility is not available.

# In[ ]:


# use the function fillna() to replace missing values in 'GarageYrBlt' with 0 
df_property['GarageYrBlt'] = df_property['GarageYrBlt'].fillna(0)


# #### The variable 'MasVnrArea' is positively skewed and thus we will fill the missing values with its median value

# In[ ]:


# use the function fillna() to fill the missing values
df_property['MasVnrArea'] = df_property['MasVnrArea'].fillna(df_property['MasVnrArea'].median())


# **Handle the missing values for categorical variables**

# In[ ]:


# filter out the numerical variables and consider only the categorical variables with missing values
cat_missing_values = missing_values[(missing_values['Total Nulls'] > 0) & (missing_values['Data Type'] == 'object')]
cat_missing_values


# In[ ]:


# according to the data definition, 'NA' denotes the absence of miscellaneous feature
# replace NA values in 'MiscFeature' with a valid value, 'None'
df_property['MiscFeature'] = df_property['MiscFeature'].fillna('None')

# replace NA values in 'Alley' with a valid value, 'No alley access' 
df_property['Alley'] = df_property['Alley'].fillna('No alley access')

# replace NA values in 'Fence' with a valid value, 'No Fence'
df_property['Fence'] = df_property['Fence'].fillna('No Fence')

# replace null values in 'FireplaceQu' with a valid value, 'No Fireplace' 
df_property['FireplaceQu'] = df_property['FireplaceQu'].fillna('No Fireplace')


# In[ ]:


# replace the missing values in the categoric variables representing the garage by `No Garage`
for col in ['GarageType', 'GarageFinish', 'GarageCond', 'GarageQual']:
    df_property[col].fillna('No Garage', inplace = True)


# In[ ]:


# according to the data definition, 'NA' denotes the absence of basement in the variabels 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2'
# replace the missing values in the categoric variables representing the basement by `No Basement`
for col in ['BsmtFinType2', 'BsmtExposure', 'BsmtQual','BsmtCond','BsmtFinType1']:
    df_property[col].fillna('No Basement', inplace = True)


# In[ ]:


# according to the data definition, 'NA' denotes the absence of masonry veneer
# replace the missing values in the categorical variable 'MasVnrType' with a value, 'None'
df_property['MasVnrType'] = df_property['MasVnrType'].fillna('None')


# In[ ]:


# replace the missing values in the categorical variable 'Electrical' with its mode
mode_electrical = df_property['Electrical'].mode()
df_property['Electrical'].fillna(mode_electrical[0] , inplace = True)


# <a id='correlation'></a>
# ### 4.1.6 Study correlation

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> To check the correlation between numerical variables, compute a correlation matrix and plot a heatmap for the correlation matrix
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **Compute a correlation matrix**

# In[ ]:


# use the corr() function to generate the correlation matrix of the numeric variables
corrmat = df_property.corr()

# print the correlation matrix
corrmat


# **2. Plot the heatmap for the diagonal correlation matrix**

# A correlation matrix is a symmetric matrix. Plot only the upper triangular entries using a heatmap.

# In[ ]:


# set the plot size
plt.figure(figsize = (35,25))

# plot the heat map
# corr: give the correlation matrix
# cmap: color code used for plotting
# annot_kws: sets the font size of the annotation
# annot: prints the correlation values in the chart
# vmax: gives a maximum range of values for the chart
# vmin: gives a minimum range of values for the chart
sns.heatmap(corrmat, annot = True, vmax = 1.0, vmin = -1.0, cmap = 'bwr', annot_kws = {"size": 11.5})

# set the size of x and y axes labels using 'fontsize'
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>The diagonal represents the correlation of the variable with itself thus all the diagonal entries are '1'. The dark red squares represent the variables with strong positive correlation. <br><br>From the above plot we can see that the highest positive correlation (= 0.88) is between the variables 'GarageArea' and 'GarageCars'. Also there is strong positive correlation between the pairs (1StFlrSF, TotalBsmtSF) and (TotRmsAbvGrd, GrlivArea). There may be multicollinearity present.<br>
#                         No two variables have strong negative correlation in the dataset.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Correlation does not imply causation. In other words, if two variables are correlated, it does not mean that one variable caused the other.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="alsoreadicon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Want to know more?</b> <br><br>
#                     <a href="https://bit.ly/2PBvA8T">Why correlation does not imply causation </a>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# <a id='cat_num'></a>
# ### 4.1.7 Analyze Relationships Between Target and Categorical Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Plot the box-and-whisker plot for visualizing relationships between target and categorical variables.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


# create an empty list to store all the categorical variables
categorical=[]

# check the data type of each variable
for column in df_property:

    # check if the variable has the categorical type 
    if is_string_dtype(df_property[column]):
        
        # append the categorical variables to the list 'categorical'
        categorical.append(column)

# plot the count plot for each categorical variable 
# 'figsize' sets the figure size
fig, ax = plt.subplots(nrows = 14, ncols = 3, figsize = (40, 100))

# plot the boxplot for each categoric and target variable
for variable, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x = variable, y = 'Property_Sale_Price', data = df_property, ax = subplot)
    
# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be seen that most of the categorical variables have an effect on the sale price of the property. The median sale price rises exponentially with respect to the rating of the overall quality of the material used.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <a id='Data_Preparation'></a>
# ## 4.2 Data Preparation

# <a id='Normality'></a>
# ### 4.2.1 Check for Normality

# **Plot a histogram and also perform the Shapiro-Wilk test**

# We use the function `hist()` from the matplotlib library to plot a histogram.

# In[ ]:


# check the distribution of target variable
df_property.Property_Sale_Price.hist(color = 'maroon')

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Target Variable (Property_Sale_Price)', fontsize = 15)
plt.xlabel('Property Sale Price', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We can see that the variable 'Property_Sale_Price' is positively skewed and thus we can say that it is not normally distributed.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We should not only make conclusions through visual representations or only using a statistical test but perform multiple ways to get the best insights.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Let us perform from Shapiro-Wilk test to check the normality of the target variable.

# The null and alternate hypothesis of Shapiro-Wilk test is as follows: <br>
# 
# <p style='text-indent:25em'> <strong> H<sub>o</sub>: The data is normally distributed</strong> </p>
# <p style='text-indent:25em'> <strong> H<sub>1</sub>: The data is not normally distributed</strong> </p>

# In[ ]:


# shapiro() returns the the test statistics along with the p-value of the test
stat, p = shapiro(df_property.Property_Sale_Price)

# print the numeric outputs of the Shapiro-Wilk test upto 3 decimal places
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# set the level of significance (alpha) to 0.05
alpha = 0.05

# if the p-value is less than alpha print we reject alpha
# if the p-value is greater than alpha print we accept alpha 
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We can see that the p-value is less than 0.05 and thus we reject the null hypothesis. It can be concluded that the data is not normally distributed.<br><br>
#                         We need to log transform the variable 'Property_Sale_Price' in order to reduce the skewness.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Shaprio Wilk Test does not work if the number of observations are more than 5000. However Shapiro Wilk test is more robust than other tests. In case where the observations are more than 5000, other tests like Anderson Darling test or Jarque Bera test may also be used.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **If the data is not normally distributed, use log transformation to reduce the skewness and get a near normally distributed data**
# 
# The log transformation can be used to reduce the skewbess. To log transform the 'Property_Sale_Price' variable we use the function `np.log()`.

# In[ ]:


# log transformation using np.log()
df_property['log_Property_Sale_Price'] = np.log(df_property['Property_Sale_Price'])

# display the top 5 rows of the data
df_property.head()


# **Recheck for normality by plotting histogram and performing Shapiro-Wilk test**
# 
# Let us first plot a histogram of `log_Property_Sale_Price`.

# In[ ]:


# recheck for normality 
# plot the histogram using hist
df_property.log_Property_Sale_Price.hist(color = 'maroon')

# add plot and axes labels
# set text size using 'fontsize'
plt.title('Distribution of Log-transformed Target Variable (log_Property_Sale_Price)', fontsize = 15)
plt.xlabel('Sale Price (log-transformed)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

# display the plot
plt.show()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be seen that the variable log_Property_Sale_Price is near normally distributed. Lets confirm it again by using the Shapiro-Wilk test.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Let us perform Shapiro-Wilk test.

# In[ ]:


# shapiro() returns the the test statistics along with the p-value of the test
stat, p = shapiro(df_property['log_Property_Sale_Price'])

# print the numeric outputs of the Shapiro-Wilk test upto 3 decimal places
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# set the level of significance (alpha) to 0.05
alpha = 0.05

# if the p-value is less than alpha print we reject alpha
# if the p-value is greater than alpha print we accept alpha 
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# In[ ]:


# find the skewness of the variable log_Property_Sale_Price
df_property['log_Property_Sale_Price'].skew()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be visually seen that the data has near-normal distribution, but Shapiro-Wilk test does not support the claim.
# <br>                    
# Note that in reality it might be very tough for your data to adhere to all assumptions your algorithm needs.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='dummy'></a>
# ### 4.2.2 Dummy Encode the Categorical Variables

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> We need to perform dummy encoding on our categorical variables before we proceed; since the method of OLS works only on the numeric data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **Filter numerical and categorical variables**

# In[ ]:


# filter out the categorical variables and consider only the numeric variables using (include=np.number)
df_numeric_features = df_property.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[ ]:


# filter out the numerical variables and consider only the categorical variables using (include=object)
df_categoric_features = df_property.select_dtypes(include = object)

# display categorical features
df_categoric_features.columns


# **Dummy encode the catergorical variables**

# In[ ]:


# to create the dummy variables  we use 'get_dummies()' from pandas 
# to create (n-1) dummy variables we use 'drop_first = True' 
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables column-wise
df_property_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_property_dummy.head()


# In[ ]:


# check the dimensions of the dataframe
df_property_dummy.shape


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>There are various forms of encoding like n-1 dummy encoding, one hot encoding, label encoding, frequency encoding.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="alsoreadicon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Want to know more?</b> <br><br>
#                     <a href="https://bit.ly/36nZQKg">1. FAQ: What is Dummy Coding? <br>
#                     <a href="https://bit.ly/2q9Omt9">2. Encoding Categorical Features
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We will now train models by fitting a linear regression model using the method of ordinary least square(OLS). </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='LinearRegression'></a>
# # 5. Linear Regression (OLS)

# <a id='withLog'></a>
# ## 5.1 Multiple Linear Regression Full Model with Log Transformed Dependent Variable (OLS)

# #### Follow the steps in order to build the OLS model:

# **1. Split the data into training and test sets**

# As the OLS function does not include the intercept term by default. Thus, we add the intercept column to the dataset.

# In[ ]:


# add the intercept column using 'add_constant()'
df_property_dummy = sm.add_constant(df_property_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
X = df_property_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_property_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train data and test data 
# what proportion of data should be included in test data is passed using 'test_size'
# set 'random_state' to get the same data each time the code is executed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **2. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_logmodel_full = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_logmodel_full.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains around 94% of the variation in dependent variable log_Property_Sale_Price. The Condition Number 1.05e+19 suggests that there is severe multicollinearity in the data. The Durbin-Watson test statistics is 1.885 i.e. close to 2.0 and thus it indicates that there is no autocorrelation. </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="quicktip.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Condition Number : Multicollinearity can be checkked by computing the condition number(CN). If condition number is between 100 and 1000, there is moderate multicollinearity, if condition number is less than 100, there is no multicollinearity and if condition number is greater 1000 there is severe multicollinearity in the data. <br><br>
#                         Durbin-Watson : The Durbin-Watson statistic will always have a value between 0 and 4. A value of 2.0 means that there is no autocorrelation detected in the sample. Values from 0 to less than 2 indicate positive autocorrelation and values from from 2 to 4 indicate negative autocorrelation.</b>     
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **3. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_logmodel_full_predictions = linreg_logmodel_full.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_logmodel_full_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **4. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_logmodel_full_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_logmodel_full_rsquared = linreg_logmodel_full.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_logmodel_full_rsquared_adj = linreg_logmodel_full.rsquared_adj 


# **5. Tabulate the results**

# In[ ]:


# create the result table for all accuracy scores
# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value
# create a list of column names
cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create a empty dataframe of the colums
# columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns = cols)

# compile the required information
linreg_logmodel_full_metrics = pd.Series({'Model': "Linreg full model with log of target variable ",
                     'RMSE':linreg_logmodel_full_rmse,
                     'R-Squared': linreg_logmodel_full_rsquared,
                     'Adj. R-Squared': linreg_logmodel_full_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_logmodel_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>We will also build a linear regression model without performing log transformation on the target variable.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='withoutLog'></a>
# ## 5.2 Multiple Linear Regression Full Model without Log Transformed Target Variable (OLS)

# In this section we build a full model with linear regression using OLS (Ordinary Least Square) technique. By full model we indicate that we consider all the independent variables that are present in the dataset.
# 
# In this case, we do not consider any kind of transformation on the dependent variable, we use the 'Property_Sale_Price' variable as it is.

# ####  We have already done train and test split while building the previous model.

# **1. Build model using sm.OLS().fit()**

# In[ ]:


# build a OLS model using function OLS()
# Property_Sale_Price is our target variable
# use fit() to fit the model on train data
linreg_nolog_model = sm.OLS(y_train['Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_nolog_model.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains around 94% of the variation in dependent variable Property_Sale_Price. The Durbin-Watson test statistics is 1.868 and indicates that there is no autocorrelation. The Condition Number 1.05e+19 suggests that there is severe multicollinearity in the data.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **2. Predict the values using test set**

# In[ ]:


# predict the 'Property_Sale_Price' using predict()
linreg_nolog_model_predictions = linreg_nolog_model.predict(X_test)


# **3. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_nolog_model_rmse = rmse(actual_Property_Sale_Price, linreg_nolog_model_predictions)

# calculate R-squared using rsquared
linreg_nolog_model_rsquared = linreg_nolog_model.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_nolog_model_rsquared_adj = linreg_nolog_model.rsquared_adj 


# **4. Tabulate the results**

# In[ ]:


# append the result table 
# compile the required information
linreg_nolog_model_metrics = pd.Series({'Model': "Linreg full model without log of target variable ",
                                                 'RMSE':linreg_nolog_model_rmse,
                                                 'R-Squared': linreg_nolog_model_rsquared,
                                                 'Adj. R-Squared': linreg_nolog_model_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_nolog_model_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>If we compare the results in the table we can see that the linreg model withh log of target variable is performing slightly better than the model without log of target variable. Thus we will continue with the target variable 'log_Property_Sale_Price'.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Let us perform feature engineering and take a look at building a linear regression full model by adding new features to the dataset. 
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Feature_Engineering'></a>
# ## 5.3 Feature Engineering

# It is the process of creating new features using domain knowledge of the data that provides more insight into the data. Let us create a few features from the existing dataset and build a regression model on the newly created data.

# <a id='feature1'></a>
# ### 5.3.1 Multiple Linear Regression (Using New Feature) - 1

# #### In order to build the model, we do the following:

# **1. Create a new feature by using variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', and 'GrLivArea'.**

# **Calculate the complete area of the house.**<br>
# Create a new variable `TotalSF` representing the total square feet area of the house by adding the area of the first floor, second floor, ground level and basement of the house.

# In[ ]:


# create a new variable 'TotalSF' using the variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', and 'GrLivArea'
df_property['TotalSF'] = df_property['TotalBsmtSF'] + df_property['1stFlrSF'] + df_property['2ndFlrSF'] + df_property['GrLivArea']

# as we have created a new variable using the variables 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF' and 'GrLivArea', we will remove them
# use 'drop()' to remove the variables
df_property = df_property.drop(["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea"], axis=1)


# In[ ]:


# filter out the categorical variables and consider only the numerical variables using (include=np.number)
df_numeric_features = df_property.select_dtypes(include=np.number)

# filter out the numerical variables and consider only the categorical variables using (include=object)
df_categoric_features = df_property.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables column-wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train data and test data 
# what proportion of data should be included in test data is specified using 'test_size'
# set 'random_state' to get the same data(rows with the exact same index) each time the code is executed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the target variable 'log_Property_Sale_Price'
# use fit() to fit the model on train data
linreg_feature_1_model = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_feature_1_model.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains around 94% of the variation in dependent variable log_Property_Sale_Price. The Condition Number 8.29e+18 suggests that there is severe multicollinearity in the data. The Durbin-Watson test statistics is 1.885 i.e. close to 2.0 and thus it indicates that there is no autocorrelation.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_feature_1_model_predictions = linreg_feature_1_model.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_feature_1_model_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_feature_1_model_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_feature_1_model_rsquared = linreg_feature_1_model.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_feature_1_model_rsquared_adj = linreg_feature_1_model.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_feature_1_model_metrics = pd.Series({'Model': "Linreg with new feature (TotalSF) ",
                                                'RMSE': linreg_feature_1_model_rmse,
                                                'R-Squared': linreg_feature_1_model_rsquared,
                                                'Adj. R-Squared': linreg_feature_1_model_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_feature_1_model_metrics, ignore_index = True)

# print the result table
result_tabulation


# <a id='feature2'></a>
# ### 5.3.2 Multiple Linear Regression (Using New Feature) - 2

# #### In order to build the model, we do the following:

# **1. Create two new feature by using variables 'Buiding_age' and 'Remodel_age'**

# In[ ]:


# 'datetime' is used to perform operations related to date and time
import datetime as dt

# 'now().year' returns the current year
current_year = int(dt.datetime.now().year)


# In[ ]:


# create 2 new variables 'Buiding_age' and 'Remoel_age' 
Buiding_age = current_year - df_property.YearBuilt
Remodel_age = current_year - df_property.YearRemodAdd


# In[ ]:


# append the newly created variables to the dataframe
df_property['Buiding_age'] = Buiding_age
df_property['Remodel_age'] = Remodel_age

# as we have added a new variable using the variables 'YearBuilt' and 'YearRemodAdd', we will drop them
# drop the variables using drop()
df_property = df_property.drop(['YearBuilt', 'YearRemodAdd'], axis=1)


# In[ ]:


# filter out the categorical variables and consider only the numerical variables using (include=np.number)
df_numeric_features = df_property.select_dtypes(include=np.number)

# filter out the numerical variables and consider only the categorical variables using (include=object)
df_categoric_features = df_property.select_dtypes(include = object)


# **Dummy encode the catergorical variables**

# In[ ]:


# use 'get_dummies()' from pandas to create dummy variables
# use 'drop_first = True' to create (n-1) dummy variables
dummy_encoded_variables = pd.get_dummies(df_categoric_features, drop_first = True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# concatenate the numerical and dummy encoded categorical variables column-wise
df_dummy = pd.concat([df_numeric_features, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
# axis=1: specifies that the column is to be dropped
X = df_dummy.drop(['Property_Sale_Price','log_Property_Sale_Price'], axis = 1)

# extract the target variable from the data set
y = df_dummy[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train data and test data 
# what proportion of data should be included in test data is specified using 'test_size'
# set 'random_state' to get the same data each time the code is executed  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a OLS model using the function OLS()
# consider the target variable "log_Property_Sale_Price" 
# use fit() to fit the model on train data
linreg_feature_2_model = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_feature_2_model.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains around 94% of the variation in dependent variable log_Property_Sale_Price. The Condition Number 8.32e+18 suggests that there is severe multicollinearity in the data. The Durbin-Watson test statistics is 1.885 and indicates that there is no autocorrelation.</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_feature_2_model_predictions = linreg_feature_2_model.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_feature_2_model_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_feature_2_model_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_feature_2_model_rsquared = linreg_feature_2_model.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_feature_2_model_rsquared_adj = linreg_feature_2_model.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_feature_2_model_metrics = pd.Series({'Model': "Linreg with new features (Building_age and Remodel_age)",
                                                'RMSE': linreg_feature_2_model_rmse,
                                                'R-Squared': linreg_feature_2_model_rsquared,
                                                'Adj. R-Squared': linreg_feature_2_model_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_feature_2_model_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>RMSE of the model with new features 'Building_age' and 'Remodel_age' is increased. The value of R-squared and aadjusted R-squared is same as the previous model.  
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='feature_selection'></a>
# # 6. Feature Selection

# <a id='vif'></a>
# ## 6.1 Variance Inflation Factor
# 
# The Variance Inflation Factor (VIF) is used to detect the presence of multicollinearity between the features. The value of VIF equal to 1 indicates that no features are correlated. We calculate the VIF of the numerical independent variables. VIF for the variable V<sub>i</sub> is given as:
# <p style='text-indent:29em'> <strong> VIF = 1 / (1 - R-squared)</strong>  </p><br>
# Where, R-squared is the R-squared of the regression model build by regressing one independent variable (say V<sub>i</sub>) on all the remaining independent variables (say V<sub>j</sub>, j ≠ i).

# In[ ]:


# consider the independent variables in the dataframe 'df_property' 
# remove the target variables 'Property_Sale_Price' and 'log_Property_Sale_Price' using drop() function
df_property_features = df_property.drop(['Property_Sale_Price', 'log_Property_Sale_Price'], axis = 1)

# filter out the categorical variables and consider only the numerical variables using (include=np.number)
df_numeric_features_vif = df_property_features.select_dtypes(include=[np.number])

# display the first five observations
df_numeric_features_vif.head()


# #### Calculate the VIF for each numeric variable.

# In[ ]:


# create an empty dataframe to store the VIF for each variable
vif = pd.DataFrame()

# calculate VIF using list comprehension 
# use for loop to access each variable 
# calculate VIF for each variable and create a column 'VIF_Factor' to store the values 
vif["VIF_Factor"] = [variance_inflation_factor(df_numeric_features_vif.values, i) for i in range(df_numeric_features_vif.shape[1])]

# create a column of variable names
vif["Features"] = df_numeric_features_vif.columns

# sort the dataframe based on the values of VIF_Factor in descending order
# 'reset_index' resets the index of the dataframe
# 'ascending = False' sorts the data in descending order
# 'drop = True' drops the index that was previously created
vif.sort_values('VIF_Factor', ascending = False).reset_index(drop = True)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> We can see that the variable 'YrSold' has the highest VIF. We will remove the variables having VIF greater than 10. We want to remove the variable for which the remaining variables explain more than 90% of the variation and thus we set the threshold to 10. The value of threshold is completely experimental i.e. it depends on the business requirements. One can choose the threshold  other than 10.
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:


# we will calculate the VIF for each numerical variable
for ind in range(len(df_numeric_features_vif.columns)):
    
    # create an empty dataframe
    vif = pd.DataFrame()

    # calculate VIF for each variable and create a column 'VIF_Factor' to store the values 
    vif["VIF_Factor"] = [variance_inflation_factor(df_numeric_features_vif.values, i) for i in range(df_numeric_features_vif.shape[1])]

    # create a column of feature names
    vif["Features"] = df_numeric_features_vif.columns

    # filter the variables with VIF greater than 10 and store it in a dataframe 'vif_more_than_10' 
    # one can choose the threshold other than 10 (it depends on the business requirements)
    vif_more_than_10 = vif[vif['VIF_Factor'] > 10]
    
    # if dataframe 'vif_more_than_10' is not empty, then sort the dataframe by VIF values
    # if dataframe 'vif_more_than_10' is empty (i.e. all VIF <= 10), then print the dataframe 'vif' and break the for loop using 'break' 
    # 'by' sorts the data using given variable(s)
    # 'ascending = False' sorts the data in descending order
    if(vif_more_than_10.empty == False):
        df_sorted = vif_more_than_10.sort_values(by = 'VIF_Factor', ascending = False)
    else:
        print(vif)
        break
    
    # if  dataframe 'df_sorted' is not empty, then drop the first entry in the column 'Features' from the numeric variables
    # select the variable using 'iloc[]'
    # 'axis=1' drops the corresponding column
    #  else print the final dataframe 'vif' with all values after removal of variables with VIF less than 10  
    if (df_sorted.empty == False):
        df_numeric_features_vif = df_numeric_features_vif.drop(df_sorted.Features.iloc[0], axis=1)
    else:
        print(vif)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> The above dataframe contains all the variables with VIF less than 10. 
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# #### In order to build the model, we do the following

# Now, let us build the model using the categorical variables and the numerical variables obtained from VIF. 

# **1. Concatenate numerical and dummy encoded categorical variables**

# In[ ]:


# lets consider the variables obtained from VIF
# use the dummy variables created previously
# concatenate the numerical and dummy encoded categorical variables
df_dummy = pd.concat([df_numeric_features_vif, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()


# **2. Split the data into train and test sets**

# In[ ]:


# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# consider independent variables
# create a copy of 'df_dummy' and store it as X
X = df_dummy.copy()

# extract the target variable from the data set
y = df_property[['Property_Sale_Price','log_Property_Sale_Price']]

# split data into train data and test data 
# what proportion of data should be included in test data is specified using 'test_size'
# set 'random_state' to get the same data each time the code is executed  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[ ]:


# build a full model using OLS()
# consider the target variable log_Property_Sale_Price 
# use fit() to fit the model on train data
linreg_vif_model = sm.OLS(y_train['log_Property_Sale_Price'], X_train).fit()

# print the summary output
print(linreg_vif_model.summary())


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This model explains around 93% of the variation in dependent variable log_Property_Sale_Price. The Condition Number 6.97e+18 suggests that there is severe multicollinearity in the data. The Durbin-Watson test statistics is 1.905 and indicates that there is no autocorrelation. </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# **4. Predict the values using test set**

# In[ ]:


# predict the 'log_Property_Sale_Price' using predict()
linreg_vif_model_predictions = linreg_vif_model.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Note that the predicted values are log transformed Property_Sale_Price. In order to get Property_Sale_Price values, we take the antilog of these predicted values by using the function np.exp()</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 

# In[ ]:


# take the exponential of predictions using np.exp()
predicted_Property_Sale_Price = np.exp(linreg_vif_model_predictions)

# extract the 'Property_Sale_Price' values from the test data
actual_Property_Sale_Price = y_test['Property_Sale_Price']


# **5. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[ ]:


# calculate rmse using rmse()
linreg_vif_model_rmse = rmse(actual_Property_Sale_Price, predicted_Property_Sale_Price)

# calculate R-squared using rsquared
linreg_vif_model_rsquared = linreg_vif_model.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_vif_model_rsquared_adj = linreg_vif_model.rsquared_adj 


# **6. Tabulate the results**

# In[ ]:


# append the accuracy scores to the table
# compile the required information
linreg_vif_model_metrics = pd.Series({'Model': "Linreg with VIF",
                                                'RMSE': linreg_vif_model_rmse,
                                                'R-Squared': linreg_vif_model_rsquared,
                                                'Adj. R-Squared': linreg_vif_model_rsquared_adj})

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_vif_model_metrics, ignore_index = True)

# print the result table
result_tabulation


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="infer.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> From the above table we can see that the linear regression with new features has the lowest RMSE value. Thus, it can be concluded that the linear regression model with new features can be used to predict the price of the house.
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:




