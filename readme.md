# Mall Customers DataSet 

* Number Of Customers **_200_**
* The Average Annual Income Per Customer **_60.56_**
* The Median Age Of Customers **_36_**
* The Standard Deviations For Age **_13.97_** And The Std For Annual Income **_26.26_**

# (2) Analysis
## Distribution

![Annual Distribution!](charts/annual_dist.png)

    Annual Income (K$) Skewness & Kurtosis  
    * Skewness:  0.322
    * Kurtosis: -0.098

![Age Distribution!](charts/age_dist.png)

    Age Skewness & Kurtosis
    * Skewness:  0.486
    * Kurtosis: -0.672


![Spending Dist!](charts/spending_dist.png)

    Spending Score Skewness & Kurtosis 
    Skewness: -0.047
    Kurtosis: -0.827

![Gender Distribution!](charts/gender_dist.png)

    Gender Total
    * Female 112
    * Male 88

# Compare distributions

![Annual KDE](charts/kde_annual.png)

![Spending KDE](charts/kde_spendingScore.png)

![Age KDE](charts/kde_age.png)

# Boxplot
![Age Group by Gender](charts/boxplot_age.png)

![Annual Income Grouped by Gender](charts/boxplot_annual_income.png)



# (3) Bi-variate Analysis
## (3.1) Scatter plot

![Scatter Plot](charts/scatter_plot.png)

## (3.2) Pair plot

![Pair Plot](charts/pairplot.png)

## (3.3) Group by
Utilize the group by function to calculate the average age, annual income, and spending score for each gender

             Age        Annual Income (k$)       Spending Score (1-100)
    Gender                                                   
    Female   38.10               59.25                   51.53
    Male     39.81               62.23                   48.51

![Group By](charts/groupby.png)
![Group By](charts/highest_Avg.png)
![Group By](charts/highest_annual.png)

## Heat Map

![Heat Map](charts/heatmap.png)


# 4.1) Pivot

                 Annual Income (k$)       CustomerID            Spending Score (1-100)      
    Gender             Female  Male     Female       Male                 Female  Male
    Age                                                                               
    66                   63.0  63.0      107.0      110.0                   50.0  48.0
    67                   47.0  45.0       63.0  65.666667                   52.0  38.0
    68                   53.5  63.0       79.5      109.0                   51.5  43.0
    69                No Data  44.0    No Data       58.0                No Data  46.0
    70                No Data  47.5    No Data       66.0                No Data  55.5


# 4.2) Where function
    mall_customers_df["age_group"] = np.where(mall_customers_df["Age"] < 30, "18-29",
                                          np.where(mall_customers_df["Age"] < 45, "30-40",
                                          np.where(mall_customers_df["Age"] < 55, "45-54", "55+")))


    grouped_df = mall_customers_df.groupby(["age_group", "Gender"])\
    .agg({"Annual Income (k$)": ["count", "mean"]}).round(2)


                                       Annual Income (k$) 

                                        count   mean

        age_group Gender   

        18-29     Female                 29     50.66
                  Male                   26     54.65
        
        30-44     Female                 46     66.17
                  Male                   30     75.90

        45-54     Female                 25     58.88
                  Male                   14     58.21

        55+       Female                 12     54.25
                  Male                   18     53.50


![Group Age](charts/plot_new_column.png)

# 4.3) Clustering 

![The Elbow Method](charts/elbow_method.png)
        

![Classification Data](charts/classification_Data.png)

