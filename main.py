import pandas as pd
import pyeda as eda
import numpy as np
import descriptive.pyeda as dp
from sklearn.cluster import KMeans


pd.set_option("display.width", 3000)
mall_customers_df = eda.import_dataset("Mall_Customers.csv")


# eda.display_column_types(mall_customers_df)
# Numerical Columns: ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
# Categorical Columns: ['Gender']

# eda.display_describe_data(mall_customers_df)
# print(f"Number of Customers {200}".title())
# print(f"The average annual income per customer {60.56}".title())
# print(f"The median age of customers {36}".title() )
# print(f"The standard deviations for age {13.97} and the std for annual income {26.26}".title()


# Numerical Columns: ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
# Categorical Columns: ['Gender']
# eda.visualize_distribution_of_numeric_col(mall_customers_df, "Annual Income (k$)", 6)
# eda.visualize_distribution_of_numeric_col(mall_customers_df, "Age", 6)
# eda.visualize_distribution_of_numeric_col(mall_customers_df, 'Spending Score (1-100)', 6)
# eda.visualize_distribution_of_categorical_col(mall_customers_df, "Gender")
# title="distribution of Annual income across gender"
# subtitle =" "
# eda.visualize_kde(mall_customers_df, 'Annual Income (k$)', "Gender", title, subtitle)
# eda.visualize_kde(mall_customers_df, 'Age', "Gender", title, subtitle)
# eda.visualize_kde(mall_customers_df, 'Spending Score (1-100)', "Gender", title, subtitle)
# subtitle = "The data can be clustered into 5 groups."
# eda.visualize_boxplot(mall_customers_df, "Age", "Gender", title, subtitle)
# eda.visualize_boxplot(mall_customers_df, 'Annual Income (k$)', "Gender", title,subtitle)
# eda.visualize_basic_scatter_plot(mall_customers_df, 'Annual Income (k$)', 'Spending Score (1-100)', subtitle)
# eda.visualize_pair_plot(mall_customers_df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], "Gender")

# df = mall_customers_df.groupby("Gender")[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
# print(df)


# title = "average age, annual income, and spending score for each gender"
# subtitle = "on average females are higher on their spending habits and behavior"

# eda.visualize_multi_numeric_columns_avg(mall_customers_df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
#                                         "Gender", title, subtitle="")
# title = "The highest average annual income by gender"
# subtitle = "working female earn less of what their male counterparts earn."

# eda.vis_top_highest_average(mall_customers_df, ["Gender"], "Spending Score (1-100)", [51.53], title, subtitle)
# eda.vis_top_highest_average(mall_customers_df,["Gender"], "Annual Income (k$)",[62.23], title, subtitle)

# eda.vis_heatmap(mall_customers_df)

"""
pvt_tbl = pd.pivot_table(mall_customers_df, index=["Age", "Gender"], aggfunc={"Annual Income (k$)": np.mean})
# print(pvt_tbl.head(10))

avg_income = mall_customers_df.groupby(["Age", "Gender"], as_index=False)["Annual Income (k$)"].mean().round(2)
# print(avg_income)


reshape_pvt = pd.pivot_table(mall_customers_df, index=["Age"], columns=["Gender"]).fillna("No Data")
reshape_pvt.tail().round(2)


mall_customers_df["age_group"] = np.where(mall_customers_df["Age"] < 30, "18-29",
                                          np.where(mall_customers_df["Age"] < 45, "30-44",
                                                   np.where(mall_customers_df["Age"] < 55, "45-54", "55+")))


grouped_df = mall_customers_df.groupby(["age_group", "Gender"])\
    .agg({"Annual Income (k$)": ["count", "mean"]}).round(2)

print(grouped_df)
grouped_df = mall_customers_df.groupby(["age_group", "Gender"])\
    .agg({"Annual Income (k$)": ["mean"]}).sort_values(("Annual Income (k$)", "mean"),
                                                       ascending=False)

eda.plot_grouped_by(grouped_df)
"""


X = mall_customers_df.iloc[:, [3, 4]].values
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
mall_customers_df["Spen_Class"] = y_kmeans
title = "Relation between spending score, annual income as spen class"
subtitle = "classification data"
# eda.visualize_advanced_scatter_plot(mall_customers_df, 'Annual Income (k$)', 'Spending Score (1-100)', "Spen_Class",
#                                     title, subtitle)

# 0 - Low Income 15 - 40  and Low spending score 1 - 40
# 1 - Low Income 15 - 40 and High spending score 60 -99
# 2 - Middle Income 40 - 70 and middle spending score 40 -60
# 3 - High Income 70 - 99 and Low spending score 1 -40
# 4 - High Income 70 -99 and High spending score 60 -99


annual_income = mall_customers_df["Annual Income (k$)"]
spending_score = mall_customers_df["Spending Score (1-100)"]
range1 = list(range(1, 40))
range2 = list(range(40, 71))
range3 = list(range(60, 100))

conditions = [
    (spending_score.isin(range1) & annual_income.isin(range1)),
    (spending_score.isin(range3) & annual_income.isin(range1)),
    (spending_score.isin(range2) & annual_income.isin(range2)),
    (spending_score.isin(range1) & annual_income.isin(range3)),
    (spending_score.isin(range3) & annual_income.isin(range3)),
            ]

choices = ["0", "1", "2", "3", "4"]

mall_customers_df["Spen_Class"] = np.select(conditions, choices, None)
clustering_df = mall_customers_df.dropna()
# print(clustering_df)
title = "Relation between spending score, annual income as spen class"
subtitle = "classification data"
# eda.visualize_advanced_scatter_plot(clustering_df,'Spending Score (1-100)','Annual Income (k$)',"Spen_Class",
#                                     title, subtitle)

campaign1 = mall_customers_df.loc[mall_customers_df["Spen_Class"] == "4"]
# print(campaign1.head())

title = "count of gender by spending score"
subtitle = "gender as High Income & High Spending Score"

eda.vis_top_ten_values(campaign1, "Gender", "Spending Score (1-100)", [17], title, subtitle)



def highlight_cells(value):
    if value == 4 and value == 2:
        color = "yellow"
    else:
        color = " "
    return "background-color: {}".format(color)





