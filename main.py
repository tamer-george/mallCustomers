import pandas as pd
import pyeda as eda
import numpy as np
import matplotlib.pyplot as plt
# import descriptive.pyeda as dp
from sklearn import preprocessing
from sklearn.cluster import KMeans


pd.set_option("display.width", 3000)
mall_customers_df = eda.import_dataset("Mall_Customers.csv")

# print(mall_customers_df.head())
# print(mall_customers_df.isnull().sum())
# print(mall_customers_df.shape)

"""
eda.display_column_types(mall_customers_df)
Numerical Columns: ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
Categorical Columns: ['Gender']
"""

mall_customers_df = mall_customers_df.drop(columns=["CustomerID"], axis=1)
# eda.display_describe_data(mall_customers_df)
# print(f"Number of Customers {200}".title())
# print(f"The average annual income per customer {60.56}".title())
# print(f"The median age of customers {36}".title() )
# print(f"The standard deviations for age {13.97} and the std for annual income {26.26}".title()

#               COLUMNS DISTRIBUTIONS

# eda.visualize_distribution_of_numeric_col(mall_customers_df, "Annual Income (k$)", 6,
#                                           "annual_dist.png")

# eda.visualize_distribution_of_numeric_col(mall_customers_df, "Age", 6, "age_dist.png")
# eda.visualize_distribution_of_numeric_col(mall_customers_df, 'Spending Score (1-100)', 6,
#                                           "spending_dist.png")
# eda.visualize_distribution_of_categorical_col(mall_customers_df, "Gender", "gender_dist.png")


#               COMPARE DISTRIBUTION

# title = "distribution of Annual income across gender"
# subtitle = " "
# eda.visualize_kde(mall_customers_df, 'Annual Income (k$)', "Gender","kde_annual.png",
#                   title, subtitle)

# title = "distribution of spending score based gender"
# subtitle = " "
# eda.visualize_kde(mall_customers_df, 'Spending Score (1-100)', "Gender",
#                   "kde_spendingScore.png", title, subtitle)

# title = "distribution of age across gender"
# subtitle = " "
# eda.visualize_kde(mall_customers_df, 'Age', "Gender", "kde_age.png", title, subtitle)

#               BOXPLOT

# title = "Distribution of gender vs age"
# subtitle = " "
# eda.visualize_boxplot(mall_customers_df, "Age", "Gender", "boxplot_age.png",
#                       title, subtitle)

# title = "Distribution of gender vs Annual Income"
# subtitle = " "
# eda.visualize_boxplot(mall_customers_df, 'Annual Income (k$)', "Gender",
#                       "boxplot_annual_income.png", title, subtitle)

#               BI-VARIATE ANALYSIS

# subtitle = "The data can be clustered into 5 groups."
# eda.visualize_basic_scatter_plot(mall_customers_df, 'Annual Income (k$)',
#                                  'Spending Score (1-100)', "scatter_plot.png", subtitle)

# eda.visualize_pair_plot(mall_customers_df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
#                         "Gender", "pairplot.png")

# df = mall_customers_df.groupby("Gender")[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
# print(df)


# title = "average age, annual income, and spending score for each gender"
# eda.visualize_multi_numeric_columns_avg(mall_customers_df,
#                                         ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
#                                         "Gender",
#                                         "groupby.png", title, subtitle="")

# title = "The highest spending score by gender"
# subtitle = "on average females are higher on their spending habits and behavior"
#
# eda.vis_top_highest_average(mall_customers_df, ["Gender"], "Spending Score (1-100)",
#                             [51.53], "highest_Avg.png", title, subtitle)

# title = "The highest average annual income by gender"
# subtitle = "working female earn less of what their male counterparts earn."
# eda.vis_top_highest_average(mall_customers_df,["Gender"], "Annual Income (k$)",
#                             [62.23], "highest_annual.png", title, subtitle)

# eda.vis_heatmap(mall_customers_df, "heatmap.png")

#               PIVOT TABLE & FEATURE ENGINEERING

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

# print(grouped_df)

grouped_df = mall_customers_df.groupby(["age_group", "Gender"])\
    .agg({"Annual Income (k$)": ["mean"]}).sort_values(("Annual Income (k$)", "mean"),
                                                       ascending=False)

eda.plot_grouped_by(grouped_df, "plot_new_column.png")
"""

#               CLUSTERING


clustering_df = mall_customers_df[["Annual Income (k$)", "Spending Score (1-100)"]]

#               STANDARDIZE THE VARIABLES

x_scaled = preprocessing.scale(clustering_df)
# print(x_scaled)


#               THE ELBOW METHOD

# wcss = []
# for i in range(1, 10):
#     kmeans = KMeans(i)
#     kmeans.fit(x_scaled)
#     wcss.append(kmeans.inertia_)
#
# plt.plot(range(1, 10), wcss)
# plt.suptitle("the elbow method".title(), fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
#              fontfamily="sans-serif")
# plt.grid(axis="y", alpha=0.2)
# plt.grid(axis="x", alpha=0)
# plt.tight_layout()
# plt.xlabel("Number of Clusters")
# plt.ylabel("WCSS")
# plt.savefig("elbow_method.png", dpi=300, bbox_inches="tight")
# plt.show()

#               EXPLORE CLUSTERING SOLUTIONS AND SELECT THE NUMBER OF CLUSTERS

kmeans = KMeans(5).fit(clustering_df)
clusters_new = clustering_df.copy()

clusters_new["cluster_pred"] = kmeans.fit_predict(clusters_new)
# print(clusters_new)

plt.scatter(clusters_new["Annual Income (k$)"], clusters_new["Spending Score (1-100)"],
            c=clusters_new["cluster_pred"], cmap="rainbow")
plt.suptitle("classification data".title(), fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
             fontfamily="sans-serif")
plt.grid(axis="y", alpha=0.2)
plt.grid(axis="x", alpha=0)
plt.tight_layout()
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
# plt.savefig("classification_Data.png", dpi=300, bbox_inches="tight")
plt.show()