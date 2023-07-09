import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import warnings


# --- Create List of Color Palettes ---
red_grad = ['#FF0000', '#BF0000', '#800000', '#400000', '#000000']
pink_grad = ['#8A0030', '#BA1141', '#FF5C8A', '#FF99B9', '#FFDEEB']
purple_grad = ['#4C0028', '#7F0043', '#8E004C', '#A80059', '#C10067']
color_mix = ['#F38BB2', '#FFB9CF', '#FFD7D7', '#F17881', '#E7525B', "#FFCDCD", "#FFBCBC", "#FF9090"]
black_grad = ['#100C07', 'dimgray', '#6D6A6A', '#9B9A9C', '#CAC9CD']
colors = ["lightslategrey", "crimson", "#06344d", "#b4dbe9", "royalblue"]
pastel = sns.color_palette("pastel")


#             ----------------  Libraries Settings  ---------------

pd.set_option("display.max_columns", None)
sns.set_style(style="whitegrid")
plt.rcParams["figure.dpi"] = 80
plt.rc("figure", autolayout=True, figsize=(10, 6))
warnings.filterwarnings("ignore")


#             ----------------  EDA Methods  ---------------


def show_values(axs, orient="v", space=.01):
    def _single(axes2):
        if orient == "v":
            for p in axes2.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                axes2.text(_x, _y, value, ha="center", color=colors[0])
        elif orient == "h":
            for p in axes2.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                axes2.text(_x, _y, value, ha="left", color=colors[0])

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def import_dataset(file_name: str):
    """    
    Read cvs data file.
    
    :param file_name: string contain the csv file name
    
    :return: pandas dataframe
    
    """
    try:
        df = pd.read_csv(file_name, parse_dates=True, header=0)
        return df
    except FileNotFoundError:
        print("No such File, upload your dataset CSV file to the project")


def display_column_types(data):
    """
    Separate numerical columns and categorical columns.
    
    :param data: variable
    
    :return: list of numerical and categorical feature names
    
    """
    numeric_variable_lst = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_variable_lst = data.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numerical Columns: {numeric_variable_lst}\nCategorical Columns: {categorical_variable_lst}")


def visualize_distribution_of_numeric_col(data_frame, column_name: str, bins: int) -> None:
    """
    Visualize the distribution of a numeric column.

    :param data_frame: variable
    
    :param column_name: str

    :param bins: int
    
    :return: Histogram, boxplot, q-q plot, skewness and kurtosis values
    
    """
    # Testing column Type
    if not data_frame[column_name].dtype in ["int64", "float64"]:
        raise TypeError(f"{column_name} Is not a Numeric Column. Only Numeric Column is Allowed")

    print(f"\033[1m" + f".: {column_name.title()} Skewness & Kurtosis :. " + "\033[0m")
    print("*" * 40)
    print("Skewness: "+"\033[1m{:.3f}".format(data_frame[column_name].skew(axis=0, skipna=True)))
    print("\033[0m"+"Kurtosis:" + "\033[1m{:.3f}".format(data_frame[column_name].kurt(axis=0, skipna=True)))
    print("\n")

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"{column_name.title()} Distribution", fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="14",
                 fontfamily="sans-serif", color=black_grad[0])

    # Histogram
    fig.add_subplot(2, 2, 2)

    ax_1 = sns.histplot(data=data_frame, x=column_name, kde=True, color=colors[1], bins=bins)
    plt.xlabel(column_name.title(), fontweight="regular", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.ylabel(None)
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)
    ax_1.set(frame_on=False)

    # Q-Q Plot
    ax_2 = fig.add_subplot(2, 2, 4)
    plt.title("Q-Q Plot", fontweight="bold", fontsize=14, fontfamily="sans-serif", color=black_grad[1])
    qqplot(data_frame[column_name], fit=True, line="45", ax=ax_2, markerfacecolor=colors[1], alpha=0.6)

    # Box Plot
    fig.add_subplot(1, 2, 1)
    ax_3 = sns.boxplot(data_frame, y=column_name, color=colors[1], boxprops=dict(alpha=0.8), linewidth=1.5)

    plt.ylabel(column_name, fontweight="regular", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)
    ax_3.set(frame_on=False)
    plt.show()


def visualize_distribution_of_categorical_col(data_frame, column_name: str):
    """
    Visualize the distribution of a categorical column.

    :param data_frame: variable

    :param column_name: str

    :return: histogram, and pie charts

    """

    # Testing column Type
    if not data_frame[column_name].dtype in ["object", "category"]:
        raise TypeError(f"{column_name} Is a Numeric Column. Only Categorical Column is Allowed")

    print("*" * 25)
    print(f"\033[1m" + f".: {column_name.title()} Total :. " + "\033[0m")
    print("*" * 25)
    print(data_frame[column_name].value_counts(dropna=False))

    order = data_frame[column_name].value_counts().index
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"{column_name.title()} Distribution", fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0])
    # Pie Chart
    plt.subplot(1, 2, 1)

    labels = data_frame[column_name].unique()
    plt.pie(data_frame[column_name].value_counts(), colors=pastel, pctdistance=0.7, autopct="%.2f%%", labels=labels,
            wedgeprops=dict(alpha=0.8, edgecolor=colors[0]), textprops={"fontsize": 12}, startangle=90)
    centre = plt.Circle((0, 0), 0.45, fc="white", edgecolor=colors[0])
    plt.gcf().gca().add_artist(centre)

    # Histogram
    plt.subplot(1, 2, 2)

    ax = sns.countplot(x=column_name, data=data_frame, palette=pastel, order=order, edgecolor=black_grad[2],
                       alpha=0.85)
    show_values(ax, "v")
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(rotation=45)
    ax.set_yticklabels([])
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.grid(axis="y", alpha=0.1)
    plt.grid(axis="x", alpha=0)
    ax.set(frame_on=False)
    plt.show()


def visualize_pair_plot(data, numerical_columns: list, by_categorical_col: str):
    """
    Visualize the relationship between multiple numeric columns by categorical column.

    :param data: variable

    :param numerical_columns: list

    :param by_categorical_col: str

    :return: pair plot
    """
    g = sns.pairplot(data, vars=numerical_columns, hue=by_categorical_col, palette="husl", diag_kind="kde",
                     kind="scatter",
                     grid_kws=dict(diag_sharey=True),
                     plot_kws=dict(s=20, alpha=0.7),
                     diag_kws=dict(fill=False, linewidth=3.0))

    g.legend.remove()

    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=labels, loc="upper center", ncol=4)
    g.fig.subplots_adjust(top=0.99, bottom=0.20, )
    plt.show()


def visualize_kde(data, first_numeric: str, categorical_col: str,
                  title: str = "Add Chart Title",
                  subtitle: str = "Explain ur data viz by subtitle"
                  ):
    """
    Visualize the kernel density estimate of a numeric column per categorical column.

    :param data: variable

    :param first_numeric: str

    :param categorical_col: str

    :param title: str

    :param subtitle: str

    :return: KDE Plot
    """
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title.title(), fontweight="heavy", x=0.069, y=0.98,
                 ha="left", fontsize="16", fontfamily="sans-serif", color=black_grad[0])

    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=colors[0], loc="left", pad=12)

    ax = sns.kdeplot(x=data[first_numeric], hue=data[categorical_col], fill=True, common_norm=False, palette=pastel,
                     alpha=.5, linewidth=0)

    plt.xlabel(first_numeric.title(), fontweight="regular", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.tight_layout()
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)
    ax.set(frame_on=False)

    plt.show()


def display_describe_data(data) -> None:
    """
    Calculate basic statistics for the whole data.

    :param data: variable

    :return: prints out describe()
    """
    print(data.describe(include="all").round(2))


def visualize_boxplot(data, numeric_column: str, categorical_column: str,
                      title: str = "Add Chart Title",
                      subtitle: str = "Explain ur data viz by subtitle"):
    """
    Visualize the distribution of a numeric column by the categories of a categorical column.

    Add new title and subtitle content as a string.

    :param data: variable

    :param numeric_column: str

    :param categorical_column: str

    :param title: str

    :param subtitle: str

    :return: Box Plot

    """
    if not data[numeric_column].dtype in ["int64", "float64"]:
        raise TypeError(f"{numeric_column} Is not a Numeric Column. Only Numeric Column is Allowed")

    fig = plt.figure(figsize=(10, 6))

    fig.suptitle(title.title(),
                 fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0])

    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=colors[0], loc="left", pad=12)

    medians = data.groupby([categorical_column])[numeric_column].median().round(2).sort_values()

    vertical_offset = data[numeric_column].median() * 0.05

    box_plot = sns.boxplot(data=data, x=categorical_column, y=numeric_column,
                           medianprops={"color": "coral"}, order=medians.index, palette=pastel)

    for xtick in box_plot.get_xticks():
        box_plot.text(xtick, medians[xtick] + vertical_offset, medians[xtick], horizontalalignment="center",
                      size=12, color=colors[2], weight="bold")

    plt.tight_layout()
    plt.xlabel(None)
    plt.ylabel(numeric_column.title(), fontweight="bold", fontsize=10, fontfamily="sans-serif", color=colors[0])
    plt.grid(axis="x", alpha=0)
    plt.grid(axis="y", alpha=0.1)
    box_plot.set(frame_on=False)
    plt.show()


def visualize_basic_scatter_plot(data, first_column: str, second_column: str,
                                 subtitle: str = "Explain ur data viz by subtitle"):
    """
    Visualize the relationship between two numeric columns.
    
    :param data: variable
    
    :param first_column: str
    
    :param second_column: str

    :param subtitle: str
    
    :return: Basic scatter plot
    
    """

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Relationship Between {second_column.title()} and {first_column.title()}",
                 fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle,
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=colors[0], loc="left", pad=8)
    ax = sns.scatterplot(data=data, x=first_column, y=second_column, legend="full")

    plt.xlabel(first_column, fontweight="bold", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.ylabel(second_column, fontweight="bold", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.ticklabel_format(style="plain", axis="both")
    plt.grid(axis="both", alpha=0.4, lw=0.2)
    ax.set(frame_on=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.show()


def visualize_advanced_scatter_plot(data, first_numeric: str, second_numeric: str, categorical_col: str,
                                    title: str = "Add Chart Title",
                                    subtitle: str = "Explain ur data viz by subtitle"):
    """
    Vis the relationship between two numeric variable's by a third categorical variable,
    to dictate the color of data point's.

    Add new title and subtitle content as a string.


    :param data: Data frame variable

    :param first_numeric: str

    :param second_numeric: str

    :param categorical_col: str

    :param title: str

    :param subtitle: str

    :return: Advanced scatter plot
    """

    plt.figure(figsize=(10, 6))
    plt.suptitle(title.title(),
                 fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize=12, fontfamily="sans-serif", color=colors[0], loc="left", pad=10)

    ax = sns.scatterplot(data=data, x=first_numeric, y=second_numeric, hue=categorical_col, palette="tab10",
                         legend="full")

    plt.xlabel(first_numeric, fontweight="bold", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.ylabel(second_numeric, fontweight="bold", fontsize=11, fontfamily="sans-serif", color=black_grad[1])
    plt.ticklabel_format(style="plain", axis="both")
    plt.grid(axis="both", alpha=0.4, lw=0.5)
    ax.legend(title=categorical_col.title(), bbox_to_anchor=(0.98, 0.90), loc="upper left", frameon=True)
    ax.set(frame_on=False)
    plt.grid(axis="x", alpha=0)
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.show()


def visualize_multi_numeric_columns_avg(data, numerical_col: list, categorical_col: str,
                                        title: str = "Add chart title ",
                                        subtitle: str = "Explain ur data viz by subtitle "):
    """
    Visualize the mean of a multiple numeric columns by the categories of a categorical column.
    Add new Title and subtitle content to describe your chart.

    :param data: variable

    :param numerical_col: list

    :param categorical_col: str

    :param title: str

    :param subtitle: str

    :return: point plot chart
    """

    axs = data.groupby([categorical_col])[numerical_col].mean().round(2).plot(kind="bar", color=pastel)
    plt.suptitle(title.title(),
                 fontweight="heavy", x=0.039, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize=12, fontfamily="sans-serif", color=colors[0], loc="left", pad=10)
    axs.grid(False)
    plt.tight_layout()
    axs.set(frame_on=False)
    show_values(axs, "v")
    plt.xlabel(None)
    plt.xticks(rotation=45)
    axs.legend(title=f"{categorical_col.title()}",
               bbox_to_anchor=(1.2, 1), shadow=True, title_fontsize="x-large")
    plt.show()


def vis_heatmap(data):
    """
    Visualize the correlation between multiple numeric columns.
    
    :param data: variable
    
    :return: heatmap
    
    """
    numeric_features = data.select_dtypes(exclude=["object", "category"])
    plt.figure(figsize=(10, 6))
    plt.suptitle("Correlation Map",
                 fontweight="heavy", x=0.15, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    sns.heatmap(numeric_features.corr(), cmap="Reds", annot=True, linewidths=0.1)

    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.show()


def vis_highest_percentage_datapoints(data, first_categorical_col: str, second_categorical_col: str,
                                      subtitle: str = "explain ur data viz by subtitle "):
    """
    Visualize the highest percentage of datapoint values,

    for categorical variable grouped by second categorical variable.

    Add new subtitle content to the third parameter.

    :param data: variable

    :param first_categorical_col: str

    :param second_categorical_col: str

    :param subtitle: str

    :return: count plot
    """
    _, ax = plt.subplots(figsize=(10, 6))
    plt.suptitle(f"{first_categorical_col.title()} Highest Percentage per {second_categorical_col.title()}",
                 fontweight="heavy", x=0.044, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=black_grad[1], loc="left", pad=40)

    ax = sns.countplot(x=first_categorical_col, hue=second_categorical_col, data=data,
                       color="aquamarine", saturation=0.5)

    ax.set(xlabel=first_categorical_col, frame_on=False, ylabel=None)

    for c in ax.containers:
        # custom label calculates percent and add an empty string so 0 value bars don't have a number
        labels = [f'{h / data[second_categorical_col].count() * 100:0.1f}%' if (h := v.get_height()) > 0 else '' for v
                  in c]

        ax.bar_label(c, labels=labels, label_type='edge', fontsize=12, color=black_grad[1])

    # highest value
    patch_h = [patch.get_height() for patch in ax.patches]
    idx_tallest = np.argmax(patch_h)
    ax.patches[idx_tallest].set_facecolor("slateblue")

    ax.legend(title=second_categorical_col.title(), bbox_to_anchor=(1, 1.02), loc="upper left")
    plt.tight_layout()
    plt.xlabel(f"{first_categorical_col}", labelpad=10, fontweight="bold", fontsize=16,
               fontfamily="sans-serif",
               color=black_grad[1])
    plt.grid(axis="x", alpha=0)
    plt.grid(axis="y", alpha=0)
    plt.show()


def vis_top_highest_average(data_frame, categorical_column: list, numerical_column: str, avg_numbers: list,
                            title: str = "Explain ur data viz by subtitle ",
                            subtitle: str = "Explain ur data viz by subtitle "):
    """
    Visualize the highest average values.

    add avg_numbers to determine which bars to be colored.

    Add new title and subtitle content as string.

    :param data_frame: variable

    :param categorical_column: list

    :param numerical_column: str

    :param avg_numbers: list

    :param title: str

    :param subtitle: str

    :return: bar chart
    """

    df = data_frame.groupby(categorical_column)[numerical_column].mean().round(2).sort_values(ascending=False)\
        .nlargest(10)
    _, ax = plt.subplots(figsize=(10, 6))
    plt.suptitle(title.title(),
                 fontweight="heavy", x=0.069, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=black_grad[1], loc="left", pad=10)
    set_colors = [colors[1] if x in avg_numbers else colors[0] for x in df.values]
    fig = sns.barplot(data=pd.DataFrame(df).transpose(), orient="h", palette=set_colors)
    fig.set_ylabel(None)
    fig.set_xlabel(None)
    fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
    fig.set_xticklabels([])

    show_values(fig, "h")
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)
    ax.set(frame_on=False)
    plt.tight_layout()
    plt.show()


def plot_grouped_by(data):
    df = data.plot(kind="bar", color=pastel, width=0.8)
    plt.suptitle("the average annual income based on age group and gender ".title(),
                 fontweight="heavy", x=0.035, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )

    show_values(df, "v")

    # highest value
    patch_h = [patch.get_height() for patch in df.patches]
    idx_tallest = np.argmax(patch_h)
    df.patches[idx_tallest].set_facecolor("slateblue")

    plt.grid(axis="y", alpha=0.1)
    plt.grid(axis="x", alpha=0)
    df.set(frame_on=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1.02), loc="upper left")
    plt.ylabel(None)
    plt.xlabel(None)
    df.set_yticklabels([])
    plt.show()


def vis_top_ten_values(data, first_column: str, by_second_column: str, color_bar: list,
                       title: str = "explain ur data viz by subtitle",
                       subtitle: str = "explain ur data viz by subtitle"):
    """
    Visualize top ten values, add color_bar as integer list to determine which bars to be colored.

    Add new title and subtitle content as a string.

    :param data: Variable

    :param first_column: str

    :param by_second_column: str

    :param color_bar: list

    :param title: str

    :param subtitle: str

    :return: Bar plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.suptitle(title.title(),
                 fontweight="heavy", x=0.085, y=0.98, ha="left", fontsize="16",
                 fontfamily="sans-serif",
                 color=black_grad[0]
                 )
    plt.title(subtitle.capitalize(),
              fontweight="heavy", fontsize="12", fontfamily="sans-serif", color=black_grad[1], loc="left", pad=10)
    top_10 = data.groupby([first_column])[by_second_column].count().sort_values(ascending=False).iloc[:10]

    set_colors = [colors[1] if x in color_bar else colors[0] for x in top_10.values]
    fig = sns.barplot(data=pd.DataFrame(top_10).transpose(), orient="h", palette=set_colors)

    show_values(fig, "h")
    plt.xlabel(f"{first_column.title()}", labelpad=10, fontweight="bold", fontsize=16,
               fontfamily="sans-serif",
               color=black_grad[1])
    plt.xlabel(None)
    plt.xticks([])
    plt.grid(axis="y", alpha=0)
    plt.grid(axis="x", alpha=0)
    ax.set(frame_on=False)
    sns.despine(right=True, top=True, left=True)
    plt.tight_layout()
    plt.show()


def highlight_cells(value):
    if value == 4:
        color = "yellow"
    else:
        color = ""
    return "background-color: {}".format(color)