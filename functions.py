import pandas as pd
def univariate(df:pd.DataFrame):
    """Input a dataframe 
    outputs a dataframe with all columns as rows
    and univariates as columns"""
    output_df = pd.DataFrame(columns=['Dtype','Numeric','Count','Missing','Unique','Mean','Min','25%ile','Median','75%ile','Max','Mode','Std','Skewness','Kurtosis'])
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype = df[col].dtype
            numeric = True
            count = df[col].count()
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            mean = df[col].mean()
            min = df[col].min()
            p25 = df[col].quantile(0.25)
            median = df[col].median()
            p75 = df[col].quantile(0.75)
            max = df[col].max()
            mode = df[col].mode().values[0]
            std = df[col].std()
            skewness = df[col].skew()
            kurtosis = df[col].kurt()
            output_df.loc[col] = [dtype,numeric,count,missing,unique,mean,
        min,p25,median,p75,max,mode,std,skewness,kurtosis]
        else:
            dtype = df[col].dtype
            numeric = False
            count = df[col].count()
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            mode = df[col].mode().values[0]
            output_df.loc[col] = [dtype,numeric,count,missing,unique,"-",
        "-","-","-","-","-",mode,"-","-","-"]
    return output_df.sort_values("Unique", ascending=True).sort_values(by=["Numeric", "Skewness"], ascending=False)
    
# To find pairwise bivariate statistics.
# label is the name of column to which to find the pairwise statistics of all columns
def bivarstats(df, label):

    from scipy import stats

    output_df = pd.DataFrame(columns=["Stat", "+/-", "Effect Size(Value)", "p-value"])

    for col in df:
        if not col == label:
            if df[col].isnull().sum() == 0: # Cannot run statistical functions where null values are present
                if pd.api.types.is_numeric_dtype(df[col]):
                    r, p = stats.pearsonr(df[label], df[col])
                    if r > 0:
                        output_df.loc[col] = ['r(Pearson)', 'Positive(+)', abs(round(r, 3)), p]
                    else:
                        output_df.loc[col] = ['r(Pearson)', 'Negative(-)', abs(round(r, 3)), p]
                    scatter(df[col],  df[label])
                else:
                    F, p = anova(df[[col, label]], col, label)
                    output_df.loc[col] = ["F", "", round(F, 3), p]
                    bar_chart(df, col, label)
            else:
                output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]
    return output_df.sort_values(["Stat", "Effect Size"], ascending=False)

def anova(df, feature, label):
    from scipy import stats

    groups = df[feature].unique()
    df_grouped = df.groupby(feature)
    group_labels = []
    for g in groups:
        g_list = df_grouped.get_group(g)
        group_labels.append(g_list[label])

    return stats.f_oneway(*group_labels)