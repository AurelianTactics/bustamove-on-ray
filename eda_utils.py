import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Convenience functions used in Exploratory Data Analysis. For the Contest, please see the jupyter notebook

#to do longer term :
#pairplot (sns)/scatter_matrix(pd) could be better
    # can add hues by target level
#describe_feature: separate one for test df or work into it with train df?
#scatter plot isn't doing what you want it to do.
    # want to be able to see target var dist for a single plot or vs. another variable (ie x=var,y=None/var2,hue=target_values
    #maybe add some feature transformations (log(x+1) to various scatter plots)
# check if dataset shuffled
# 	rolling mean by row index
# 	leave a stub to consider more things
# find duplicate columns
# springer notebook has remove duplicated features
# 	me: so drops duplicate, but does it keep at least 1 version of the non duplicate?
#Plot mean (or some other statistic) of features by a value in a graph
# df.mean().sort_values().plot(style=’.’)
# 	feature mean is y axis, feature value is the x axis
#3D scatter plots?


# convenience functions

# high level summary of the data frame
def describe_data_frame(df, na_value=None):
    column_length = len(df.columns)
    row_length = len(df)
    print("Number of rows: {}".format(row_length))
    print("Number of columns: {}".format(column_length))

    # checks for NaN
    if na_value != None:
        df = df.replace(na_value, np.NaN)
    numeric_count = 0
    na_count = 0
    na_index_list = []

    for i in range(len(df.columns)):
        if np.issubdtype(df.iloc[:, i].dtype, np.number):
            numeric_count += 1
        if df.iloc[:, i].isnull().values.any():
            na_count += 1
            na_index_list.append(i)
            zCount = len(df[df.iloc[:, i].isnull().values])
            zFloat = zCount / row_length
            print("NA found in {}: count {}, percentage {}".format(df.columns[i],zCount, str(np.round(zFloat, 2))))

    print("Number of numeric columns: {}".format(numeric_count))
    print("Number of columns with a NA value: {}".format(na_count))

    if na_count > 0:
        # print(df.columns[na_index_list])
        msno.matrix(df=df.iloc[:, na_index_list], figsize=(20, 14), color=(0.42, 0.1, 0.05))


# display correlation
def display_corr(df):
    df_temp = df.corr()
    colormap = plt.cm.inferno
    plt.figure(figsize=(16, 12))
    plt.title('Pearson correlation of continuous features', y=1.05, size=15)
    sns.heatmap(df_temp, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
    plt.show()


# pandas.plotting.scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False,
#            diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds)
# seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter',
#          diag_kind='hist', markers=None, size=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None)
def display_scatter(df,target_column=None):
    if target_column != None:
        print("add hues for target column to pair plot")

    if len(df.columns)<10:
        zlen = len(df)
        if zlen <= 300000:
            df_temp = df
        else:
            z_float = 300000./zlen
            df_temp = df.sample(frac=z_float,replace=False,random_state=13)
        sns.pairplot(df_temp, diag_kind='kde', markers='.')
    else:
        print("To do: add scatter plot for more variables")

    # if len(df.columns) <= 30:
    #     pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='kde')
#    else:
        # df too big, do it in sequence
        # for i in df_temp.columns:
        #     # df_train.reindex(df_train.id.abs().sort_index(inplace=False,ascending=False).index)['id'][0:10]
        #     print("\n " + str(i))
        #     df_temp2 = pd.DataFrame({i: df_temp[i], 'zz_abs': df_temp[i].abs()})
        #     print(df_temp2.sort_values(['zz_abs'], ascending=False)[i][1:11])


def describe_features(df, target_column, is_target_categorical, is_violin_plot=True):
    # gets baseline for categorical target
    if is_target_categorical:
        total_records = len(df)
        if target_column != '':
            target_counts = df[target_column].value_counts()
            print('Target Column Counts and Distribution: ')
            print(target_counts)
            target_percentage = np.round(100. * target_counts / total_records, 1)
            print(target_percentage)

    for i in df.columns:
        print('\n' + i + ' -- Feature ')
        details = ''

        # more NA checks elsewhere, this is a duplicate
        if df[i].isnull().values.any():
            zFloat = len(df[df[i].isnull().values]) / len(df[i])
            details += " | NaN " + str(np.round(zFloat, 2))

        unique_values = len(df[i].unique())

        is_numeric = np.issubdtype(df[i].dtype, np.number)
        if is_numeric:

            z1 = 0
            if np.mean(df[i]) < 100:
                z1 = 2
            temp_describe = df[i].describe()
            print(np.round(temp_describe, z1))
            # print(df[i].describe())
            details += " | is numeric"

            # plot histogram
            if unique_values > 50:
                range_min = temp_describe['25%'] - temp_describe['std'] * 3
                range_max = temp_describe['75%'] + temp_describe['std'] * 3
                if range_min < temp_describe['min']:
                    range_min = temp_describe['min']
                if range_max > temp_describe['max']:
                    range_max = temp_describe['max']
                # print(range_min, range_max)
            else:
                range_min = temp_describe['min']  # - temp_describe['std']*3
                range_max = temp_describe['max']  # + temp_describe['std']*3

            with np.errstate(invalid='ignore'):
                n, bins, patches = plt.hist(df[i], 25, range=[range_min, range_max])
                plt.show()


        else:
            # print(False)
            details += " | is not numeric"

        details += " | " + str(unique_values) + " unique values "
        if unique_values < 100 or unique_values < len(df[i]) / 20:
            if is_numeric:
                details += " | consider one-hot? (ordinal?) "
            else:
                details += " | consider one-hot?"

        print(details)

        if unique_values <= 20 and target_column != '':
            if is_target_categorical:
                # print( df.fillna('NaN').groupby([i,target_column]).agg({i:'count'}) )
                g0 = df.fillna('NaN').groupby([i, target_column]).agg({i: 'count'})
                # g1 = g0.groupby(level=0).apply( lambda x:100 * x / float(x.sum()) )
                g1 = g0.groupby(level=0).apply(lambda x: np.round(100 * x / float(x.sum()), 1))
                print('target percentages: ', target_percentage)
                print(g1)
                # g3 = g2.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
                # df_raw.groupby(['created_month', 'interest_level']).agg({'created_month': 'count'})

            else:
                print('Levels | Count | Mean | Median | std')
                g0 = df.fillna('NaN').groupby([i]).agg({target_column: 'count'})
                g1 = df.fillna('NaN').groupby([i]).agg({target_column: 'mean'})
                g2 = df.fillna('NaN').groupby([i]).agg({target_column: 'median'})
                g3 = df.fillna('NaN').groupby([i]).agg({target_column: 'std'})
                result = pd.concat([g0, g1, g2, g3], axis=1, join_axes=[g0.index])
                print(result)

            # do violin plot, target variable has to make sense
            if is_violin_plot:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x=i, y=target_column, data=df)
                plt.show()


def index_plots(df_train, df_test=None):
    print('Simple plt.plot, looking for horizontal lines for repeated values or vertical lines for non shuffled')
    for i in df_train.columns:
        if np.issubdtype(df_train[i].dtype, np.number):
            print('Train {}'.format(i))
            plt.plot(df_train[i], '.')
            plt.show()

            if df_test is not None and i in df_test.columns:
                print('Test {}'.format(i))
                plt.plot(df_train[i], '.')
                plt.show()


def scatter_plot(df_train, df_test=None, target_var_column=None, is_target_categorical=False, jitter=False):

    if target_var_column != None:
        print("Scatter plots vs. target variable. Consider setting jitter value")
        # No test set, go through train set features
        # if target variable is categorical color various values
        # if test set and categorical, add a new label for the test set to compare the distributions

        df_temp = df_train
        if df_test is not None and is_target_categorical:
            is_numeric = np.issubdtype(df_train[target_var_column].dtype, np.number)
            if is_numeric:
                df_test[target_var_column] = np.max(df_train[target_var_column]) + 1
            else:
                df_test[target_var_column] = "test"

            if len(df_test.columns) == len(df_train.columns):
                df_temp = pd.concat([df_train, df_test], axis=0)

        if np.issubdtype(df_train[target_var_column].dtype, np.number):
            hue_var = None
        else:
            hue_var = target_var_column

        for i in df_train.columns:
            print("{} vs. {} ".format(i, target_var_column))
            sns.stripplot(x=i, y=target_var_column, hue=hue_var, data=df_temp, jitter=jitter)
            plt.show()

#check columns without unique values, consider dropping
def check_column_counts(df):
    feat_counts = df.nunique(dropna=False)
    print("Columns with least unique values: ")
    print(feat_counts.sort_values()[:10])
    print("Histogram of unique values")
    plt.hist(feat_counts.astype(float) / df.shape[0], bins=100)
    plt.show()


#check for values that are only in train set/not in test set and vice versa
#only doing for categorical columns and numeric columns with 30 or fewer unique values
#for numeric with wide spread, see plots
def check_unique_train_test_set(df_train,df_test):
    for i in df_train.columns:
        if i in df_test.columns:
            #n_unique_train = df_train[i].nunique(drop_na=False)
            n_unique_train, n_unique_train_count = np.unique(df_train[i],return_counts=True)
            if len(n_unique_train) > 30 and np.issubdtype(df_train[i].dtype, np.number):
                continue
            n_unique_test, n_unique_test_count = np.unique(df_test[i],return_counts=True)
            if len(n_unique_train) != len(n_unique_test):
                print("{}: Feature has different train and test values -- {} vs. {} ".format(i,len(n_unique_train), len(n_unique_test)))
                print(pd.DataFrame({'train__features': n_unique_train, 'train_count': n_unique_train_count}))
                print(pd.DataFrame({'test__features': n_unique_test, 'test_count': n_unique_test_count}))


def check_unique_rows(df_train, target_col=None):
    print("Total Unique Values (Total rows {})".format(len(df_train)))
    print(df_train.nunique())
    if target_col != None:
        df_temp = df_train.drop([target_col], axis=1)
    else:
        df_temp = df_train

    df_temp = df_temp.groupby(df_temp.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    df_temp = df_temp.sort_values('count', ascending=False)
    df_temp = df_temp[df_temp['count'] > 1]
    print("Rows with duplicated values (no target col): {}".format(len(df_temp)))

    if target_col != None:
        df_temp2 = df_train.groupby(df_train.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
        df_temp2 = df_temp2.sort_values('count', ascending=False)
        df_temp2 = df_temp2[df_temp2['count'] > 1]
        print("Rows with duplicated values (including target col): {}".format(len(df_temp2)))
        print(df_train.iloc[df_temp.index].sort_values(target_col))
        return df_train.iloc[df_temp.index].sort_values(target_col)
    else:
        print(df_train.iloc[df_temp.index])
        return df_train.iloc[df_temp.index]
