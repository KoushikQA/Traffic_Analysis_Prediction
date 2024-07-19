import importPackages

df = pd.read_csv('traffic.csv', parse_dates=True, index_col='DateTime')
print(df)

# Extract Year, Month, Day, Hour From Index

# extract year from date
df['Year'] = pd.Series(df.index).apply(lambda x: x.year).to_list()

# extract month from date
df['Month'] = pd.Series(df.index).apply(lambda x: x.month).to_list()

# extract day from date
df['Day'] = pd.Series(df.index).apply(lambda x: x.day).to_list()

# extract hour from date
df['Hour'] = pd.Series(df.index).apply(lambda x: x.hour).to_list()
print(df.head())

# Drop the ID Column

df.drop('ID', axis=1, inplace=True)
print(df)

# Data Exploration

# Plotting Histograms
# Plot the histogram for every junctions (Vehicles vs probability)
# Plot the time series at every junction (Date-time vs Vehicles)

def make_hist(junction=1):
    data = df[df['Junction'] == junction]
    f, ax = plt.subplots(figsize=(15, 5))
    ax = sns.histplot(data['Vehicles'], kde=True, stat='probability')
    ax.set_title(f'Plot show the distribution of data in junction {junction}')
    ax.grid(True, ls='-.', alpha=0.75)
    plt.show()

# Histogram for Junction 1
make_hist(1)

# Histogram for Junction 2
make_hist(2)

# Histogram for Junction 3
make_hist(3)

# Histogram for Junction 4
make_hist(4)




