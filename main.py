import pysqlite3 as sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10


connection = None
DB_LOCATION = "/home/ivan/Personal/Repositorios/bet-scraper-DB/scrapping"

def get_sqlite_connection() -> sqlite3.Connection:
    """
    Singleton to return a SQLite connection or to declare it if there is none
    Returns:
        SQLite connection
    """
    global connection
    if connection is None:
        connection = sqlite3.connect(DB_LOCATION)
        connection.row_factory = sqlite3.Row
    return connection


def close_sqlite_connection():
    """
    Close SQLite connection if there is one open and initialize to None the connection variable.
    Returns:
        None
    """
    global connection
    if connection is not None:
        connection.close()
    connection = None

def select_from_sqlite(query_select, one=True):
    con = get_sqlite_connection()
    cur = con.cursor()
    cur.execute(query_select)
    if one:
        rows = cur.fetchone()
    else:
        rows = cur.fetchall()
    cur.close()
    return rows

#query_select = "SELECT Season, Jornada, Jornada_Date, Leagues,  Bote, cast(Recaudacion as float) as Recaudacion  FROM Quinielas_new WHERE Recaudacion IS NOT NULL"

#rows = select_from_sqlite(query_select, one=False)

query_select = "SELECT Season, Jornada, Jornada_Date, Leagues,  Bote, Recaudacion FROM Quinielas_new WHERE Recaudacion IS NOT NULL"
"""
rows = select_from_sqlite(query_select, one=False)
dict_rows = [dict(row) for row in rows]
for row in dict_rows:
    #print(type(row['Recaudacion']))
    #if len(split())
    try:
        row['Recaudacion'] = float(row['Recaudacion'])#.replace('.', ''))#.strip(",")[0])
    except Exception as ex:
        print(row['Recaudacion'])
        row['Recaudacion'] = int(row['Recaudacion'].replace(".",""))

df = pd.DataFrame(dict_rows, dtype=str)
"""
# Create your connection.
cnx = sqlite3.connect(DB_LOCATION)

df = pd.read_sql_query(query_select, cnx)

# df correlations for all columns
#print(df)

#Modify Jornada_Date with a substring
#df['Jornada_Date_minus'] = df['Jornada_Date'].str[:7]

df['Jornada_Date'] = pd.to_datetime(df['Jornada_Date'])
df['Week_Day'] = df['Jornada_Date'].dt.dayofweek
df['Month'] = df['Jornada_Date'].dt.month

print(df)

# print types of columns
print(df.dtypes)
# Remove decimal part of column Recaudacion using regex
#df['Recaudacion'] = df['Recaudacion'].str.replace(r'\,[0-9]*', '')
#df['Recaudacion'] = df['Recaudacion'].str.replace(".","")#.apply(lambda x: str(x).replace(',', '.'))#
print(df['Recaudacion'])

#Transform Recaudacion to float
df['Recaudacion'] = df['Recaudacion'].astype(float)
#plt.plot(df["Recaudacion"],label='Recaudacion Price history')

#Plot previous dataframe using Jornada_Date_minus as x axis and Recaudacion as y axis
#plt.plot(df["Jornada_Date_minus"],df["Recaudacion"],label='Recaudacion Price history')

#Get max and min of recaudacion
max_recaudacion = df['Recaudacion'].max()
min_recaudacion = df['Recaudacion'].min()
print(max_recaudacion)
print(min_recaudacion)
# select Season, Jornada, Jornada_Date from pandas DataFrame
# Filter by Season
#df_season = df[df['Season'] == '20/21']
# Transform NaN from Bote_Next_Jornada to 0
df['Bote'] = df['Bote'].fillna(0)
#final_df = df_season[["Jornada","Bote_Next_Jornada","Recaudacion","Week_Day","Month"]]
#final_df = df[["Bote","Recaudacion","Week_Day","Month"]]
final_df = df[["Bote","Recaudacion","Week_Day","Month"]]
print(final_df)
print(final_df.describe())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = final_df.drop(["Recaudacion"], axis=1)
Y = final_df["Recaudacion"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test)
print(y_train)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(train_scaled, y_train)
print(model.predict(train_scaled))
print(y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

mse = mean_squared_error(y_train, model.predict(train_scaled))
mae = mean_absolute_error(y_train, model.predict(train_scaled))

test_mse = mean_squared_error(y_test, model.predict(test_scaled))
predictions = model.predict(test_scaled)
test_mae = mean_absolute_error(y_test, model.predict(test_scaled))
print("mse = ",test_mse," & mae = ",test_mae," & rmse = ", sqrt(test_mse))

# Create pandas dataframe with columns y_test and predictions
df_predictions = pd.DataFrame({"y_test": y_test, "predictions": predictions})
print(df_predictions)
print(X_train)
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)
model.predict(train_scaled)