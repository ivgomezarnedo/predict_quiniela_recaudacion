import pysqlite3 as sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
rcParams['figure.figsize']=20,10

#https://nbviewer.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Yacht%20Resistance%20with%20Linear%20Regression.ipynb


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



query_select = "SELECT Season, Jornada, Jornada_Date, Leagues,  Bote, Recaudacion FROM Quinielas_new WHERE Recaudacion IS NOT NULL"

# Create your connection.
cnx = sqlite3.connect(DB_LOCATION)

df = pd.read_sql_query(query_select, cnx)

df['Jornada_Date'] = pd.to_datetime(df['Jornada_Date'])
df['Week_Day'] = df['Jornada_Date'].dt.dayofweek
df['Month'] = df['Jornada_Date'].dt.month

# Split Season column by '-' and create a new column with the first element of the split
#df['Season_1'] = df['Season'].str.split('/').apply(lambda x: x[0])
#df['Season_2'] = df['Season'].str.split('/').apply(lambda x: x[1])

#Transform Recaudacion to float
df['Recaudacion'] = df['Recaudacion'].astype(float)


#Get max and min of recaudacion
max_recaudacion = df['Recaudacion'].max()
min_recaudacion = df['Recaudacion'].min()
print(max_recaudacion)
print(min_recaudacion)
# Filter by Season
#df_season = df[df['Season'] == '20/21']
# Transform NaN from Bote_Next_Jornada to 0
df['Bote'] = df['Bote'].fillna(0)

#final_df = df[["Jornada","Bote","Recaudacion","Week_Day","Month","Season_1","Season_2"]]
final_df = df[["Jornada","Bote","Recaudacion","Week_Day","Month"]]
print(final_df)
#print(final_df.describe())



X = final_df.drop(["Recaudacion"], axis=1)
Y = final_df["Recaudacion"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.03, random_state=123)

# Split Leagues column by ',' and create a new column with a numerical representation of each value
# https://stackoverflow.com/questions/34007308/linear-regression-analysis-with-string-categorical-features-variables


scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(train_scaled, y_train)

mse = mean_squared_error(y_train, model.predict(train_scaled))
mae = mean_absolute_error(y_train, model.predict(train_scaled))

test_mse = mean_squared_error(y_test, model.predict(test_scaled))
predictions = model.predict(test_scaled)
test_mae = mean_absolute_error(y_test, model.predict(test_scaled))
print("mse = ",test_mse," & mae = ",test_mae," & rmse = ", sqrt(test_mse))


# Create pandas dataframe with columns y_test and predictions
df_predictions = X_test
df_predictions['real'] = y_test
df_predictions['predictions'] = predictions

print(len(df_predictions))
#sort df_predictions by jornada
df_predictions = df_predictions.sort_values(by=['Jornada'])

plt.plot(df_predictions['Jornada'], df_predictions['real'], 'o', label='Recaudacion real')
plt.plot(df_predictions['Jornada'], df_predictions['predictions'], 'o', label='Prediccion')

# https://pythonguides.com/python-plot-multiple-lines/#:~:text=with%20a%20legend-,Python%20plot%20multiple%20lines%20from%20array,plot()%20function.
plt.legend()
plt.show()
"""
print(df_predictions)
print(X_train)
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)
model.predict(train_scaled)
"""