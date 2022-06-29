import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


filename = 'quiniela_predict.sav'
loaded_model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

dict_predict = {"Jornada":66,"BOTE":0,"Week_Day":6,"Month":7} #Params lambda function
predict_df = pd.DataFrame(dict_predict, index=[0])
test_scaled = scaler.transform(predict_df)

print(loaded_model.predict(test_scaled))