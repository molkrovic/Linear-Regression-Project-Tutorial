import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
df = pd.read_csv(url)

#Codificar las variables categóricas utilizando valores numéricos
sex_dict = {'male':1, 'female':0}
df['sex'] = df['sex'].map(sex_dict)

smoker_dict = {'yes':1, 'no':0}
df['smoker'] = df['smoker'].map(smoker_dict)

region_dict = {'southeast':0, 'southwest':1, 'northwest':2, 'northeast':3}
df['region'] = df['region'].map(region_dict)

# Dividir el dataset en variables explicativas y target
X = df.drop(columns=['charges'])
y = df['charges']

# Dividir el dataset en train, val y test
X_train, X_test_aux, y_train, y_test_aux = train_test_split(X, y, test_size=0.3, random_state=19)
X_val, X_test, y_val, y_test = train_test_split(X_test_aux, y_test_aux, test_size=0.5, random_state=11)

# Construir el modelo
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Guardar el modelo
filename = 'modelo_regresion_lineal'
pickle.dump(lin_reg, open(filename, 'wb'))

# Evaluar el modelo
y_train_pred = lin_reg.predict(X_train)
y_val_pred = lin_reg.predict(X_val)
y_test_pred = lin_reg.predict(X_test)

RMSE_train = mean_squared_error(y_train, y_train_pred, squared=False)
RMSE_val = mean_squared_error(y_val, y_val_pred, squared=False)
RMSE_test = mean_squared_error(y_test, y_test_pred, squared=False)

print(F'RMSE train: {round(RMSE_train)}')
print(F'RMSE val: {round(RMSE_val)}')
print(F'RMSE test: {round(RMSE_test)}')
print()
print('Se guardó el modelo.')