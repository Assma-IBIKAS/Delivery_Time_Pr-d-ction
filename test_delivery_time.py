import pytest
import pandas as pd
from sklearn.metrics import mean_absolute_error
from Delivery_Time import prepare_data,train_model_with_grid

data = pd.read_csv("dataset-68eb8f248e099277381881.csv")

@pytest.fixture
def x_y():
    X = data.drop(columns=['Order_ID', 'Delivery_Time_min'])
    y = data['Delivery_Time_min']
    return X, y

# Test 1 : Vérification du format et des dimensions
def test_prepare_data_format(x_y):
    X, y = x_y
    X_train, X_test, y_train, y_test, X_cat, X_num = prepare_data(data)
    
    # Vérification des dimensions 
    assert X.shape[0] == y.shape[0], "X et y n'ont pas le même nombre de lignes" 
    assert X.shape[1] == 7 
    assert len(y.shape) == 1
    assert X_train.shape[0] == y_train.shape[0], "X_train et y_train incohérents"
    assert X_test.shape[0] == y_test.shape[0], "X_test et y_test incohérents"

    # Vérification du nombre de colonnes
    assert set(X_cat) == {'Weather', 'Traffic_Level'}, "Colonnes catégorielles incorrectes"
    assert set(X_num) == {'Preparation_Time_min', 'Distance_km'}, "Colonnes numériques incorrectes"

# Test 2 : Vérification que la MAE maximale ne dépasse pas un seuil
@pytest.mark.parametrize("model_type", ["rf", "svr"])
def test_mae(x_y, model_type):
    X, y = x_y
    X_train, X_test, y_train, y_test, X_cat, X_num = prepare_data(data)
    
    # Entraîner le modèle
    model = train_model_with_grid(X_train, y_train, X_num, X_cat, model_type)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Vérifie que la MAE est sous le seuil défini 
    mae = mean_absolute_error(y_test, y_pred)
    assert mae < 7