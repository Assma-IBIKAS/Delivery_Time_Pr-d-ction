import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR



def load_data(data):
    return pd.read_csv(data)


data = load_data("dataset-68eb8f248e099277381881.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()

#remplacer les champs catégorielles manquantes avec les valeurs les plus fréquentes

imputer = SimpleImputer(strategy='most_frequent')
data[["Weather","Traffic_Level","Time_of_Day"]] = imputer.fit_transform(data[["Weather","Traffic_Level","Time_of_Day"]])

#remplacer les champs numériques manquantes avec la moyenne 

data['Courier_Experience_yrs'].fillna(data['Courier_Experience_yrs'].mean(), inplace=True)
data.to_csv('my_data.csv', index=False)

def count_plot(data, column, hue=None):
    plt.figure(figsize=(8,4))
    sns.countplot(x=column, data=data, hue=hue)
    plt.title(f"Count Plot de {column}" + (f" par {hue}" if hue else ""))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

columns_to_plot = ["Weather","Traffic_Level","Time_of_Day","Vehicle_Type"]

for col in columns_to_plot:
    count_plot(data,col,hue=None)


columns = data[['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs','Delivery_Time_min']]

def correlation(columns):
    #columns_to_encoder = data.select_dtypes(include=['int64','float'])
    corr = columns.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
    plt.show()

correlation(columns)

def plot_boxplots(data, target_col='Delivery_Time_min'):
    # Sélection des colonnes catégorielles
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Boucle sur les colonnes catégorielles
    for col in categorical_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(x=col, y=target_col, data=data)
        plt.title(f'Boxplot de {target_col} selon {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_boxplots(data, target_col='Delivery_Time_min')


def target_distribution(data, target_col='Delivery_Time_min'):
    plt.figure(figsize=(8,5))
    sns.histplot(data[target_col], bins=30, kde=True)
    plt.title(f'Distribution de {target_col}')
    plt.xlabel(f'{target_col}')
    plt.ylabel('Nombre')
    plt.tight_layout()
    plt.show()


target_distribution(data, target_col='Delivery_Time_min')

def prepare_data(data):
    # Séparer X et y
    X = data.drop(columns=['Order_ID', 'Delivery_Time_min'])
    y = data['Delivery_Time_min']

    # Colonnes catégorielles et numériques
    cat_cols = ['Weather', 'Traffic_Level']
    num_cols = ['Preparation_Time_min', 'Distance_km']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, cat_cols, num_cols

prepare_data(data)

# Préparer les données
X_train, X_test, y_train, y_test, X_cat, X_num = prepare_data(data)


def train_model_with_grid(X_train, y_train, X_num, X_cat, model_type='rf'):
    # Prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), X_cat)
        ]
    )

    # Choix du modèle et des paramètres
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'feature_selection__k': [5, 8, 10],
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'svr':
        model = SVR()
        param_grid = {
            'feature_selection__k': [2, 3, 4],
            'regressor__C': [0.1, 1, 10],
            'regressor__epsilon': [0.01, 0.1, 0.5],
            'regressor__kernel': ['linear', 'rbf']
        }
    else:
        raise ValueError("model_type doit être 'rf' ou 'svr'")

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regressor', model)
    ])

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Entraînement
    grid_search.fit(X_train, y_train)

    return grid_search


# RandomForest
grid_rf = train_model_with_grid(X_train, y_train, X_num, X_cat, model_type='rf')

# SVR
grid_svr = train_model_with_grid(X_train, y_train, X_num, X_cat, model_type='svr')


# 1. Afficher les meilleurs paramètres
print("Meilleurs paramètres RandomForest :")
print(grid_rf.best_params_)

print("\nMeilleurs paramètres SVR :")
print(grid_svr.best_params_)

def evaluate_model(grid, X_test, y_test):
    y_pred = grid.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE  : {mae:.2f}")
    print(f"R2   : {r2:.2f}")
    
    return y_pred

print("Performance RandomForest :")
y_pred_rf = evaluate_model(grid_rf, X_test, y_test)

print("\nPerformance SVR :")
y_pred_svr = evaluate_model(grid_svr, X_test, y_test)
