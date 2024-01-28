from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar el conjunto de datos
datos = pd.read_csv("./PredicEstres/Data/DATOS_ESTRES -Universidades.csv")
datos = datos.drop(["Unnamed: 0", "DEPARTAMENTO", "UNIVERSIDAD", "DNI"], axis=1)

# Convertir la variable categórica 'SEXO' a variables dummy
dummies_sex = pd.get_dummies(datos["SEXO"], drop_first=True)
datos = pd.concat([datos, dummies_sex], axis=1)
datos = datos.drop(["SEXO"], axis=1)
datos.rename(columns={'MASCULINO': 'SEXO'}, inplace=True)

# Definir las características (X) y la variable objetivo (y)
X = datos[['EDAD', 'SEXO']]
y = datos['NIVEL DE ESTRES']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_ent, y_ent)

@app.route('/')
def index():
    return render_template('index.html', resultado="")

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        edad = int(request.form['edad'])
        sexo = int(request.form['sexo'])

        nueva_persona = [[edad, sexo]]
        prediccion = int(modelo.predict(nueva_persona)[0])  # Convertir a entero

        # Devuelve el resultado como un número del 1 al 5
        resultado = {"Nivel_de_estres": prediccion}

        # Imprimir mensajes de depuración
        print("Solicitud recibida. Edad:", edad, "Sexo:", sexo, "Predicción:", prediccion)
        
        return jsonify(resultado)

    # Si la solicitud no es POST, podrías manejarlo de alguna manera o simplemente devolver un mensaje de error.
    return jsonify({"error": "Método no permitido"}), 405

if __name__ == '__main__':
    app.run(debug=True)