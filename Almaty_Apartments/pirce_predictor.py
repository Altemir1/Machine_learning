from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)
df = pd.read_csv('almaty_apartments_usable.csv')

X = df.drop('price', axis=1)
y = df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Gradient boosting regression
model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)

# Random forest regression
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

def perform_calculations(number_of_rooms, district, structure_type, year_of_construction, floor, area, quality):
    district_encoding = {"Бостандыкский р-н": 3, "Ауэзовский р-н": 2, "Алмалинский р-н": 1, "Медеуский р-н": 5, "Наурызбайский р-н": 6, "Алатауский р-н": 0, "Турксибский р-н": 7, "Жетысуский р-н": 4}
    structure_type_encoding = {"монолитный": 2, "панельный": 3, "кирпичный": 1, "иное": 0}
    quality_encoding = {"хорошее": 3, "среднее": 1, "черновая отделка": 4, "требует ремонта": 2, "свободная планировка": 0}


    property_dict = {
        'number_of_rooms': number_of_rooms,
        'district': district_encoding[district],
        'structure_type': structure_type_encoding[structure_type],
        'year_of_construction': year_of_construction,
        'floor': floor,
        'area': area,
        'quality': quality_encoding[quality]
    }

    df_row = pd.DataFrame(property_dict, index=[0])

    lr = model_lr.predict(df_row)
    rf = model_rf.predict(df_row)
    gb = model_gb.predict(df_row)

    result = f"Linear regression: {int(lr):.2f} | Random forest: {int(rf):.2f} | Gradient boosting: {int(gb):.2f}"

    return result

@app.route('/process_property_form', methods=['POST'])
def process_property_form():
    # Retrieve form data
    number_of_rooms = request.form.get('number_of_rooms')
    district = request.form.get('district')
    structure_type = request.form.get('structure_type')
    residential_complex = request.form.get('residential_complex')
    year_of_construction = request.form.get('year_of_construction')
    floor = request.form.get('floor')
    area = request.form.get('area')
    quality = request.form.get('quality')
    bathroom = request.form.get('bathroom')
    internet_type = request.form.get('internet_type')

    print(number_of_rooms)
    result = perform_calculations(number_of_rooms, district, structure_type, year_of_construction, floor, area, quality)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

