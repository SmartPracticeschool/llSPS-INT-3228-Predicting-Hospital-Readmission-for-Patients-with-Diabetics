
from flask import Flask
from flask import request,render_template
from keras.models import load_model
import tensorflow  as tf
global graph
graph = tf.get_default_graph()
import numpy as np
app = Flask(__name__)
@app.before_first_request
def load_model_to_app():
    app.predictor = load_model("PROJECT.h5")
@app.route('/')
def hello_world():
    return render_template("index.html",pred = 0)
@app.route('/predict', methods = ["POST"])
def predict():
    data=  [     request.form['encounter_id'],
                 request.form['patient_number'],
                 request.form['admission_type_id_Emergency'],
                 request.form['Discharged_to_home id'],
                 request.form['admission__source_id_Emergency'],
                request.form['medical_specialty'],
                 request.form['time_in_hospital'],
                 request.form['num_lab_procedures'],
                request.form['num_procedures'],
                 request.form['number_outpatient'],
             request.form['number_emergency'],
                 request.form['number_inpatient'],
                 request.form['number_diagnoses'],
                 request.form['ageGroup'],
                request.form['weight'],
                 request.form['race'],
                 request.form['gender'],
                 request.form['max_glu_serum'],
                 request.form['A1Cresult'],
                 request.form['metformin'],
                 request.form['repaglinide'],
                 request.form['nateglinide'],
                 request.form['chlorpropamide'],
                 request.form['glimepiride'],
                 request.form['acetohexamide'],
                 request.form['glipizide'],
                 request.form['glyburide'],
                request.form['tolbutamide'],
                 request.form['pioglitazone'],
                 request.form['rosiglitazone'],
                 request.form['acarbose'],
                 request.form['miglitol'],
                 request.form['troglitazone'],
                 request.form['tolazamide'],
                 request.form['examide'],
                 request.form['citoglipton'],
                request.form['insulin_no'],
                 request.form['gluburide-metformin'],
                 request.form['glipizide-metformin'],
                 request.form['glimepiride-pioglitazone'],
                 request.form['metformin-rosiglitazone'],
                 request.form['metformin-pioglitazone'],
                 request.form['change_med_no'],
                 request.form['insulin_Steady']]
    data = np.array([np.asarray(data, dtype=float)])
    predictions = app.predictor.predict(data)
    print('predicted class'.format(predictions))
    class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]
    index = ['not readmitted','readmitted greater than 30 days ',"readmittes less than 30 days"]
    return render_template('index.html',Submit = "will patients get readmitted = " + str(index[class_[0]]) )




if __name__ == '__main__':
    app.run(debug = True)

    
    
    
    
