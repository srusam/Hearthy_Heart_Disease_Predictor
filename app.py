# Importing essential libraries
import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))
model2=load_model('ECG.h5')

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
	return render_template('index.html')
    
@app.route('/lipid_test')
def lipid_test():
    return render_template('lipid_test.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
@app.route('/g_map')
def g_map():
    return render_template('g_map.html')

@app.route('/upload')
def upload():
    return render_template("index6.html")

@app.route('/videoCall')
def videoCall():
     return render_template('vid.html')

@app.route('/lifestyleSurvey')
def lifestyleSurvey():
     return render_template('lifestyle.html')

@app.route('/lifestyleResult', methods=['GET', 'POST'])
def lifestyleResult():
     if request.method == 'POST':
        # Fetch all form data
        form_data = {}
        for field in request.form:
            form_data[field] = request.form[field]

        # Pass form data to the result.html template
        return render_template('lifestyle_result.html', form_data=form_data)
    
@app.route("/predictECG",methods=["GET","POST"])
def predictecg():
    if request.method=='POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        pred=model2.predict(x)#predicting classes
        y_pred = np.argmax(pred)
        print("prediction",y_pred)#printing the prediction
    
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[y_pred])

        return result#resturing the result
    return None

if __name__ == '__main__':
	app.run(debug=True)

