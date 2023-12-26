import joblib
from flask import Flask, request, render_template, redirect, url_for
from itertools import islice

app = Flask(__name__,template_folder='templates')

model = joblib.load('kneighbors.sav')

roles = ['Database Administrator', 'Hardware Engineer',
       'Application Support Engineer', 'Cyber Security Specialist',
       'Networking Engineer', 'Software Developer', 'API Specialist',
       'Project Manager', 'Information Security Specialist',
       'Technical Writer', 'AI ML Specialist', 'Software tester',
       'Business Analyst', 'Customer Service Executive', 'Data Scientist',
       'Helpdesk Engineer', 'Graphics Designer']
role_encodings = [ 7,  9,  2,  5, 12, 14,  1, 13, 11, 16,  0, 15,  3,  4,  6, 10,  8]
feature_labels = [4, 5, 1, 0, 3, 2, 6]
feature_keys = ['Not Interested', 'Poor', 'Beginner', 'Average', 'Intermediate','Excellent', 'Professional']
feature_inputs = dict(zip(feature_keys,feature_labels))
label_inputs = dict(zip(role_encodings,roles))
prediction = "Not Interested"
top_roles = []
confidence_scores = []

@app.route("/",methods= ["GET","POST"])
def Home():
    if request.method == 'POST':
        input = [feature_inputs[request.form[i]] for i in request.form]
        global prediction
        if  4 in set(input) and len(set(input)) == 1 :
            prediction = "Not Interested"
            return redirect(url_for('predict'))
        else:    
            prediction = model.predict([input])[0]
            prediction = label_inputs[prediction]
            predictions = model.predict_proba([input])
            predictions = dict(zip(['AI ML Specialist', 'API Specialist',
            'Application Support Engineer', 'Business Analyst',
            'Customer Service Executive', 'Cyber Security Specialist',
            'Data Scientist', 'Database Administrator', 'Graphics Designer',
            'Hardware Engineer', 'Helpdesk Engineer',
            'Information Security Specialist', 'Networking Engineer',
            'Project Manager', 'Software Developer', 'Software tester',
            'Technical Writer'],predictions[0]))
            print(predictions)
            predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse= True)}
            result = dict(islice(predictions.items(), 3))
            print(result)
            global top_roles
            top_roles = list(result.keys())
            global confidence_scores
            confidence_scores = list(result.values())
            return redirect(url_for('predict'))
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template('result.html', prediction =str(prediction), top_roles=top_roles, confidence_scores=confidence_scores)

if __name__ == '__main__':

    app.run(debug=True)

    