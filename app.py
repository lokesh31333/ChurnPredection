import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
shap.initjs()
import time
from flask import Flask, request, jsonify, render_template, session
import base64
import io
import openai
import os
from flask_session import Session  # you might need to install this with pip

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
survmodel = pickle.load(open('survivemodel.pkl', 'rb'))
openai.api_key = "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"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    gender = 0
    if request.form["gender"] == 1:
        gender = 1
    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1
    Partner = 0
    if 'Partner' in request.form:
        Partner = 1
    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1
    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = int(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges*Tenure

    PhoneService = 0
    if 'PhoneService' in request.form:
        PhoneService = 1

    MultipleLines = 0
    if 'MultipleLines' in request.form and PhoneService == 1:
        MultipleLines = 1

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if request.form["InternetService"] == 0:
        InternetService_No = 1
    elif request.form["InternetService"] == 2:
        InternetService_Fiberoptic = 1

    OnlineSecurity = 0
    if 'OnlineSecurity' in request.form and InternetService_No == 0:
        OnlineSecurity = 1

    OnlineBackup = 0
    if 'OnlineBackup' in request.form and InternetService_No == 0:
        OnlineBackup = 1

    DeviceProtection = 0
    if 'DeviceProtection' in request.form and InternetService_No == 0:
        DeviceProtection = 1

    TechSupport = 0
    if 'TechSupport' in request.form and InternetService_No == 0:
        TechSupport = 1

    StreamingTV = 0
    if 'StreamingTV' in request.form and InternetService_No == 0:
        StreamingTV = 1

    StreamingMovies = 0
    if 'StreamingMovies' in request.form and InternetService_No == 0:
        StreamingMovies = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if request.form["Contract"] == 1:
        Contract_Oneyear = 1
    elif request.form["Contract"] == 2:
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if request.form["PaymentMethod"] == 1:
        PaymentMethod_CreditCard = 1
    elif request.form["PaymentMethod"] == 2:
        PaymentMethod_ElectronicCheck = 1
    elif request.form["PaymentMethod"] == 3:
        PaymentMethod_MailedCheck = 1

    features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
       InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
       PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
       'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)

    output = prediction[0,1]

    # Shap Values
    explainer = joblib.load(filename="explainer.bz2")
    shap_values = explainer.shap_values(np.array(final_features))
    shap_img = io.BytesIO()
    shap.force_plot(explainer.expected_value[1], shap_values[1], columns, matplotlib = True, show = False).savefig(shap_img, bbox_inches="tight", format = 'png')
    shap_img.seek(0)
    shap_url = base64.b64encode(shap_img.getvalue()).decode()

    # Hazard and Survival Analysis
    surv_feats = np.array([gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
       InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
       PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck])

    surv_feats = surv_feats.reshape(1, len(surv_feats))

    hazard_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_cumulative_hazard(surv_feats).plot(ax = ax, color = 'red')
    plt.axvline(x=Tenure, color = 'blue', linestyle='--')
    plt.legend(labels=['Hazard','Current Position'])
    ax.set_xlabel('Tenure', size = 10)
    ax.set_ylabel('Cumulative Hazard', size = 10)
    ax.set_title('Cumulative Hazard Over Time')
    plt.savefig(hazard_img, format = 'png')
    hazard_img.seek(0)
    hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

    surv_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_survival_function(surv_feats).plot(ax = ax, color = 'red')
    plt.axvline(x=Tenure, color = 'blue', linestyle='--')
    plt.legend(labels=['Survival Function','Current Position'])
    ax.set_xlabel('Tenure', size = 10)
    ax.set_ylabel('Survival Probability', size = 10)
    ax.set_title('Survival Probability Over Time')
    plt.savefig(surv_img, format = 'png')
    surv_img.seek(0)
    surv_url = base64.b64encode(surv_img.getvalue()).decode()

    life = survmodel.predict_survival_function(surv_feats).reset_index()
    life.columns = ['Tenure', 'Probability']
    max_life = life.Tenure[life.Probability > 0.1].max()

    CLTV = max_life * MonthlyCharges

    # Gauge plot
    def degree_range(n):
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
              colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):

        N = len(labels)
        colors = colors[::-1]


        """
        begins the plotting
        """

        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)

        labels = labels[::-1]

        """
        plots the sectors and the arcs
        """
        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]


        """
        set the labels (e.g. 'LOW','MEDIUM',...)
        """

        for mid, lab in zip(mid_points, labels):

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation = rot_text(mid))

        """
        set the bottom banner and the title
        """
        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability,2).astype(str), horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        """
        plots the arrow now
        """

        pos = (1-Probability)*180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                     width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        """
        removes frame and ticks, and makes axis equal and tight
        """

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format = 'png')
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability = output)

    t = time.time()
    
    prediction_text = 'Churn probability is {} Percent and Expected Life Time Value is ${}'.format(round(output, 2)*100, CLTV)
    print("Prediction Text:", prediction_text)

    session['prediction_text'] = prediction_text  # Store the prediction text in session

    return render_template('index.html', prediction_text=prediction_text, url_1=gauge_url, url_2=shap_url, url_3=hazard_url, url_4=surv_url)


@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    prediction_text = session.get('prediction_text', '')
    data = request.json
    user_input = data['input']
    chat_log = data.get('chat_log')
    print("chatbot resposne "+ prediction_text)
    print(chat_log)
    
    # Assuming you have the requests library installed
    import requests
    import json

    # Prepare the headers and data for the POST request
    headers = {
        'Authorization': f'Bearer {openai.api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for a Churn Churn retention team in Telecom Company."},
            {"role": "user", "content": prediction_text},
            {"role": "user", "content": user_input}
        ] + ([{"role": "assistant", "content": chat_log}] if chat_log else [])
    }

    # Make the POST request to the OpenAI API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response
        response_data = response.json()
        # Extract the message content from the response
        answer = response_data['choices'][0]['message']['content']
        return jsonify({'response': answer, 'chat_log': update_chat_log(chat_log, user_input, answer)})
    else:
        # Handle errors
        return jsonify({'error': 'Error in generating response from the model.'}), response.status_code

# ... rest of your code remains unchanged ...

def create_prompt(user_input, chat_log):
    if chat_log is None:
        return f"The following is a conversation with an AI assistant. The assistant is helpful in answering question regarding Telecom Customer Churn and Customer Retention Strategies.\n\nHuman: {user_input}\nAI:"
    else:
        return f"{chat_log}\nHuman: {user_input}\nAI:"

def update_chat_log(chat_log, user_input, answer):
    if chat_log is None:
        return f"Human: {user_input}\nAI: {answer}"
    else:
        return f"{chat_log}\nHuman: {user_input}\nAI: {answer}"


if __name__ == "__main__":
    app.run(debug=True)
