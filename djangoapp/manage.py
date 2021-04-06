#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import time
import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template, url_for , request
import pickle
import joblib
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/rankhome')
def rank_home():
    return render_template('rank.html')

@app.route('/rankresult', methods=['POST', 'GET'])
def rank_flight():
    # get the parameters
    origin_airport = str(request.form['origin_airport'])
    dest_airport = str(request.form['dest_airport'])
    series = recommend(origin_airport,dest_airport)
    df = pd.DataFrame(columns=['Airlines', 'Score'])
    i=0
    for items in series.iteritems(): 
        df.loc[i] = ([items[0]]+[items[1]])
        i=i+1
    table_content = df.to_html(classes=["table-light", "table-striped","table-hover"])
    context = {'table_content': table_content}
    return render_template('results.html',origin=(origin_airport),destination=(dest_airport),table_content=(table_content))

@app.route('/percentagehome')
def percentage_home():
    return render_template('percentagehome.html')

@app.route('/percentagedelay', methods=['POST', 'GET'])
def precentage_delay():
    origin_airport = str(request.form['origin_airport'])
    dest_airport = str(request.form['dest_airport'])
    airlines_name=str(request.form['airlines_name'])
    chart_content=percentageofdelay(origin_airport,dest_airport,airlines_name)
    img_path="static/djangoapp/"+origin_airport+"_"+dest_airport+'_'+airlines_name+".jpg"
    return render_template('percentage.html',origin=(origin_airport),destination=(dest_airport),airlinesname=(airlines_name),img_path=(img_path))

@app.route('/dornothome')
def dornot_home():
    return render_template('delayornothome.html')

@app.route('/dornotresult', methods=['POST', 'GET'])
def dornot_result():
    origin_airport = request.form['origin_airport']
    dest_airport = request.form['dest_airport']
    airlines_name=request.form['airlines_name']
    distance=request.form['distance']
    departure_delay=request.form['departure_delay']
    scheduled_time=request.form['scheduled_time']
    airtime=request.form['airtime']
    taxi_out=request.form['taxi_out']
    val=dornot_result(airlines_name,origin_airport,dest_airport,distance,departure_delay,scheduled_time,airtime,taxi_out)
    if(val==[0]):
        valans=''
    else:
        valans='1'
    # valans=str(val)
    return render_template('delayornotresult.html',answer=(valans))


def recommend(src,dest):
    flightsinfo = pd.read_csv("C:\\python programs\\django_sample\\djangoapp\\data\\flights.csv",nrows=200000)
    airport = pd.read_csv('C:\python programs\django_sample\djangoapp\data\\airports.csv')
    airlines = pd.read_csv('C:\python programs\django_sample\djangoapp\data\\airlines.csv')
    le=LabelEncoder()
    flights=flightsinfo
    Flights1 = flightsinfo
    Flights1=flightsinfo.drop(['YEAR','MONTH','DAY','DAY_OF_WEEK','TAIL_NUMBER','DEPARTURE_TIME','WHEELS_OFF','WHEELS_ON','SCHEDULED_ARRIVAL','ARRIVAL_TIME','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'], axis = 1)
    Flights1['Is_Delayed'] = np.where(Flights1['ARRIVAL_DELAY']<=0, 0,1)
    Flights2=Flights1
    Flights1=Flights1.loc[(Flights1['ORIGIN_AIRPORT'] == src)  | (Flights1['DESTINATION_AIRPORT'] == dest)]
    Flights2=Flights2.loc[(Flights2['ORIGIN_AIRPORT'] == src)  & (Flights2['DESTINATION_AIRPORT'] == dest)]
    airlines_dict = dict(zip(airlines['IATA_CODE'],airlines['AIRLINE']))
    airport_dict = dict(zip(airport['IATA_CODE'],airport['AIRPORT']))
    Flights1 = Flights1.dropna(subset = ['TAXI_IN','ARRIVAL_DELAY'])
    X= Flights1.drop(['ELAPSED_TIME','DIVERTED','SCHEDULED_DEPARTURE','CANCELLED','FLIGHT_NUMBER','Is_Delayed','TAXI_IN'], axis = 1)
    Flights1['DESC_AIRLINE'] = flightsinfo['AIRLINE'].apply(lambda x: airlines_dict[x])
    Flights2['DESC_AIRLINE'] = flightsinfo['AIRLINE'].apply(lambda x: airlines_dict[x])
    Flights1['Is_Delayed'] = np.where(Flights1['ARRIVAL_DELAY']<=0, 0,1)
    Flights2 = Flights2.dropna(subset = ['TAXI_IN','ARRIVAL_DELAY'])
    X['AIRLINE']= le.fit_transform(X['AIRLINE'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    # print(mapping['AS'])
    X['ORIGIN_AIRPORT'] = le.fit_transform(X['ORIGIN_AIRPORT'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    srcno=mapping[src]
    X['DESTINATION_AIRPORT'] = le.fit_transform(X['DESTINATION_AIRPORT'])
    X = X.drop(['ARRIVAL_DELAY'],axis = 1)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    destno=mapping[dest]
    y = Flights1['Is_Delayed']
    X_test=X.loc[(X['ORIGIN_AIRPORT'] == srcno)  & (X['DESTINATION_AIRPORT'] == destno)]
    # return X_test
    xgb_from_joblib = joblib.load('C:\\python programs\\django_sample\\djangoapp\\xgbmodel.pkl')    
    y_pred=xgb_from_joblib.predict(X_test)
    Flights2['delayed']=y_pred
    Flights2=Flights2.loc[(Flights2['delayed'] == 0)]
    rank_airlines = pd.DataFrame(Flights2.groupby('DESC_AIRLINE').count()['SCHEDULED_DEPARTURE'])
    rank_airlines['CANCELLED']=Flights2.groupby('DESC_AIRLINE').sum()['CANCELLED']
    rank_airlines['OPERATED']=rank_airlines['SCHEDULED_DEPARTURE']-rank_airlines['CANCELLED']
    rank_airlines['RATIO_OP_SCH']=rank_airlines['OPERATED']/rank_airlines['SCHEDULED_DEPARTURE']
    rank_airlines.drop(rank_airlines.columns[[0,1,2]],axis=1,inplace=True)
    Flights2['FLIGHT_SPEED'] = 60*Flights2['DISTANCE']/Flights2['AIR_TIME']
    rank_airlines['FLIGHT_SPEED'] = Flights2.groupby('DESC_AIRLINE')['FLIGHT_SPEED'].mean()
    Flights2.groupby('DESC_AIRLINE')[['ARRIVAL_DELAY','DEPARTURE_DELAY']].mean()
    rank_airlines['ARRIVAL_DELAY']= Flights2.groupby('DESC_AIRLINE')['ARRIVAL_DELAY'].mean()
    rank_airlines['ARRIVAL_DELAY']=rank_airlines['ARRIVAL_DELAY'].apply(lambda x:x/60) 
    rank_airlines['FLIGHTS_VOLUME'] = Flights2.groupby('DESC_AIRLINE')['FLIGHT_NUMBER'].count()
    total = rank_airlines['FLIGHTS_VOLUME'].sum()
    rank_airlines['FLIGHTS_VOLUME'] = rank_airlines['FLIGHTS_VOLUME'].apply(lambda x:(x/float(total))) 
    for i in rank_airlines.columns:
        a = rank_airlines.RATIO_OP_SCH*rank_airlines.FLIGHT_SPEED*rank_airlines.FLIGHTS_VOLUME
        b = rank_airlines.ARRIVAL_DELAY
        rank_airlines['SCORE'] = a/(1+b)
        rank_airlines.sort_values(['SCORE'],ascending=False,inplace=True)
    # for i in range(len(rank_airlines)):print(rank_airlines.index[i])
    return rank_airlines['SCORE']

def percentageofdelay(src,dest,airlinesname):
    flightsinfo = pd.read_csv("C:\\python programs\\django_sample\\djangoapp\\data\\flights.csv",nrows=200000)
    airport = pd.read_csv('C:\python programs\django_sample\djangoapp\data\\airports.csv')
    airlines = pd.read_csv('C:\python programs\django_sample\djangoapp\data\\airlines.csv')
    le=LabelEncoder()
    flights=flightsinfo
    Flights1 = flightsinfo
    Flights1=flightsinfo.drop(['YEAR','MONTH','DAY','DAY_OF_WEEK','TAIL_NUMBER','DEPARTURE_TIME','WHEELS_OFF','WHEELS_ON','SCHEDULED_ARRIVAL','ARRIVAL_TIME','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'], axis = 1)
    # Flights1['Is_Delayed'] = np.where(Flights1['ARRIVAL_DELAY']<=0, 0,1)
    Flights1=Flights1.loc[(Flights1['ORIGIN_AIRPORT'] == src)  & (Flights1['DESTINATION_AIRPORT'] == dest)  & (Flights1['AIRLINE'] == airlinesname)]
    Flights1['AIRLINE']= le.fit_transform(Flights1['AIRLINE'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    Flights1['ORIGIN_AIRPORT'] = le.fit_transform(Flights1['ORIGIN_AIRPORT'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    srcno=mapping[src]
    Flights1['DESTINATION_AIRPORT'] = le.fit_transform(Flights1['DESTINATION_AIRPORT'])
    Flights1 = Flights1.drop(['ARRIVAL_DELAY'],axis = 1)
    X_test=Flights1.drop(['ELAPSED_TIME','DIVERTED','SCHEDULED_DEPARTURE','CANCELLED','FLIGHT_NUMBER','TAXI_IN'], axis = 1)
    # return X_test
    xgb_from_joblib = joblib.load('C:\\python programs\\django_sample\\djangoapp\\xgbmodel.pkl')     
    y_pred=xgb_from_joblib.predict(X_test)
    Flights1['Is_Delayed']=y_pred
    totalcount=Flights1['Is_Delayed'].count()
    delayedcount=Flights1.loc[(Flights1['Is_Delayed'] == 1)]['Is_Delayed'].count()
    nodelayedcount=Flights1.loc[(Flights1['Is_Delayed'] == 0)]['Is_Delayed'].count()
    axis = plt.subplots(figsize=(10,14))
    Name = ['Delayed','Not Delayed']
    values = [(delayedcount/totalcount)*100,(nodelayedcount/totalcount)*100]
    plt.pie(values,labels=Name,autopct='%5.0f%%')
    filepath=src+'_'+dest+'_'+airlinesname
    plt.savefig('C:\python programs\django_sample\djangoapp\static\djangoapp\\'+filepath+'.jpg')
    return plt

def dornot_result(airlines_name,origin_airport,dest_airport,distance,departure_delay,scheduled_time,airtime,taxi_out):
    xgb_from_joblib = joblib.load('C:\\python programs\\django_sample\\djangoapp\\xgbmodel.pkl')    
    data = [[int(airlines_name),int(origin_airport),int(dest_airport),int(float(departure_delay)),int(float(taxi_out)),int(scheduled_time),int(float(airtime)),int(distance)]]
    # data=[[0,15,265,1448,-11.0,205,169.0,4.0,21.0]]
    df_final = pd.DataFrame(data,columns=['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DEPARTURE_DELAY','TAXI_OUT','SCHEDULED_TIME','AIR_TIME','DISTANCE'])
    y_final_val=xgb_from_joblib.predict(df_final)
    return y_final_val

if __name__ == '__main__':
	app.run(debug=False)

# def main():
#     """Run administrative tasks."""
#     os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoapp.settings')
#     try:
#         from django.core.management import execute_from_command_line
#     except ImportError as exc:
#         raise ImportError(
#             "Couldn't import Django. Are you sure it's installed and "
#             "available on your PYTHONPATH environment variable? Did you "
#             "forget to activate a virtual environment?"
#         ) from exc
#     execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
