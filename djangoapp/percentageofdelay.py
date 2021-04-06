def percentageofdelay(src,dest):
  flightsinfo = pd.read_csv("drive/My Drive/flights.csv",nrows=200000)
  airport = pd.read_csv('drive/My Drive/airports.csv')
  airlines = pd.read_csv('drive/My Drive/airlines.csv')
  flights=flightsinfo
  Flights1 = flightsinfo
  Flights1=flightsinfo.drop(['YEAR','MONTH','DAY','DAY_OF_WEEK','TAIL_NUMBER','DEPARTURE_TIME','WHEELS_OFF','WHEELS_ON','SCHEDULED_ARRIVAL','ARRIVAL_TIME','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'], axis = 1)
  Flights1['Is_Delayed'] = np.where(Flights1['ARRIVAL_DELAY']<=0, 0,1)
  Flights1=Flights1.loc[(Flights1['ORIGIN_AIRPORT'] == src)  & (Flights1['DESTINATION_AIRPORT'] == dest)]
  totalcount=Flights1['Is_Delayed'].count()
  delayedcount=Flights1.loc[(Flights1['Is_Delayed'] == 1)]['Is_Delayed'].count()
  nodelayedcount=Flights1.loc[(Flights1['Is_Delayed'] == 0)]['Is_Delayed'].count()
  axis = plt.subplots(figsize=(10,14))
  Name = ['Delayed','Not Delayed']
  values = [(delayedcount/totalcount)*100,(nodelayedcount/totalcount)*100]

  plt.pie(values,labels=Name,autopct='%5.0f%%')
  plt.show()
  plt.savefig('drive/My Drive/percentageofdelayimage.png')
  return delayedcount
