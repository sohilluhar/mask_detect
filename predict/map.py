import requests

URL = "https://discover.search.hereapi.com/v1/discover"
latitude = 19.0940581
longitude = 72.8966304

api_key = "OzY0jbh2SnqI5LeBhsvblvlLpa4xKet3XPTEngeyVAs"
query = 'private hospital'
limit = 5

PARAMS = {
    'apikey': api_key,
    'q': query,
    'limit': limit,
    'at': '{},{}'.format(latitude, longitude)
}

# sending get request and saving the response as response object
r = requests.get(url=URL, params=PARAMS)
data = r.json()
print(data)
# print(data['items'][0]['categories'][0]['name'])
'''
hospitalOne = data['items'][0]['title']
hospitalOne_address =  data['items'][0]['address']['label']
hospitalOne_latitude = data['items'][0]['position']['lat']
hospitalOne_longitude = data['items'][0]['position']['lng']
hospitalOne_contect = data['items'][0]['contacts'][0]
hospitalOne_Docter_name = data['items'][0]['categories'][0]['name']

hospitalTwo = data['items'][1]['title']
hospitalTwo_address =  data['items'][1]['address']['label']
hospitalTwo_latitude = data['items'][1]['position']['lat']
hospitalTwo_longitude = data['items'][1]['position']['lng']
hospitalTwo_contect = data['items'][1]['contacts'][0]
print(hospitalOne,hospitalOne_address,hospitalOne_contect)
print(hospitalTwo,hospitalTwo_address,hospitalTwo_contect)
'''
