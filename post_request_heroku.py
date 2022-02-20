"""
Heroku Api test script
"""
import requests

data = {'age': 70,
        'workclass': 'Private',
        'fnlgt': 124191,
        'education': 'Masters',
        'marital_status': 'Never-married',
        'occupation': 'Exec-managerial',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'hoursPerWeek': 50,
        'nativeCountry': 'United-States'
        }

r = requests.post('https://udacity-heroku-deployment.herokuapp.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
