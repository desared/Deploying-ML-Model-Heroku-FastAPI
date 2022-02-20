"""
Test the API server.
"""


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome!"}


def test_post_high(client):
    request = client.post("/", json={'age': 70,
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
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}


def test_post_low(client):
    request = client.post("/", json={'age': 19,
                                     'workclass': 'Private',
                                     'fnlgt': 149184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}


def test_post_malformed(client):
    r = client.post("/", json={
        "age": 32,
        "workclass": "",
        "education": "Some-college",
        "maritalStatus": "",
        "occupation": "",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 422
