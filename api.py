import requests
import json
from secrets import api_key

def get_picture_face_id(picture_link):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/detect'

    param = {'returnFaceId': 'true', 
                    'returnFaceLandmarks': 'false',
                    'recognitionModel': 'recognition_02',
                    'returnRecognitionModel': 'false',
                    'detectionModel': 'detection_02'
                    }

    header = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key' : api_key}

    picture =  open(picture_link,'rb').read()

    request = requests.post(url, headers = header, params = param, data = picture)

    return request

def create_person_group(group_id):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id

    header = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key' : api_key}

    json_data = {
        'name' : group_id,
        'recognitionModel' : 'recognition_02'
    }

    json_data = json.dumps(json_data)

    request = requests.put(url, headers = header,  data = json_data)

    return request

def delete_person_group(group_id):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.delete(url, headers = header)

    return request

def train_person_group(group_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/train'

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.post(url, headers = header)

    return request

def get_group_training_status(group_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/training'

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.get(url, headers = header)

    return request

def list_groups():
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups'

    param = {
        'start': '', 
        'top': '1000',
        'returnRecognitionModel' : 'false'
    }

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.get(url, headers = header, params = param)

    return request

def create_person(group_id, person_name):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons'

    header = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key' : api_key}

    json_data = {
        'name' : person_name
    }

    json_data = json.dumps(json_data)

    request = requests.post(url, headers = header,  data = json_data)

    return request

def delete_person(group_id, person_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons/' + person_id

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.delete(url, headers = header)

    return request

def get_person(group_id, person_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons/' + person_id

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.get(url, headers = header)

    return request

def add_person_face(group_id, person_id, picture_link):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons/' + person_id + '/persistedFaces'

    header = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key' : api_key}

    param = {'detectionModel': 'detection_02'}

    picture =  open(picture_link,'rb').read()

    request = requests.post(url, headers = header, params = param,  data = picture)

    return request

def delete_person_face(group_id, person_id, face_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons/' + person_id + '/persistedFaces/' + face_id

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.delete(url, headers = header)

    return request

def list_persons(group_id):
    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/' + group_id + '/persons'

    param = {
        'start': '', 
        'top': '1000'
    }

    header = {'Ocp-Apim-Subscription-Key' : api_key}

    request = requests.get(url, headers = header, params = param)

    return request

def identify_person_from_model(group_id, face_id):
    url = "https://bill.cognitiveservices.azure.com/face/v1.0/identify"

    header = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key' : api_key}

    json_data = {
        'personGroupId' : group_id,
        'faceIds' : [
            face_id
        ],
        'maxNumOfCandidatesReturned' : 1,
        'confidenceThreshold' : 0.5
    }

    json_data = json.dumps(json_data)

    request = requests.post(url, headers = header,  data = json_data)

    return request

#print(delete_person_group('main_model').status_code)
#print(create_person_group('main_model').status_code)
#print(create_person('mygroupid', 'ref_0').text)
#print(delete_person('main_model', '5b703a0e-6a2b-454c-bf80-599182d504ab').status_code)
#print(add_person_face('mygroupid', '7150b650-40b0-4dc5-9c60-77ca4eb40e11', 'img/model/ref_0.jpg').text)
#print(list_groups().text)
#print(json.dumps(list_persons('main_model').json(), indent=4))
#print(get_group_training_status("main_model").text)
print(identify_person_from_model('main_model', "ff47061b-1ded-43fc-bede-0ced16b94952").text)
#print(json.dumps(list_persons('main_model').json(), indent=4))