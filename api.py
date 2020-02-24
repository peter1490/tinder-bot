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

def add_person_face(group_id, person_id, picture_link):

    url = 'https://bill.cognitiveservices.azure.com/face/v1.0/persongroups/'+ group_id + '/persons/' + person_id + '/persistedFaces'

    header = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key' : api_key}

    param = {'detectionModel': 'detection_02'}

    picture =  open(picture_link,'rb').read()

    request = requests.post(url, headers = header, params = param,  data = picture)

    return request

#print(delete_person_group('mygroupid').status_code)
#print(create_person_group('mygroupid').text)
#print(create_person('mygroupid', 'ref_0').text)
#print(delete_person('mygroupid', '7150b650-40b0-4dc5-9c60-77ca4eb40e11').text)
print(add_person_face('mygroupid', '7150b650-40b0-4dc5-9c60-77ca4eb40e11', 'img/model/ref_0.jpg').text)