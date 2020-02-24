import pip._vendor.requests


url_detect = 'https://bill.cognitiveservices.azure.com/face/v1.0/detect'

param_detect = {'returnFaceId': 'true', 
                'returnFaceLandmarks': 'false',
                'recognitionModel': 'recognition_02',
                'returnRecognitionModel': 'false',
                'detectionModel': 'detection_02'
                }