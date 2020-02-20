import json

data_file = "/home/iki/sashi//robotkoop/CLEVR_v1.0/questions/CLEVR_train_questions.json"

with open(data_file) as json_file:
    data = json.load(json_file)