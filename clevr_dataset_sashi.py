import os
import json
import pickle
import utils_sashi

from PIL import Image
from torch.utils.data import Dataset


class ClevrDataset(Dataset):

    def __init__(self, clevr_dir, train, dictionaries, transform=None):

        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(clevr_dir, "images", "train")
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(clevr_dir, "images", "val")

        cached_questions = quest_json_filename.replace('.json', '.pkl')

        if os.path.exists(cached_questions):
            print("===> using cached questions: {}".format(cached_questions))
            with open(cached_questions, 'rb') as f:
                self.questions = pickle.load(f)
        else:
            with open(quest_json_filename, 'r') as json_file:
                self.questions = json.load(json_file)['questions']
            with open(cached_questions, 'wb') as f:
                pickle.dump(self.questions, f)

        self.clevr_dir = clevr_dir
        self.transform = transform
        self.dictionaries = dictionaries

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')

        question = utils_sashi.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils_sashi.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])

        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample