import os
import pickle
import re
import torch

from tqdm import tqdm


classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }


def build_dictionaries(clevr_dir):

    def compute_class(answer):
        for name, values in classes.items():
            if answer in values:
                return name

            raise ValueError('Answer {} does not belong to a known class'.format(answer))

    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')

    if os.path.exists(cached_dictionaries):
        print('===> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)

    quest_to_ix = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')

    with open(json_train_filename, 'r') as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question = tokenize(q['questions'])
            answer = q['answer']

            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix) + 1

            a = answer.lower()
            if a not in answ_to_ix:
                ix = len(answ_to_ix) + 1
                answ_to_ix[a] = ix
                answ_ix_to_class[ix] = compute_class(a)

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def to_dictionary_indexes(dictionary, sentence):

    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])

    return idxs


def collate_samples_from_pixels(batch):
    return collate_samples(batch, False, False)


def collate_samples(batch, state_description, only_images):

    batch_size = len(batch)

    if only_images:
        images = batch
    else:
        images = [d['image'] for d in batch]
        answers = [d['answer'] for d in batch]
        questions = [d['question'] for d in batch]

        max_len = max(map(len, questions))

        padded_questions = torch.LongTensor(batch_size, max_len).zero_()
        for i, q in enumerate(questions):
            padded_questions[i, :len(q)] = q

    if only_images:
        collated_batch = torch.stack(images)
    else:
        collated_batch = dict(image=torch.stack(images),
                              answer=torch.stack(answers),
                              question=padded_questions)

    return collated_batch



def tokenize(sentence):

    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    split = s.split()

    lower = [w.lower() for w in split]
    return lower


def load_tensor_data(data_batch, cuda, invert_questions, volatile=False):

    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    qst = data_batch['question']

    if invert_questions:
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len, -1, -1, -1).long())

    import ipdb; ipdb.set_trace()

    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    qst = torch.autograd.Variable(qst, **var_kwargs)
    label = torch.autograd.Variable(data_batch['answer'], **var_kwargs)

    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)

    return img, qst, label


def load_tensor_data_sashi(data_batch, cuda, invert_questions, requires_grad=True):

    qst = data_batch['question']

    if invert_questions:
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len, -1, -1, -1).long())

    import ipdb; ipdb.set_trace()

    img = torch.Tensor(data_batch['image'], requires_grad=requires_grad)
    qst = torch.Tensor(qst, requires_grad=requires_grad)
    label = torch.Tensor(data_batch['answer'], requires_grad=requires_grad)

    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)

    return img, qst, label


