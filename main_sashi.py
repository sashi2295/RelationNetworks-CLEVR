import argparse
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils_sashi

from torch.nn.utils import clip_grad_norm
from clevr_dataset_sashi import ClevrDataset
from model_sashi import RN
from torchvision import transforms
from torch.utils.data import DataLoader


def train(data, model, optimizer, epoch, args):

    model.train()

    avg_loss = 0.0
    n_batches = 0

    for batch_idx, sample_batched in enumerate(data):
        #img, qst, label = utils_sashi.load_tensor_data(sample_batched, args.cuda, args.invert_questions, volatile=True)
        img, qst, label = sample_batched['image'], sample_batched['question'], sample_batched['answer']
        label = (label - 1).squeeze()

        if args.cuda:
            img, qst, label = img.cuda(), qst.cuda(), label.cuda()

        optimizer.zero_grad()
        output = model(img, qst)
        loss = F.nll_loss(output, label)
        loss.backward()

#       if args.clip_norm:
#           clip_grad_norm(model.parameters(), args.clip_norm)

#       avg_loss += loss.data[0]
        avg_loss += loss.item()
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0


def reload_loaders(clevr_dataset_train, clevr_dataset_test, train_bs, test_bs, state_description=False):

    clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs, shuffle=False,
                                    num_workers=2, collate_fn=utils_sashi.collate_samples_from_pixels)
    clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs, shuffle=False,
                                   num_workers=2, collate_fn=utils_sashi.collate_samples_from_pixels)

    return clevr_train_loader, clevr_test_loader


def initialize_dataset(clevr_dir, dictionaries):

    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.Pad(8),
                                           transforms.RandomCrop((128, 128)),
                                           transforms.RandomRotation(2.8),
                                           transforms.ToTensor()])

#    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
#                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])


    clevr_dataset_train = ClevrDataset(clevr_dir, True, dictionaries, train_transforms)
    clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, test_transforms)

    return clevr_dataset_train, clevr_dataset_test


def main(args):

    with open(args.config) as config_file:
        hyp = json.load(config_file)['hyperparams'][args.model]

    print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Building dictionaries from all words in the dataset')
    dictionaries = utils_sashi.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initialising CLEVR dataset...')
    clevr_dataset_train, clevr_dataset_test = initialize_dataset(args.clevr_dir, dictionaries)
    print('CLEVR dataset initialised')

    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])

    model = RN(args, hyp)

    if args.cuda:
        model.cuda()

    bs = args.batch_size
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        clevr_train_loader, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test,
                                                               bs, args.test_batch_size, hyp['state_description'])

        train(clevr_train_loader, model, optimizer, epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.000005)
    #parser.add_argument('--clip-norm', type=int, default=50)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=10)
    #parser.add_argument('--resume', type=str)
    parser.add_argument('--clevr-dir', type=str, default='/home/iki/sashi/robotkoop/CLEVR_v1.0/')
    parser.add_argument('--model', type=str, default='original-fp')
    parser.add_argument('--no-invert-questions', action='store_true', default=True)
    #parser.add_argument('--test', action='store_true', default=False)
    #parser.add_argument('--conv-transfer-learn', type=str)
    #parser.add_argument('--lr-max', type=float, default=0.0005)
    #parser.add_argument('--lr-gamma', type=float, default=2)
    #parser.add_argument('--lr-step', type=int, default=-1)
    #parser.add_argument('--bs-max', type=int, default=-1)
    #parser.add_argument('--bs-gamma', type=float, default=1)
    #parser.add_argument('--bs-step', type=int, default=20)
    #parser.add_argument('--dropout', type=float, default=-1)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--question-injection', type=int, default=-1)

    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions

    main(args)
