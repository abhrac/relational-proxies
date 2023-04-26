import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import ProxyAnchorLoss
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from networks.ast import AST
from networks.relation_net import RelationNet
from utils import constants


class RelationalProxies(nn.Module):
    def __init__(self, backbone, num_classes, logdir):
        super(RelationalProxies, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR

        self.backbone = backbone
        self.aggregator = AST(num_inputs=backbone.num_local, dim=self.feature_dim, depth=3, heads=3, mlp_dim=256)
        self.relation_net = RelationNet(feature_dim=self.feature_dim)

        self.optimizer = torch.optim.SGD(chain(
            backbone.parameters(), self.aggregator.parameters(), self.relation_net.parameters()),
            lr=self.lr, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)

        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.proxy_criterion = ProxyAnchorLoss(num_classes=num_classes, embedding_size=self.feature_dim)

        self.writer = SummaryWriter(logdir)

    def train_one_epoch(self, trainloader, epoch, save_path):
        print('Training %d epoch' % epoch)
        self.train()
        device = self.proxy_criterion.proxies.device  # hacky, but keeps the arg list clean
        epoch_state = {'loss': 0, 'correct': 0}
        for i, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            self.optimizer.zero_grad()

            global_repr, summary_repr, relation_repr = self.compute_reprs(im)
            loss = self.proxy_criterion(relation_repr, labels)

            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict(global_repr, summary_repr, relation_repr, labels, epoch_state)

        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            print('Testing %d epoch' % epoch)
            self.eval()
            device = self.proxy_criterion.proxies.device  # hacky, but keeps the arg list clean
            epoch_state = {'loss': 0, 'correct': 0}
            for i, data in enumerate(tqdm(testloader)):
                im, labels = data
                im, labels = im.to(device), labels.to(device)

                global_repr, summary_repr, relation_repr = self.compute_reprs(im)
                epoch_state = self.predict(global_repr, summary_repr, relation_repr, labels, epoch_state)

                loss = self.proxy_criterion(relation_repr, labels)
                epoch_state['loss'] += loss.item()

            self.post_epoch('Test', epoch, epoch_state, len(testloader.dataset), None)

    def compute_reprs(self, im):
        global_embed, local_embeds = self.backbone(im)

        summary_repr = self.aggregator(local_embeds)
        relation_repr = self.relation_net(global_embed, summary_repr)

        return global_embed, summary_repr, relation_repr

    @torch.no_grad()
    def predict(self, global_repr, summary_repr, relation_repr, labels, epoch_state):
        global_logits = F.linear(global_repr, self.proxy_criterion.proxies)
        summary_logits = F.linear(summary_repr, self.proxy_criterion.proxies)
        relation_logits = F.linear(relation_repr, self.proxy_criterion.proxies)

        mean_logits = (global_logits + summary_logits + relation_logits) / 3
        pred = mean_logits.max(1, keepdim=True)[1]
        epoch_state['correct'] += pred.eq(labels.view_as(pred)).sum().item()

        return epoch_state

    @torch.no_grad()
    def post_epoch(self, phase, epoch, epoch_state, num_samples, save_path):
        accuracy = epoch_state['correct'] / num_samples
        loss = epoch_state['loss']

        print(f'{phase} Loss: {loss}')
        print(f'{phase} Accuracy: {accuracy * 100}%')
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)

        if (phase == 'Train') and ((epoch % constants.SAVE_EVERY == 0) or (epoch == constants.END_EPOCH)):
            self.scheduler.step()
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'learning_rate': self.lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

    def post_job(self):
        """Post-job actions"""
        self.writer.flush()
        self.writer.close()
