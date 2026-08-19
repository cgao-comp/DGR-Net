import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        positive = torch.eq(labels, labels.T).float().to(features.device)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        non_self = 1 - torch.eye(features.size(0), device=features.device)
        positive *= non_self
        log_prob = logits - torch.log((torch.exp(logits) * non_self).sum(1, keepdim=True) + 1e-6)
        return -((positive * log_prob).sum(1) / (positive.sum(1) + 1e-6)).mean()


class DGR:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.scl_loss = SupConLoss(temperature=0.1)
        # Table 3 values are dataset specific and provided through the config.
        self.gamma_aux, self.gamma_decomp = args.gamma_aux, args.gamma_decomp
        self.gamma_scl, self.gamma_reg = args.gamma_scl, args.gamma_reg
        self.lambda_rec, self.lambda_cyc = 1.0, 0.1
        self.lambda_mar, self.lambda_orth = 0.1, 0.1
        self.history = {}  # dataset index -> [language, audio, vision] historical MAE

    @staticmethod
    def _class_labels(labels):
        """Map continuous [-3, 3] sentiment labels to the paper's seven classes."""
        return labels.view(-1).round().clamp(-3, 3).long() + 3

    def _cross_modal_triplet(self, shared, class_labels):
        """Eq. 7: cross-modal positives, same-modal semi-hard negatives, margin 0.2."""
        vectors = [F.normalize(x.mean(dim=-1), dim=1) for x in shared]
        losses = []
        for anchor_mod, anchor_vec in enumerate(vectors):
            sim_same = anchor_vec @ anchor_vec.T
            for i in range(anchor_vec.size(0)):
                positives = []
                for positive_mod, positive_vec in enumerate(vectors):
                    if positive_mod == anchor_mod:
                        continue
                    candidate = torch.where(class_labels == class_labels[i])[0]
                    candidate = candidate[candidate != i]
                    if candidate.numel():
                        positives.append((anchor_vec[i] @ positive_vec[candidate].T).max())
                negatives = torch.where(class_labels != class_labels[i])[0]
                if positives and negatives.numel():
                    pos = torch.stack(positives).mean()
                    neg_sims = sim_same[i, negatives]
                    # Semi-hard: negative less similar than positive but closest to it.
                    semi_hard = neg_sims[neg_sims < pos]
                    neg = (semi_hard if semi_hard.numel() else neg_sims).max()
                    losses.append(F.relu(0.2 - pos + neg))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=class_labels.device)

    def _historical_ranking_loss(self, sample_indices, modal_errors, weights):
        historic = torch.tensor([self.history.get(int(idx), errs.detach().tolist())
                                 for idx, errs in zip(sample_indices, modal_errors)], device=weights.device)
        loss = torch.zeros((), device=weights.device)
        for modality in range(3):
            error_diff = historic[:, modality:modality + 1] - historic[:, modality:modality + 1].T
            weight_diff = weights[:, modality:modality + 1] - weights[:, modality:modality + 1].T
            loss += F.relu(weight_diff) .mul((error_diff > 0).float()).mean()
        # Update after calculating the loss; this makes history refer only to prior observations.
        for idx, errs in zip(sample_indices, modal_errors):
            old = self.history.get(int(idx), errs.detach().tolist())
            self.history[int(idx)] = [0.9 * value + 0.1 * float(new) for value, new in zip(old, errs.detach().cpu())]
        return loss

    def _losses(self, net, output, labels, sample_indices):
        class_labels = self._class_labels(labels)
        main = self.criterion(output['output_logit'], labels)
        reg_losses = torch.stack([self.criterion(logit, labels) for logit in output['reg_logits']]).sum()
        ce_losses = torch.stack([self.ce(logit, class_labels) for logit in output['cls_logits']]).sum()
        auxiliary = reg_losses + self.args.lambda_ce * ce_losses
        info = output['decomp_info']
        mse = nn.MSELoss()
        reconstruction = sum(mse(a, b) for a, b in zip(info['recon'], info['original']))
        cycle = sum(mse(encoder(recon), private) for encoder, recon, private in zip(
            (net.enc_private_l, net.enc_private_a, net.enc_private_v), info['recon'], info['private']))
        orthogonality = sum(torch.mean(torch.abs(F.cosine_similarity(shared.flatten(1), private.flatten(1))))
                            for shared, private in zip(info['shared'], info['private']))
        triplet = self._cross_modal_triplet(info['shared'], class_labels)
        decomposition = self.lambda_rec * reconstruction + self.lambda_cyc * cycle + self.lambda_mar * triplet + self.lambda_orth * orthogonality
        modal_errors = torch.cat([torch.abs(logit.detach() - labels) for logit in output['reg_logits']], dim=1)
        ranking = self._historical_ranking_loss(sample_indices, modal_errors, output['gate_weights'])
        contrastive = self.scl_loss(output['contrastive_feat'], class_labels)
        total = main + self.gamma_aux * auxiliary + self.gamma_decomp * decomposition + self.gamma_scl * contrastive + self.gamma_reg * ranking
        return total, {'Task': main, 'Aux': auxiliary, 'SCL': contrastive, 'Reg': ranking, 'Decomp': decomposition,
                       'Rec': reconstruction, 'Cyc': cycle, 'Mar': triplet, 'Orth': orthogonality}

    def do_train(self, model, dataloader, return_epoch_results=False):
        net = model[0]
        optimizer = optim.AdamW(net.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        warmup_epochs = self.args.warmup_epochs
        warmup = LambdaLR(optimizer, lambda epoch: min(1.0, (epoch + 1) / max(1, warmup_epochs)))
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, self.args.max_epochs - warmup_epochs))
        best_valid, best_epoch = float('inf'), 0
        epoch_results = {'train': [], 'valid': [], 'test': []} if return_epoch_results else None
        for epoch in range(1, self.args.max_epochs + 1):
            net.train()
            meters = {key: 0.0 for key in ('Total', 'Task', 'Aux', 'SCL', 'Reg', 'Decomp', 'Rec', 'Cyc', 'Mar', 'Orth')}
            predictions, targets = [], []
            for batch in tqdm(dataloader['train'], leave=False):
                optimizer.zero_grad()
                labels = batch['labels']['M'].to(self.args.device).view(-1, 1)
                output = net(batch['text'].to(self.args.device), batch['audio'].to(self.args.device), batch['vision'].to(self.args.device))
                total, parts = self._losses(net, output, labels, batch['index'])
                total.backward()
                if self.args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(net.parameters(), self.args.grad_clip)
                optimizer.step()
                meters['Total'] += total.item()
                for key, value in parts.items(): meters[key] += value.item()
                predictions.append(output['output_logit'].detach().cpu()); targets.append(labels.cpu())
            if epoch <= warmup_epochs: warmup.step()
            else: cosine.step()
            for key in meters: meters[key] /= len(dataloader['train'])
            train_results = self.metrics(torch.cat(predictions), torch.cat(targets)); train_results['Loss'] = meters['Total']
            validation = self.do_test(net, dataloader['valid'], 'VAL')
            if validation['Loss'] < best_valid - 1e-6:
                best_valid, best_epoch = validation['Loss'], epoch
                torch.save(net.state_dict(), self.args.model_save_path)
            if return_epoch_results:
                epoch_results['train'].append(train_results); epoch_results['valid'].append(validation)
            logger.info('Epoch %d: %s', epoch, dict_to_str({**meters, **validation}))
            if epoch - best_epoch >= self.args.early_stop:
                logger.info('Early stopping at epoch %d.', epoch)
                break
        return epoch_results

    def do_test(self, model, dataloader, mode='VAL', return_sample_results=False):
        model.eval(); predictions, targets, loss = [], [], 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=mode, leave=False):
                labels = batch['labels']['M'].to(self.args.device).view(-1, 1)
                output = model(batch['text'].to(self.args.device), batch['audio'].to(self.args.device), batch['vision'].to(self.args.device))
                loss += self.criterion(output['output_logit'], labels).item()
                predictions.append(output['output_logit'].cpu()); targets.append(labels.cpu())
        results = self.metrics(torch.cat(predictions), torch.cat(targets)); results['Loss'] = round(loss / len(dataloader), 4)
        logger.info('%s-(%s) >> %s', mode, self.args.model_name, dict_to_str(results))
        return results
