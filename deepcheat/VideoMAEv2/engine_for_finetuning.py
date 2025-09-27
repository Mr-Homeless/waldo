# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import os
import sys
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.special import softmax
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import matplotlib.pyplot as plt
import utils

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)#.squeeze()
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    #breakpoint()
    for data_iter_step, (samples, targets, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        #if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
        #    for i, param_group in enumerate(optimizer.param_groups):
        #        if lr_schedule_values is not None:
        #            param_group["lr"] = lr_schedule_values[it] * param_group[
        #                "lr_scale"]
        #        if wd_schedule_values is not None and param_group[
        #                "weight_decay"] > 0:
        #            param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # mixup handle 3th & 4th dimension
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
            samples = samples.view(B, C, T, H, W)
            targets = targets[:,1].view(B,-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets,
                                             criterion)
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, output = train_class_batch(model, samples, targets,
                                                 criterion)

        loss_value = loss.item()



        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq

            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                # Removed debug print that was causing slowdowns
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, min_eval_score=None, max_eval_score=None, make_fig_path=''):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    all_outputs = []
    all_targets = []

    # First loop: Normalize all outputs to range 0-1
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        output = model(images).reshape(-1)
        loss = criterion(output, target.float())

        # Normalize output using sigmoid for values between 0 and 1

        all_outputs.extend(output.detach().cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    if min_eval_score is not None:
        min_val = min_eval_score
        max_val = max_eval_score
        all_outputs = np.array([(x - min_val) / (max_val - min_val) for x in all_outputs])
        return all_outputs


    min_val, max_val = min(all_outputs), max(all_outputs)
    
    print("min ", min_val   , " max ", max_val)
    all_outputs = np.array([(x - min_val) / (max_val - min_val) for x in all_outputs])

    # Second loop: Compute precision, recall, and F1 score
    preds = (all_outputs > 0.5).astype(float)  # Thresholding at 0.5

    precision = precision_score(all_targets, preds)
    recall = recall_score(all_targets, preds)
    f1 = f1_score(all_targets, preds)

    metric_logger.meters['precision'].update(precision, n=1)
    metric_logger.meters['recall'].update(recall, n=1)
    metric_logger.meters['f1'].update(f1, n=1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # After loop
    #breakpoint()
    all_probs = torch.sigmoid(torch.tensor(all_outputs).float()).cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    auc_pr = auc(recall, precision)
    

    if make_fig_path != '':
        # Plotting
        plt.figure(figsize=(10, 7))
        plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Thresholds')

        # Choose a set of probability thresholds
        threshold_values = np.linspace(0.1, 0.9, 9)
        for thresh in threshold_values:
            idx = (np.abs(thresholds - thresh)).argmin()
            plt.scatter(recall[idx], precision[idx], label=f'Prob={thresh:.1f}')
            plt.annotate(f'{thresh:.1f}', (recall[idx], precision[idx]), textcoords="offset points", xytext=(0,5), ha='center')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cheater_precision_recall_curve.png')
        # plt.show()


    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret["auc-pr"] = auc_pr
    return ret , all_probs, all_targets


@torch.no_grad()
def final_test_killshot(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    correct_count = 0
    total =0
    distance_from_correct = []
    kth_guess = []
    print(len(data_loader))
    for batch in data_loader:
        print(total)
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        #target = target.to(device, non_blocking=True)
        B = images.shape[0]
        # compute output
        with torch.amp.autocast('cuda'):
            #breakpoint()
            votes = np.zeros((images.shape[0],images.shape[2]))
            counts = np.zeros((images.shape[0],images.shape[2]))

            for i in range(16):

                output = model(images[:,:,i:i+16])
                votes[:,i:i+16] += output.cpu().numpy()
                counts[:,i:i+16] += np.ones((B,16))
                #loss = criterion(output, target)d
        out = votes/counts

        correct_count += (out.argmax(1) == target.argmax(1).numpy()).sum()
        total += B
        distance_from_correct.extend(np.abs(out.argmax(1) - target.argmax(1).numpy()))
        kth_guess.extend(np.sum(out > torch.from_numpy(out[:,15]).unsqueeze(-1).numpy(), axis=1))
        #breakpoint()
    print(images.shape)

    acc = correct_count / total


    return acc, distance_from_correct, kth_guess


@torch.no_grad()
def predict_killshot_scores(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    correct_count = 0
    total =0
    distance_from_correct = []
    kth_guess = []

    print(len(data_loader))
    #breakpoint()
    T = len(data_loader)
    votes = np.zeros(len(data_loader))
    counts = np.zeros(len(data_loader))
    B = 48
    i = 0
    V_L = 16
    #breakpoint()
    generator = data_loader.sequential_load(V_L)
    batched = []
    final = False
    while True:
        try:
            if not final:
                batch = next(generator)
                batched.append(batch)
            elif len(batched) == 0: break


            if len(batched) == B or final: 
                images = torch.stack(batched, dim=0).to(device, non_blocking=True)

                # compute output
                with torch.amp.autocast('cuda'):
                    outputs = model(images).cpu().numpy()
                for pred in outputs: 
                    votes[i:i+V_L]+=pred
                    counts[i:i+V_L]+=np.ones(len(pred))
                    i+=1
                batched = []

                if i % (5 * B) == 0:
                    print(f"frame {i} out of {T}")

            # Process batch
        except StopIteration:
            print("All batches processed!")
            final = True

    out = votes/counts

    return out


def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
