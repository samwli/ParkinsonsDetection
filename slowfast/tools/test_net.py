#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, balanced_accuracy_score
import sklearn.metrics

#from cam_framework import CamFramework 

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    #model.eval()
    model.eval()

    # model = CamFramework(model)

    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta, bboxes) in enumerate(
        test_loader
    ):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()

            print("bboxes: ", bboxes.shape)
            #print("inputs: ", inputs.shape)
            print("labels: ", labels.shape)
            bboxes = bboxes.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            # print("FORWARD PASS HIT")
            # print("input len: ", len(inputs))
            # print("input shape: ", inputs[0].shape)
        
            # target_layers = [model.layer4[-1]]
            # target_layers = [model.s5.pathway0_res2.branch2.c]
            # cam = GradCAM(model=model, target_layers=target_layers)

            # # We have to specify the target we want to generate
            # # the Class Activation Maps for.
            # # If targets is None, the highest scoring category
            # # will be used for every image in the batch.
            # # Here we use ClassifierOutputTarget, but you can define your own custom targets
            # # That are, for example, combinations of categories, or specific outputs in a non standard model.

            # targets = [ClassifierOutputTarget(281)]
            # targets = None

            # input_tensor = inputs[0]
            # input tensor is x

            # i goes over the batch dimension
            # for i in range(input_tensor.size(0)):
            #     # j goes over the T dimension
            #     for j in range(input_tensor.size(1)):
            #         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                # cam_grayscale = cam(input_tensor=input_tensor[i, j, :, :].unsqueeze(0),
                #                                 method="gradcam",
                #                                 target_category=target_category)

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            # grayscale_cam = cam(input_tensor=inputs, targets=targets)

            # # In this example grayscale_cam has only one image in the batch:
            # grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # inputs[0].requires_grad = True
            # print("the real requires grad: ", inputs[0].requires_grad)

            print("Inputs shape: ", inputs[0].shape)
            preds = model(inputs, bboxes)
            # print("preds[0].shape: ", preds[0].shape)
            # print("Outputs shape: ", preds.shape)
            # # print("preds shape: ", preds.shape)
            # print("requires grad: ", preds.requires_grad)
            # pred = preds.argmax(dim=1)
            # print("the truly real requires grad: ", pred.requires_grad)
            # print("pred: ", pred)
            # loss = preds[:, pred.item()]
            # loss.requires_grad = True
            # print("loss: ", loss.requires_grad)
            # loss.backward()
            # gradients = model.get_activations_gradient()

            # print("gradients shape: ", gradients.shape)
            # pool the gradients across the channels
            # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            # print("predicted class score : ", )
            # print("predicted class: ", pred)
            # 1/0

            # 1/0
        
        
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()

        print("preds shape: ", preds.shape)
        print("labels shape: ", labels.shape)
        print("video_idx shape: ", video_idx.shape)

        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics((1, 2))
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrainied video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    out_str_prefix = "lin" if cfg.MODEL.DETACH_FINAL_FC else ""

    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=False)

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        train_loader = loader.construct_loader(cfg, "train")
        out_str_prefix = "knn"
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES
            if not cfg.TASK == "ssl"
            else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None
    print(model)
    
    if cfg.MODEL.NUM_CLASSES < 11:
        k = "top2_acc"
    else:
        k = "top5_acc"
    
    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    preds_ =  [np.argmax(item) for item in test_meter.video_preds.tolist()]
    # one hot labels
    """
    y_hot = np.zeros((len(test_meter.video_labels), cfg.MODEL.NUM_CLASSES))
    y_hot[np.arange(len(test_meter.video_labels)), test_meter.video_labels] = 1
    print("yhot shapes are")
    print(y_hot.shape, test_meter.video_labels.shape)
    """
    auroc_score = roc_auc_score(y_true=test_meter.video_labels, y_score=test_meter.video_preds, average='macro', multi_class = 'ovr')
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = test_meter.video_labels, y_score = test_meter.video_preds[:, 1], pos_label = 1) 
    #auroc_score = sklearn.metrics.auc(fpr, tpr)
    f1_macro = sklearn.metrics.f1_score(y_true=test_meter.video_labels, y_pred=preds_, average='macro')
    #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = test_meter.video_labels, probas_pred = test_meter.video_preds[:, 1], pos_label = 1)
    #auc_precision_recall = sklearn.metrics.auc(recall, precision)
    matrix = confusion_matrix(test_meter.video_labels, preds_)
    top1_macro = matrix.diagonal()/matrix.sum(axis=1)
    print("top1_macro:")
    print(top1_macro)
    mse = (np.square(np.array(test_meter.video_labels) - np.array(preds_))).mean()

    bin_labels = torch.zeros_like(test_meter.video_labels)
    bin_labels[test_meter.video_labels > 0] = 1

    neg = test_meter.video_preds[:,0].view(test_meter.video_preds.shape[0],1)
    pos = torch.sum(test_meter.video_preds[:,1:],dim=1).view(test_meter.video_preds.shape[0],1)  
    bin_preds = torch.cat([neg,pos],-1)

    bin_preds_ = torch.zeros_like(torch.Tensor(preds_))
    bin_preds_[torch.Tensor(preds_) > 0] = 1

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = bin_labels, y_score = bin_preds[:, 1], pos_label = 1) 
    auroc_bin = sklearn.metrics.auc(fpr, tpr)
    f1_bin = sklearn.metrics.f1_score(y_true=bin_labels, y_pred=bin_preds_)
    bin_recall = recall_score(y_true=bin_labels, y_pred=bin_preds_)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = bin_labels, probas_pred = bin_preds[:, 1], pos_label = 1)
    auc_precision_recall = sklearn.metrics.auc(recall, precision)
    bin_acc = accuracy_score(bin_labels, bin_preds_)
    if writer is not None:
        writer.close()
    result_string = (
        "dataset: {}{}, Top1 Acc: {}, top1_macro: {}, top1_macro_pos: {}, F1_macro: {}, auroc_macro_ovr: {}, mse: {}, bin_acc: {}, f1_bin: {}, bin_recall: {}, auroc_bin: {}, auprc_bin: {}"
        "".format(
            cfg.TEST.DATASET[0],
            cfg.MODEL.NUM_CLASSES,
            test_meter.stats["top1_acc"],
            np.mean(top1_macro),
            np.mean(np.array(top1_macro[1:])),
            f1_macro,
            auroc_score,
            mse,
            bin_acc,
            f1_bin,
            bin_recall,
            auroc_bin,
            auc_precision_recall
        )
    )
    logger.info("testing done: {}".format(result_string))
    
    with open('./experiments/results.txt', 'a+') as f:
        f.write(" Test accuracy: ")
        f.write(result_string + "\n")

    return result_string