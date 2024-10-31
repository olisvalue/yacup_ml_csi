import logging
import wandb

import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from models.data_model import BatchDict, Postfix, TestResults, ValDict
from models.early_stopper import EarlyStopper
from models.modules import Bottleneck, Resnet50, TransformerEncoderModel
from models.new_modules import Model
from models.utils import (
    calculate_ranking_metrics,
    calculate_ranking_metrics_batched,
    dataloader_factory,
    dir_checker,
    reduce_func,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions,
    Zero
)

from models.loss import CenterLoss, FocalLoss, HardTripletLoss
from models.scheduler import UserDefineExponentialLR

# current_dir = os.getcwd()
# os.chdir('/home/olisvalue/contests/baseline/EfficientAT')
# sys.path.insert(0, os.getcwd())
# # from models.dymn.model import get_model as get_dymn
# os.chdir(current_dir)
# sys.path.pop(0)

# class WandbHandler(logging.Handler):
#     def emit(self, record):
#         log_message = self.format(record)
#         wandb.log({"log_message": log_message})  # Измените "log" на "log_message"

# Настройка логгера
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# wandb_handler = WandbHandler()
# wandb_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# wandb_handler.setFormatter(formatter)
# logger.addHandler(wandb_handler)

# Пример использования
logger.info("This log message will be sent to wandb!")


class TrainModule:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.state = "initializing"
        self.best_model_path: str = None
        self.num_classes = self.config["train"]["num_classes"]
        self.max_len = 50

        # self.model = get_dymn(pretrained_name="dymn10_as", num_classes=self.num_classes)
        # self.model = Resnet50(
        #     Bottleneck,
        #     num_channels=self.config["num_channels"],
        #     num_classes=self.num_classes,
        #     dropout=self.config["train"]["dropout"]
        # )

        self.model = Model(config)
        self.model.to(self.config["device"])

        # for name, param in self.model.named_parameters():
        #     if not name.startswith("fc"):
        #         param.requires_grad = False

        self.postfix: Postfix = {}

        #self.triplet_loss = nn.TripletMarginLoss(margin=config["train"]["triplet_margin"])
        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=config["train"]["triplet_margin"])
        # self.cls_loss = nn.CrossEntropyLoss(label_smoothing=config["train"]["smooth_factor"])

        # Loss
        if "alpha" in config["train"].keys():
            alpha = np.load(hp["ce"]["alpha"])
            alpha = 1.0 / (alpha + 1)
            alpha = alpha / np.sum(alpha)
            logger.info("use alpha with {}".format(len(alpha)))
        else:
            alpha = None
            logger.info("Not use alpha")

        self.triplet_loss = HardTripletLoss(margin=config["train"]["triplet_margin"])
        
        self.cls_loss = FocalLoss(alpha=alpha, gamma=config["train"]["gamma"],
                              num_cls=config["train"]["num_classes"])
        
        self.center_loss = CenterLoss(num_classes=config["train"]["num_classes"],
                                    feat_dim=config["embed_dim"], use_gpu=True)

        self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])
        self.optimizer = self.configure_optimizers()
        self.scheduler = UserDefineExponentialLR(
                optimizer=self.optimizer, gamma=config["train"]["lr_decay"],
                min_lr=config["train"]["min_lr"], last_epoch=-1)

        if self.config["device"] != "cpu":
            #self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["train"]["mixed_precision"])
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.config["train"]["mixed_precision"])

    def pipeline(self) -> None:
        self.config["val"]["output_dir"] = dir_checker(self.config["val"]["output_dir"])

        if self.config["train"]["model_ckpt"] is not None:
            checkpoint = torch.load(self.config["train"]["model_ckpt"])
            # checkpoint.pop('conv1.weight', None)
            # checkpoint.pop('fc.bias', None)
            self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["train"]["model_ckpt"]}')

        self.t_loader = dataloader_factory(config=self.config, data_split="train")
        self.v_loader = dataloader_factory(config=self.config, data_split="val")

        self.state = "running"

        self.pbar = trange(
            self.config["train"]["epochs"], disable=(not self.config["progress_bar"]), position=0, leave=True
        )
        for epoch in self.pbar:
            if self.state in ["early_stopped", "interrupted", "finished"]:
                return

            self.postfix["Epoch"] = epoch
            self.pbar.set_postfix(self.postfix)

            try:
                self.train_procedure()
            except KeyboardInterrupt:
                logger.warning("\nKeyboard Interrupt detected. Attempting gracefull shutdown...")
                self.state = "interrupted"
            except Exception as err:
                raise (err)

            #'''
            if self.state == "interrupted":
                self.validation_procedure()
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
                )
            #'''

        self.state = "finished"

    def validate(self) -> None:
        self.v_loader = dataloader_factory(config=self.config, data_split="val")
        self.state = "running"
        self.validation_procedure()
        self.state = "finished"

    def test(self) -> None:
        self.test_loader = dataloader_factory(config=self.config, data_split="test")
        self.test_results: TestResults = {}
        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path), strict=False)
            logger.info(f"Best model loaded from checkpoint: {self.best_model_path}")
        elif self.config["test"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["test"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["test"]["model_ckpt"]}')
        elif self.state == "initializing":
            logger.warning("Warning: Testing with random weights")

        self.state = "running"
        self.test_procedure()
        self.state = "finished"

    def train_procedure(self) -> None:
        self.model.train()
        train_loss_list = []
        train_cls_loss_list = []
        train_triplet_loss_list = []
        train_center_loss_list = []

        self.max_len = self.t_loader.dataset.max_len
        for step, batch in tqdm(
            enumerate(self.t_loader),
            total=len(self.t_loader),
            disable=(not self.config["progress_bar"]),
            position=2,
            leave=False,
        ):
            train_step = self.training_step(batch)
            self.postfix["train_loss_step"] = float(f"{train_step['train_loss_step']:.3f}")
            train_loss_list.append(train_step["train_loss_step"])
            self.postfix["train_cls_loss_step"] = float(f"{train_step['train_cls_loss']:.3f}")
            train_cls_loss_list.append(train_step["train_cls_loss"])
            self.postfix["train_triplet_loss_step"] = float(f"{train_step['train_triplet_loss']:.3f}")
            train_triplet_loss_list.append(train_step["train_triplet_loss"])
            self.postfix["train_center_loss_step"] = float(f"{train_step['train_center_loss']:.3f}")
            train_center_loss_list.append(train_step["train_center_loss"])
            self.pbar.set_postfix(
                {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
            )
            if step % self.config["train"]["log_steps"] == 0:
                save_logs(
                    dict(
                        epoch=self.postfix["Epoch"],
                        seq_len=self.max_len,
                        step=step,
                        train_loss_step= self.postfix["train_loss_step"],
                        train_cls_loss_step=self.postfix["train_cls_loss_step"],
                        train_triplet_loss_step=self.postfix["train_triplet_loss_step"],
                        train_center_loss_step=self.postfix["train_center_loss_step"],
                    ),
                    output_dir=self.config["val"]["output_dir"],
                    name="log_steps",
                    use_wandb=self.config["use_wandb"]
                )
            if (step+1) % self.config["train"]["lr_update_steps"] == 0:
                self.scheduler.step()
            if (step+1) % self.config["val"]["val_period"] == 0:
                self.validation_procedure()

        train_loss = torch.tensor(train_loss_list)
        train_cls_loss = torch.tensor(train_cls_loss_list)
        train_triplet_loss = torch.tensor(train_triplet_loss_list)
        train_center_loss = torch.tensor(train_center_loss_list)
        self.postfix["train_loss"] = train_loss.mean().item()
        self.postfix["train_cls_loss"] = train_cls_loss.mean().item()
        self.postfix["train_triplet_loss"] = train_triplet_loss.mean().item()
        self.postfix["train_center_loss"] = train_center_loss.mean().item()
        
        self.validation_procedure()
        self.overfit_check()
        self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}})

    def training_step(self, batch: BatchDict) -> Dict[str, float]:
        with torch.autocast(
            device_type=self.config["device"].split(":")[0], enabled=self.config["train"]["mixed_precision"]
        ):
            # print("*"*50)
            # print(batch["anchor"].shape)
            # print("*"*50)
            # batch["anchor"] = batch["anchor"].unsqueeze(1)
            # batch["positive"] = batch["positive"].unsqueeze(1)
            # batch["negative"] = batch["negative"].unsqueeze(1)

            anchor = self.model.forward(batch["anchor"].to(self.config["device"]))
            positive = self.model.forward(batch["positive"].to(self.config["device"]))
            negative = self.model.forward(batch["negative"].to(self.config["device"]))
            # labels = nn.functional.one_hot(batch["anchor_label"].long(), num_classes=self.num_classes)
            labels_anchor = batch["anchor_label"].long()
            labels_neg = batch["negative_label"].long()


            embeddings = torch.cat([anchor["f_t"], positive["f_t"], negative["f_t"]], dim=0)

            batch_size = embeddings.shape[0]
            
            labels = torch.cat([
                labels_anchor,  
                labels_anchor, 
                labels_neg
            ]).to(self.config["device"]).long()

            tri_loss = self.triplet_loss(embeddings, labels.to(self.config["device"]))*self.config["ce"]["weight"]
            ce_loss = self.cls_loss(anchor["cls"], labels_anchor.float().to(self.config["device"]))*self.config["triplet"]["weight"]

            if self.config["center"]["weight"] < 0.001:
                center_loss = Zero()
            else:
                center_loss = self.center_loss(anchor["f_c"], labels_anchor.to(self.config["device"]))*self.config["center"]["weight"]

            loss = tri_loss + ce_loss + center_loss


        self.optimizer.zero_grad()
        if self.config["device"] != "cpu":
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)  # Размасштабирование градиентов перед отсечением
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            

        return {"train_loss_step": loss.item(), 
                "train_triplet_loss": tri_loss.item(), 
                "train_cls_loss": ce_loss.item(),
                "train_center_loss": center_loss.item()}

    def validation_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[int, torch.Tensor] = {}
        for batch in tqdm(self.v_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            val_dict = self.validation_step(batch)
            if val_dict["f_t"].ndim == 1:
                val_dict["f_c"] = val_dict["f_c"].unsqueeze(0)
                val_dict["f_t"] = val_dict["f_t"].unsqueeze(0)
            for anchor_id, triplet_embedding, embedding in zip(val_dict["anchor_id"], val_dict["f_t"], val_dict["f_c"]):
                embeddings[anchor_id] = torch.stack([triplet_embedding, embedding])

        val_outputs = self.validation_epoch_end(embeddings)
        logger.info(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.postfix.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )

        if self.config["val"]["save_val_outputs"]:
            val_outputs["val_embeddings"] = torch.stack(list(embeddings.values()))[:, 1].numpy()
            save_predictions(val_outputs, output_dir=self.config["val"]["output_dir"])
            save_logs(self.postfix, output_dir=self.config["val"]["output_dir"],
                      use_wandb=self.config["use_wandb"])
        self.model.train()

    def validation_epoch_end(self, outputs: Dict[int, torch.Tensor]) -> Dict[str, np.ndarray]:
        #val_loss = torch.zeros(len(outputs))
        #pos_ids = []
        #neg_ids = []
        clique_ids = []
        for k, (anchor_id, embeddings) in enumerate(outputs.items()):
            #clique_id, pos_id, neg_id = self.v_loader.dataset._triplet_sampling(anchor_id)
            #val_loss[k] = self.triplet_loss(embeddings[0], outputs[pos_id][0], outputs[neg_id][0]).item()
            #pos_ids.append(pos_id)
            #neg_ids.append(neg_id)
            clique_id = self.v_loader.dataset.version2clique.loc[anchor_id, 'clique']
            clique_ids.append(clique_id)
        #anchor_ids = np.stack(list(outputs.keys()))
        preds = torch.stack(list(outputs.values()))[:, 1]
        #self.postfix["val_loss"] = val_loss.mean().item()
        rranks, average_precisions = calculate_ranking_metrics_batched(embeddings=preds.numpy(), cliques=clique_ids)
        self.postfix["mrr"] = rranks.mean()
        self.postfix["mAP"] = average_precisions.mean()
        return {
            #"triplet_ids": np.stack(list(zip(clique_ids, anchor_ids, pos_ids, neg_ids))),
            "rranks": rranks,
            "average_precisions": average_precisions,
        }

    def validation_step(self, batch: BatchDict) -> ValDict:
        anchor_id = batch["anchor_id"]
        features = self.model.forward(batch["anchor"].to(self.config["device"]))

        return {
            "anchor_id": anchor_id.numpy(),
            "f_t": features["f_t"].squeeze(0).detach().cpu(),
            "f_c": features["f_c"].squeeze(0).detach().cpu(),
        }

    def test_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[str, torch.Tensor] = {}
        trackids: List[int] = []
        embeddings: List[np.array] = []

        i = 0
        for batch in tqdm(self.test_loader, disable=(not self.config["progress_bar"])):
            test_dict = self.validation_step(batch)
            if test_dict["f_c"].ndim == 1:
                test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
            for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["f_c"]):
                trackids.append(anchor_id)
                embeddings.append(embedding.numpy())
        predictions = []
        for chunk_result in pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func, working_memory=40):
            for query_indx, query_nearest_items in chunk_result:
                predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
        save_test_predictions(predictions, output_dir=self.config["test"]["output_dir"])
    # def validation_step(self, batch: BatchDict):
    #     anchor_id = batch["anchor_id"]
    #     features = batch["anchor"].to(self.config["device"])

    #     # print('*'*50)
    #     # print(features.shape)
    #     # print(features)
    #     # print('*'*50)

    #     # Averaging embeddings along the second dimension (axis 1)
    #     avg_features_f_c = features.mean(dim=2)

    #     # print(avg_features_f_c.shape)
    #     # print(avg_features_f_c)
        
    #     # exit()

    #     return {
    #         "anchor_id": anchor_id.numpy(),
    #         "f_c": avg_features_f_c.squeeze(0).detach().cpu(),
    #     }

    # def test_procedure(self) -> None:
    #     # self.model.eval()
    #     trackids: List[int] = []
    #     embeddings: List[np.array] = []
    #     for batch in tqdm(self.test_loader, disable=(not self.config["progress_bar"])):
    #         test_dict = self.validation_step(batch)
    #         if test_dict["f_c"].ndim == 1:
    #             test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
    #         for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["f_c"]):
    #             trackids.append(anchor_id)
    #             embeddings.append(embedding.numpy())
    #     predictions = []
    #     for chunk_result in pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func, working_memory=100):
    #         for query_indx, query_nearest_items in chunk_result:
    #             predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
    #     save_test_predictions(predictions, output_dir=self.config["test"]["output_dir"])

    def overfit_check(self) -> None:
        if self.early_stop(self.postfix["mAP"]):
            logger.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopped"

        if self.early_stop.counter > 0:
            logger.info("\nValidation mAP was not improved")
        else:
            logger.info(f"\nMetric improved. New best score: {self.early_stop.max_validation_mAP:.3f}")
            save_best_log(self.postfix, output_dir=self.config["val"]["output_dir"],
                          use_wandb=self.config["use_wandb"])

            logger.info("Saving model...")
            epoch = self.postfix["Epoch"]
            max_secs = self.max_len
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.config["val"]["output_dir"], "model", f"best-model-{epoch=}-{max_secs=}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            betas=[self.config["train"]["adam_b1"], self.config["train"]["adam_b2"]]
        )
        return optimizer
