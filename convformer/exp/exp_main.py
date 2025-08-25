import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s "
           "[%(filename)s:%(lineno)d %(funcName)s] %(message)s",
    datefmt="%d-%m-%YT%H:%M:%S",
    force=True,  # reconfigure safely in REPL/notebooks/apps
)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, Performer
from models.Convformer_Family import Informer_ConvStem, Informer_FAVOR, Informer_Decomp, Convformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import json

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Performer': Performer,
            'Informer_ConvStem': Informer_ConvStem,
            'Informer_FAVOR': Informer_FAVOR,
            'Informer_Decomp': Informer_Decomp,
            'Convformer': Convformer
        }
        logging.info(f'RUNNING MODEL: {self.args.model}')
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args.use_torch_compile:
            model = torch.compile(model, mode="max-autotune")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # last = batch_y[:, self.args.label_len-1:self.args.label_len, :]            # (B,1,C)        #
        # pad  = last.repeat(1, self.args.pred_len, 1)                               # (B,pred_len,C) #
        # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], pad], dim=1).to(self.device)      #

        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def train_epoch(self, train_loader, criterion, optimizer, epoch, scaler=None):
        self.model.train()
        
        running_loss = 0.
        total_size   = 0
        iter_count   = 0
        train_steps = len(train_loader)
        time_now = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            batch_x = batch_x.float().to(self.device, non_blocking=False)
            batch_y = batch_y.float().to(self.device, non_blocking=False)

            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=False)
            batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=False)

            batch_size = batch_x.size(0)
            optimizer.zero_grad()

            outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(outputs, batch_y)

            if self.args.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if self.args.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                optimizer.step()

            with torch.no_grad():
                loss = loss.item()

                # accumulate loss
                running_loss += loss * batch_size
                total_size   += batch_size

                # evaluate time
                if (i + 1) % 100 == 0:
                    logging.info(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    minutes, leftover_seconds = left_time//60, left_time%60
                    logging.info(f'\tspeed: {speed:.4f}s/iter; left time: {minutes}m {leftover_seconds:.4f}s')
                    iter_count = 0
                    time_now = time.time()

        loss = running_loss / total_size
        return loss

    @torch.no_grad()
    def val_epoch(self, vali_loader, criterion):
        self.model.eval()
        
        running_loss = 0.
        total_size   = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device, non_blocking=False)
            batch_y = batch_y.float().to(self.device, non_blocking=False)

            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=False)
            batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=False)

            batch_size = batch_x.size(0)

            outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

            pred = outputs  # outputs.detach().cpu()
            true = batch_y  # batch_y.detach().cpu()

            loss = criterion(pred, true)

            running_loss += loss.item() * batch_size
            total_size   += batch_size

        loss = running_loss / total_size
        return loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val') 
        test_data, test_loader   = self._get_data(flag='test') 

        path = os.path.join('../generated', setting, self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion   = self._select_criterion()

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            
            train_loss = self.train_epoch(train_loader, criterion, model_optim, epoch, scaler) 
            vali_loss  = self.val_epoch(vali_loader, criterion) 
            test_loss  = self.val_epoch(test_loader, criterion)
            
            logging.info(f'Epoch: {epoch + 1}, cost time: {time.time() - epoch_time}')
            logging.info(f'Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logging.info(f'Early stopping triggered')
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return  

    @torch.inference_mode()
    def test(self, setting, test=0):
        self.model.eval()
        test_data, test_loader = self._get_data(flag='test')
        if test:
            logging.info(f'Loading model')
            path = os.path.join('../generated', setting, self.args.checkpoints)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []
        visual_dir = os.path.join('../generated', setting, 'visual')
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device, non_blocking=False)
            batch_y = batch_y.float().to(self.device, non_blocking=False)

            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=False)
            batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=False)

            outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

            outputs = outputs.detach().cpu().numpy() 
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                inp = batch_x.detach().cpu().numpy()
                gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, self.args.seq_len, os.path.join(visual_dir, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logging.info(f'test shape: {preds.shape} {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logging.info(f'test shape: {preds.shape} {trues.shape}')

        # result save
        results_path = os.path.join('../generated', setting, 'results.jsonl')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        record = {
            "mae":  float(mae),  "mse":  float(mse), "rmse": float(rmse),
            "mape": float(mape), "mspe": float(mspe)
        }

        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'metrics': record}, ensure_ascii=False) + "\n")

        return

    @torch.inference_mode()
    def predict(self, setting, load=False):
        self.model.eval()
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join('../generated', setting, self.args.checkpoints)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = f'../generated/{setting}/predict_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
