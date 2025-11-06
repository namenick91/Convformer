import logging
logging.basicConfig(
    level=logging.INFO, # was WARNING
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s "
           "[%(filename)s:%(lineno)d %(funcName)s] %(message)s",
    datefmt="%d-%m-%YT%H:%M:%S",
    force=True,  # reconfigure safely in REPL/notebooks/apps
)

import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","1","y"): return True
    if v in ("no","false","f","0","n"): return False
    raise argparse.ArgumentTypeError("boolean value expected")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    seed = 2025
    set_seed(seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--num_rand_features', type=int, default=256, help='for FAVOR+')
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')  # must divide d_model evenly
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')  # for the series decomposition module (in Autoformer)
    parser.add_argument('--factor', type=int, default=1, help='attn factor')  # In Autoformer/Informer, this controls the “sparsity factor” for attention
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--grad_clip', type=float, help='clip gradients', default=None)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--use_torch_compile', type=str2bool, default=True, help='compile model')
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu] if args.use_gpu else [0]

    args.use_torch_compile = args.use_torch_compile and hasattr(torch, 'compile')
    assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    print('args in experiment:')
    print(args)

    logging.info('args in experiment:')
    logging.info(args)
    logging.info(f"use_gpu={args.use_gpu} multi_gpu={args.use_multi_gpu} "
                f"gpu={args.gpu} device_ids={args.device_ids}")

    Exp = Exp_Main

    def write_json(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def time_block(fn, *a, **kw):
        # GPU event timing when available; fallback to wall clock
        if args.use_gpu:
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*a, **kw)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            secs = ms / 1000.0
        else:
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            secs = time.perf_counter() - t0
        minutes = int(secs // 60)
        rem = secs - 60 * minutes
        return out, {"seconds": secs, "readable": f"{minutes}m {rem:.3f}s"}
    
    if args.is_training:
        print(f'args.itr = {args.itr}')
        for ii in range(args.itr):
            # setting record of experiments
            setting = (
                f"{args.model_id}_{args.model}_{args.data}"
                f"_ft{args.features}"
                f"_sl{args.seq_len}"
                f"_ll{args.label_len}"
                f"_pl{args.pred_len}"
                f"_dm{args.d_model}"
                f"_nh{args.n_heads}"
                f"_el{args.e_layers}"
                f"_dl{args.d_layers}"
                f"_df{args.d_ff}"
                f"_fc{args.factor}"
                f"_eb{args.embed}"
                f"_dt{args.distil}"
                f"_sd{seed}"
                f"_{args.des}_{ii}"
            )

            results_path = os.path.join('../generated',  setting, 'results.jsonl')

            # save timestamp
            current_time = datetime.now().isoformat(timespec="seconds")
            write_json(results_path, {'timestamp': current_time})

            # save exact arguments for reproducibility
            args_dict = json.loads(json.dumps(vars(args), default=str))
            write_json(results_path, {'args': vars(args)})

            exp = Exp(args)  # set experiments
            
            logging.info(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            _, ttrain = time_block(exp.train, setting)

            logging.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            _, ttest = time_block(exp.test, setting)

            # save training and testing time
            write_json(results_path, {"time": {"train": ttrain, "test": ttest}})

            if args.do_predict:
                logging.info(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.predict(setting, True)

            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f"{args.model_id}_{args.model}_{args.data}"
            f"_ft{args.features}"
            f"_sl{args.seq_len}"
            f"_ll{args.label_len}"
            f"_pl{args.pred_len}"
            f"_dm{args.d_model}"
            f"_nh{args.n_heads}"
            f"_el{args.e_layers}"
            f"_dl{args.d_layers}"
            f"_df{args.d_ff}"
            f"_fc{args.factor}"
            f"_eb{args.embed}"
            f"_dt{args.distil}"
            f"_{args.des}_{ii}"
        )

        exp = Exp(args)  # set experiments
        logging.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
