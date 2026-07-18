"""
eval_with_mean_z.py
====================
Script eval ĐỘC LẬP -- KHÔNG sửa gì trong neural_process.py/train_np.py.

Load model checkpoint đã train xong, PATCH _sample_latent tại runtime
(monkey-patch) để dùng mean tất định thay vì sample ngẫu nhiên, rồi eval
trên test set. So kết quả với con số cũ để xác nhận nhiễu sampling có
phải nguyên nhân khiến NP thua origin hay không.

Cách chạy: đặt file này CÙNG THƯ MỤC với train_np.py, modules.py, utils.py,
neural_process.py, memorys.py, layers.py, sampler.py, rồi chạy với đúng
các tham số bạn dùng lúc train, cộng thêm --model_path trỏ tới file .pkl
đã lưu:

    python eval_with_mean_z.py --path <path> --data <data> --config <config> \\
        --gpu 0 --base_model snp --ode --latent_dist gmm \\
        --model_path /path/to/model_xxx.pkl \\
        --eval_neg_samples 50

(Các flag khác giữ giống hệt lúc bạn chạy train_np.py để build đúng model.)
"""

import argparse
import os
import types

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='custom path')
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_path', type=str, required=True, help='path to saved .pkl checkpoint')
parser.add_argument('--base_model', type=str, choices=['origin', 'np', 'snp', 'anp', 'mnp'], default='origin')
parser.add_argument('--ode', action='store_true')
parser.add_argument('-d', '--determinstic', action='store_true')
parser.add_argument('--resize_ratio', default=0, type=int)
parser.add_argument('--rand_edge_features', type=int, default=0)
parser.add_argument('--rand_node_features', type=int, default=0)
parser.add_argument('--eval_neg_samples', type=int, default=1)
parser.add_argument('--zero_feat', action='store_true')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--latent_dist', type=str, choices=['gaussian', 'gmm'], default='gmm')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
import random
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from neural_process import NeuralProcess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

# ---- Load graph / feature / config, y hệt train_np.py ----
g, df = load_graph(args.path, args.data)
node_feats, edge_feats = load_feat(args.path, args.data, g['indptr'].shape[0] - 1, len(df),
                                   args.rand_edge_features, args.rand_node_features, args.zero_feat)
sample_param, memory_param, gnn_param, np_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True

if args.resize_ratio == 0:
    resize_ratio = 10 ** (len(str(int(df['time'].max()))) - 1)
else:
    resize_ratio = args.resize_ratio

model = NeuralProcess(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, np_param,
                      args.base_model, args.ode, args.determinstic, resize_ratio, args.latent_dist,
                      combined=combine_first).cuda()

mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy'] == 'recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

val_losses = list()


# ---- eval() y hệt bản trong train_np.py ----
def eval(mode='val', stage='val'):
    neg_samples = 1
    model.eval()
    y_pred_list, y_label_list, ts_list = list(), list(), list()
    ap_list, auc_list, aucs_mrrs, mrr_ts = list(), list(), list(), list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
        neg_samples = args.eval_neg_samples
        batch_size = train_param['test_batch_size']
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
        batch_size = train_param['test_batch_size']
    elif mode == 'train':
        eval_df = df[:train_edge_end]
        batch_size = train_param['batch_size']
    with torch.no_grad():
        total_loss = 0
        creterion = torch.nn.BCEWithLogitsLoss()
        for _, rows in eval_df.groupby(eval_df.index // batch_size):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values,
                                         neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, ts, neg_samples=neg_samples, data=mode)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).view(-1).sigmoid().cpu().numpy()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0).view(-1).numpy()
            y_pred_list.append(y_pred)
            y_label_list.append(y_true)
            ap_list.append(average_precision_score(y_true, y_pred))
            auc_list.append(roc_auc_score(y_true, y_pred))
            ts_list.append(np.tile(rows.time.values, neg_samples + 1).astype(np.float32))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(
                    pred_pos.squeeze().unsqueeze(0) < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1
                ).type(torch.float))
                mrr_ts.append(rows.time.values)
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid,
                                       model.memory_updater.last_updated_memory, root_nodes, ts,
                                       mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid,
                                      model.memory_updater.last_updated_memory, root_nodes,
                                      model.memory_updater.last_updated_ts, neg_samples=neg_samples)
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_label_list)
    all_ts = np.concatenate(ts_list)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    auc_mrr = -1
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    if mode == "test":
        raw_results = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 't': all_ts})
        raw_mrr_results = None
        if neg_samples > 1:
            raw_mrr = torch.cat(aucs_mrrs).cpu().numpy()
            raw_mrr_results = pd.DataFrame({'mrr': raw_mrr, 't': np.concatenate(mrr_ts)})
        return ap, auc, auc_mrr, [raw_results, raw_mrr_results]
    return ap, auc, auc_mrr


# ---- Load checkpoint ----
print(f"Loading checkpoint: {args.model_path}")
model.load_state_dict(torch.load(args.model_path))


# ==========================================================
# PATCH TẠI RUNTIME -- không đụng vào file neural_process.py
# Thay _sample_latent bằng bản trả về MEAN tất định. Vì script
# này chỉ eval (không train), an toàn để thay vô điều kiện.
# ==========================================================
def _mean_latent(self, mu, sigma, pi_logits=None, sample_shape=None):
    if self.latent_dist == 'gmm':
        pi = F.softmax(pi_logits, dim=-1)                 # (batch, K)
        z_mean = (pi.unsqueeze(-1) * mu).sum(dim=1)        # (batch, z_dim)
    else:
        z_mean = mu
    return z_mean.unsqueeze(0).expand(self.l, *z_mean.shape)


model._sample_latent = types.MethodType(_mean_latent, model)
print("Đã patch _sample_latent -> dùng mean thay vì sample ngẫu nhiên (chỉ áp dụng trong script này).")


# ---- Eval giống hệt khối cuối của train_np.py ----
model.eval()
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
model.reset()
model.test = True

print("Đang chạy eval trên train/val để cập nhật running_r/mailbox trước test...")
if args.base_model != 'origin' or mailbox is not None:
    eval('train', stage='test')
    eval('val', stage='test')

print("Đang eval trên test set (dùng mean z)...")
all_ap, all_auc, all_mrr, raw_results = eval('test')

print(f"\n=== KẾT QUẢ VỚI z = MEAN (không sample) ===")
print(f"AP:  {all_ap:.6f}")
print(f"AUC: {all_auc:.6f}")
print(f"MRR: {all_mrr:.6f}")
print(f"\n(So sánh với kết quả cũ dùng sample ngẫu nhiên: AP=0.7665 AUC=0.9957 MRR=0.9255)")