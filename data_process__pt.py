import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

"""Prepare temporal edge data into the feature and dataframe format used by the training pipeline."""


def preprocess(data_name):
  """Read a raw edge CSV file and convert it into a dataframe plus a feature matrix."""
  # Collect parsed values for each edge so they can be saved into a tabular format.
  src_list, dst_list, ts_list, label_list = [], [], [], []
  feat_l = []
  int_roll_list, ext_roll_list = [], []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      src = int(e[1])
      dst = int(e[2])

      ts = float(e[3])
      label = float(e[4])  # int(e[3])

      int_roll = int(e[-2])
      ext_roll = int(e[-1])

      feat = np.array([float(x) for x in e[5:-2]])

      src_list.append(src)
      dst_list.append(dst)
      ts_list.append(ts)
      label_list.append(label)
      int_roll_list.append(int_roll)
      ext_roll_list.append(ext_roll)

      feat_l.append(feat)
  return pd.DataFrame({'src': src_list,
                       'dst': dst_list,
                       'time': ts_list,
                       'label': label_list,
                       'int_roll': int_roll_list,
                       'ext_roll': ext_roll_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.src.max() - df.src.min() + 1 == len(df.src.unique()))
    assert (df.dst.max() - df.dst.min() + 1 == len(df.dst.unique()))

    upper_src = df.src.max() + 1
    new_dst = df.dst + upper_src

    new_df.dst = new_dst
    new_df.src += 1
    new_df.dst += 1
  else:
    new_df.src += 1
    new_df.dst += 1

  return new_df

def run(data_name, bipartite=True):
  # The original project used a Google Drive-backed data root; keep this path explicit for compatibility.
  path = '/content/drive/MyDrive/GraphML_Project/Code/GSNOP/code/DATA'
  PATH = '{}/IEEE/Old-IEEE/{}.csv'.format(path, data_name)
  OUT_DF = '{}/IEEE/Test/{}.csv'.format(path, data_name)
  OUT_FEAT = '{}/IEEE/Test/{}.npy'.format(path, data_name)
  OUT_NODE_FEAT = '{}/IEEE/Test/{}_node.npy'.format(path, data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.src.max(), new_df.dst.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)