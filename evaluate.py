from utility import *

train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record = load_metrics()

plot_metrics(train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record)
