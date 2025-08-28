# coding=utf-8

# 在代码中屏蔽警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from B1_FR_ST_joint import run_experiment  # MTF

def main():
    # 定义不同网络对应的参数
    network_params = {
        'APNN': {'train_batch_size': 1, 'n_epochs': 2, 'patience': 30, 'lr': 0.001, 'log_freq': 1},
        # 'APNN': {'train_batch_size': 1, 'n_epochs': 500, 'patience': 30, 'lr': 0.001, 'log_freq': 5},
       
        # 'FusionNet': {'train_batch_size': 1, 'n_epochs': 3, 'patience': 30, 'lr': 0.00001, 'log_freq': 1},
        # 'FusionNet': {'train_batch_size': 1, 'n_epochs': 500, 'patience': 30, 'lr': 0.001, 'log_freq': 5},
        
        # 'PanNet': {'train_batch_size': 1, 'n_epochs': 3, 'patience': 30, 'lr': 0.001, 'log_freq': 1},        
        # 'PanNet': {'train_batch_size': 1, 'n_epochs': 500, 'patience': 30, 'lr': 0.001, 'log_freq': 5},

        # 'PNN': {'train_batch_size': 1, 'n_epochs': 3, 'patience': 30, 'lr': 0.001, 'log_freq': 1},
        # 'PNN': {'train_batch_size': 1, 'n_epochs': 500, 'patience': 30, 'lr': 0.001, 'log_freq': 5},

    }

    # 定义数据集名称
    # dataset_names = ['qb','gf2','wv2']
    # dataset_names = ['wv2']
    # dataset_names = ['qb']
    dataset_names = ['gf2']


    # 遍历每个网络和数据集
    for network_name, params in network_params.items():
        for dataset_name in dataset_names:
            run_experiment(
                network_name=network_name,
                dataset_name=dataset_name,
                train_batch_size=params['train_batch_size'],
                n_epochs=params['n_epochs'],
                patience=params['patience'],
                lr=params['lr'],
                log_freq=params['log_freq']
            )

if __name__ == "__main__":
    
    main()
