#%%c
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gatmodel import BRepGATClassifier
from dataset_creation.dataloader import get_dataloader


import warnings
#%%
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed_everything(3407, workers=True)  # 3407

    # 数据准备
    train_path = "../data/dataset/train.pt"
    val_path = "../data/dataset/val.pt"
    test_path = "../data/dataset/test.pt"

    batch_size = 1
    train_loader = get_dataloader(train_path, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = get_dataloader(val_path, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = get_dataloader(test_path, batch_size=batch_size, shuffle=False, num_workers=1)
    print("=======================Prepare Data Finish!=======================")

    # 模型准备
    model = BRepGATClassifier(
    in_node_features=14,
    in_edge_features=15,
    hidden_dim=640,
    num_classes=27,
    dropout=0.3
)
    print(type(model))
    print(isinstance(model, L.LightningModule))
#%%
    print("=======================Model Summary=======================")
    summary = ModelSummary(model, max_depth=-1)
    print(summary)

    print("=======================Begin Train=======================")
    # 训练
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename='{epoch}-{cls_val_accuracy:.4f}-{seg_val_ap:.4f}-{rel_val_ap:.4f}'
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        check_on_train_epoch_end=False
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=20, annealing_epochs=10)

    logger = TensorBoardLogger("tb_logs", name="gat_model")
    trainer = L.Trainer(
        logger=logger,
        deterministic=False,
        max_epochs=2,
        default_root_dir="../checkpoints3/FRModel-final/",
        # profiler="simple",
        log_every_n_steps=1,
        precision="16-mixed",
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            swa_callback
        ]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#%%
    print("=======================Begin Test=======================")
    # 测试
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
