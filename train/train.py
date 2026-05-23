import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data_utils import M4RawDataset
from utils.checkpoint import save_checkpoint, load_checkpoint

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.residual_unet import ResU_Net
from models.r2_unet import R2U_Net
from models.rau_net import Attention_ResU_Net
from models.unet_plus_plus import NestedUNet

DATA_DIR_TRAIN = '/path/to/M4Raw_multicoil_train/multicoil_train'  
DATA_DIR_VAL   = '/path/to/M4Raw_multicoil_val/multicoil_val'      

ACCELERATION = 4          
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_NAME = "UNet"       
CHECKPOINT_PATH = f'checkpoint_{MODEL_NAME}_R{ACCELERATION}.pth'


train_dataset = M4RawDataset(DATA_DIR_TRAIN, acceleration=ACCELERATION)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)

val_dataset = M4RawDataset(DATA_DIR_VAL, acceleration=ACCELERATION)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)


if MODEL_NAME == "UNet":
    model = UNet(in_chans=1, out_chans=1, chans=64).cuda()
elif MODEL_NAME == "AttentionUNet":
    model = AttentionUNet(in_channel=1, out_channel=1).cuda()
elif MODEL_NAME == "ResUNet":
    model = ResU_Net(img_ch=1, output_ch=1).cuda()
elif MODEL_NAME == "R2UNet":
    model = R2U_Net(img_ch=1, output_ch=1).cuda()
elif MODEL_NAME == "RAUNet":
    model = Attention_ResU_Net(img_ch=1, output_ch=1).cuda()
elif MODEL_NAME == "UNetPlusPlus":
    model = NestedUNet(num_classes=1, input_channels=1, deep_supervision=False).cuda()
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch, _, _, train_losses, val_losses = load_checkpoint(model, optimizer, CHECKPOINT_PATH)


for epoch in tqdm(range(start_epoch, NUM_EPOCHS), desc=f"Training {MODEL_NAME}"):
  
    model.train()
    train_loss = 0.0
    for zero_filled, target, _ in train_dataloader:
        zero_filled = zero_filled.float().unsqueeze(1).cuda()
        target = target.float().unsqueeze(1).cuda()

        optimizer.zero_grad()
        output = model(zero_filled)
        
        # Handle U-Net++ deep supervision
        if hasattr(model, 'deep_supervision') and model.deep_supervision:
            loss = sum(criterion(out, target) for out in output) / len(output)
            output = output[-1]  
        else:
            loss = criterion(output, target)
            
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

  
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for zero_filled, target, _ in val_dataloader:
            zero_filled = zero_filled.float().unsqueeze(1).cuda()
            target = target.float().unsqueeze(1).cuda()
            output = model(zero_filled)
            
            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                loss = sum(criterion(out, target) for out in output) / len(output)
            else:
                loss = criterion(output, target)
                
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss, 
                   train_losses, val_losses, CHECKPOINT_PATH)


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Curve - {MODEL_NAME} (R={ACCELERATION})')
plt.legend()
plt.grid(True)
plt.savefig(f'training_curve_{MODEL_NAME}_R{ACCELERATION}.png')
plt.show()

print(f"Training completed for {MODEL_NAME} at acceleration R={ACCELERATION}")
