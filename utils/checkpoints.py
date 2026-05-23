import torch
import os

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_losses, val_losses, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizer, filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Loaded checkpoint from {filepath} (epoch {epoch})")
        return epoch + 1, train_loss, val_loss, train_losses, val_losses
    else:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return 0, 0.0, 0.0, [], []
