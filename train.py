import torch
import torch.optim as optim
from tqdm import tqdm

from util import create_masks, LabelSmoothing, TransformerLRScheduler
from model import Transformer
from data_loder import create_dataloaders

def train_transformer(model, train_dataloader, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    """
    Training loop for transformer

    Args:
        model: Transformer model
        train_dataloader: DataLoader for training data
        criterion: Loss function (with label smoothing)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
    """
    model = model.to(device)
    model.train()
    
    all_losses = []
    
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            src_mask,tgt_mask = create_masks(src, tgt)
            
            tgt_input = tgt[:,:-1]
            tgt_output = tgt[:,1:].reshape(-1) # 展平 TODO
            
            optimizer.zero_grad()
            
            outputs = model(src, tgt_input)
            outputs = outputs.view(-1, outputs.size(-1))
            
            loss = criterion(outputs, tgt_output)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(step_num=epoch + 1) 
            
            epoch_loss += loss.item()
            
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        all_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')

    return all_losses
def main():          
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataloader, _, vocab_src, vocab_tgt = create_dataloaders(batch_size=32)

    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    )
    criterion = LabelSmoothing(smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)
    # Now you can use your training loop
    losses = train_transformer(
        model=model,
        train_dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device,        
    )
    
if __name__ == "__main__":
    main()
