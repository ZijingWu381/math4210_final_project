import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb


def train_model(model, train_loader, val_loader, **train_kwargs):
    """
    PyTorch function to train a Variational Autoencoder (VAE) model on some dataset.

    Args:
        train_args (dict): Dictionary containing training parameters.
        model (nn.Module): The VAE model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """
    # Unpack training arguments
    epochs = train_kwargs['epochs']
    learning_rate = train_kwargs['lr']
    device = train_kwargs['device']
    cache_dir = train_kwargs['cache_dir']
    use_wandb = train_kwargs['use_wandb']

    trial_id_offset_onvalset = len(train_loader)

    # send model to device
    model = model.to(device)

 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_val_loss = 1e10
    beta = 100.0

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()  # Set the model to training mode
        train_total_loss = {"train_loss": 0.0, "train_recon_loss": 0.0,
                            "train_kl_loss": 0.0, "train_constraint_value": 0.0, "train_log_jacob_g": 0.0}

        for data in train_loader:
            if len(data) == 2:
                x, u = data
                x = x.to(device)
                u = u.to(device)
                data = (x, u)
            elif len(data) == 4:
                x, u, trial_ids, time_stamps = data
                x = x.to(device)
                u = u.to(device)
                trial_id = trial_ids[0]
                data = (x, u, trial_id, time_stamps)

            optimizer.zero_grad()  # Zero the gradients
            constraint_loss = 0.0
            log_jacob_g = 0.0
            try:
                _, recon_loss, kl_loss = model(data=data)
            except:
                try:
                    _, recon_loss, kl_loss, constraint_loss = model(data=data)
                except:
                    _, recon_loss, kl_loss, constraint_loss, log_jacob_g = model(
                        data=data)

            # train_loss = kl_loss + recon_loss + log_jacob_g
            train_loss = beta * kl_loss + recon_loss + log_jacob_g

            train_total_loss["train_loss"] += train_loss
            train_total_loss["train_recon_loss"] += recon_loss + log_jacob_g
            # train_total_loss["train_kl_loss"] += kl_loss
            train_total_loss["train_kl_loss"] += beta * kl_loss
            train_total_loss["train_constraint_value"] += constraint_loss
            train_total_loss["train_log_jacob_g"] += log_jacob_g

            train_loss.backward()
            optimizer.step()

        train_total_loss = {k: v / len(train_loader)
                            for k, v in train_total_loss.items()}

        assert train_loader.batch_size != 1 or len(
            train_loader) == len(train_loader.dataset)

        print(f"Epoch {epoch}, Training Loss: {train_total_loss['train_loss']}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_total_loss = {"val_loss": 0.0, "val_recon_loss": 0.0,
                          "val_kl_loss": 0.0, "val_constraint_value": 0.0, "val_log_jacob_g": 0.0}
        with torch.no_grad():
            for data in val_loader:
                if len(data) == 2:
                    x, u = data
                    x = x.to(device)
                    u = u.to(device)
                    data = (x, u)
                elif len(data) == 4:
                    x, u, trial_ids, time_stamps = data
                    x = x.to(device)
                    u = u.to(device)
                    trial_id = trial_ids[0] + trial_id_offset_onvalset
                    data = (x, u, trial_id, time_stamps)

                constraint_loss = 0.0
                log_jacob_g = 0.0
                try:
                    _, recon_loss, kl_loss = model(data=data)
                except:
                    try:
                        _, recon_loss, kl_loss, constraint_loss = model(data=data)
                    except:
                        _, recon_loss, kl_loss, constraint_loss, log_jacob_g = model(
                            data=data)

                # val_loss = kl_loss + recon_loss + log_jacob_g
                val_loss = beta * kl_loss + recon_loss + log_jacob_g

                val_total_loss["val_loss"] += val_loss
                val_total_loss["val_recon_loss"] += recon_loss + log_jacob_g
                # val_total_loss["val_kl_loss"] += kl_loss
                val_total_loss["val_kl_loss"] += beta * kl_loss
                val_total_loss["val_constraint_value"] += constraint_loss
                val_total_loss["val_log_jacob_g"] += log_jacob_g

        val_total_loss = {k: v / len(val_loader) for k, v in val_total_loss.items()}
        if val_total_loss['val_loss'] < min_val_loss:
            min_val_loss = val_total_loss['val_loss']
            print(f"Saving model with validation loss {min_val_loss}")
            torch.save(model.state_dict(), cache_dir + "/best_model.pt")

        if use_wandb:
            wandb.log({"train_loss": train_total_loss["train_loss"],
                       "train_recon_loss": train_total_loss["train_recon_loss"],
                       "train_kl_loss": train_total_loss["train_kl_loss"],
                       "train_constraint_value": train_total_loss["train_constraint_value"],
                       "train_log_jacob_g": train_total_loss["train_log_jacob_g"],
                       "val_loss": val_total_loss["val_loss"],
                       "val_recon_loss": val_total_loss["val_recon_loss"],
                       "val_kl_loss": val_total_loss["val_kl_loss"],
                       "val_constraint_value": val_total_loss["val_constraint_value"],
                       "val_log_jacob_g": val_total_loss["val_log_jacob_g"]},
                      step=epoch)

        print(f"Epoch {epoch}, Validation Loss: {val_total_loss['val_loss']}")

    print("Training finished.")
