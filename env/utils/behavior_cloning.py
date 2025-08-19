import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

def behavior_cloning_pretrain(model, episodes, epochs=50, batch_size=128):
    obs_list, act_list, mask_list = [], [], []

    # Prepare the dataset from the episodes
    for reward, episode in episodes:
        for obs, act, mask in episode:
            obs_list.append(obs)
            act_list.append(act)
            mask_list.append(mask)

    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
    act_tensor = torch.tensor(act_list, dtype=torch.long)
    mask_tensor = torch.tensor(np.array(mask_list), dtype=torch.bool)

    dataset = TensorDataset(obs_tensor, act_tensor, mask_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.policy.train()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_losses = []  # To store loss for each epoch

    for epoch in trange(epochs, desc="📚 Pretraining", unit="epoch"):
        epoch_loss = 0
        num_batches = 0

        for batch_obs, batch_act, batch_mask in loader:
            dist = model.policy.get_distribution(batch_obs.to(model.device))
            logits = dist.distribution.logits
            

            loss = loss_fn(logits, batch_act.to(model.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

    # Plotting the loss
    plt.plot(range(epochs), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Behavior Cloning Pretraining Loss")
    plt.show()

    return model

