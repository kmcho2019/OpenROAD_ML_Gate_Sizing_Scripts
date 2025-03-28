import torch
import torch.nn as nn
import math
import os
import numpy as np

from torch.utils.data import TensorDataset, Dataset, DataLoader
from model import TwoEncoderTransformer, export_model_weights_robust, test_attention_masking, test_two_encoder_transformer_masking

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import torch.nn.functional as F

def load_embeddings(filename):
    with open(filename, 'rb') as f:
        # Read N and D
        N = int.from_bytes(f.read(8), byteorder='little')
        D = int.from_bytes(f.read(8), byteorder='little')

        arr = np.frombuffer(f.read(), dtype=np.float32).reshape(N, D)

        # Read the embeddings tensor
        #embeddings = torch.from_numpy(np.frombuffer(f.read(), dtype=np.float32).reshape(N, D))
        return torch.from_numpy(arr.copy())

def load_3d_tensor(filename):
    """Load 3D float tensor from binary file (shape [N, L, D])"""
    with open(filename, 'rb') as f:
        N = int.from_bytes(f.read(8), 'little')
        L = int.from_bytes(f.read(8), 'little')
        D = int.from_bytes(f.read(8), 'little')
        arr = np.frombuffer(f.read(), dtype=np.float32).reshape(N, L, D)
        return torch.from_numpy(arr.copy())

def load_2d_int_tensor(filename):
    """Load 2D int tensor from binary file (shape [R, C])"""
    with open(filename, 'rb') as f:
        R = int.from_bytes(f.read(8), 'little')
        C = int.from_bytes(f.read(8), 'little')
        arr = np.frombuffer(f.read(), dtype=np.int32).reshape(R, C)
        return torch.from_numpy(arr.copy())


# -----------------------------------------------------------------------------
# Custom Dataset for TransSizer
# -----------------------------------------------------------------------------

class TransSizerDataset(Dataset):
    """
    A PyTorch Dataset that loads the five binary files from a given design
    directory. Each design directory is expected to contain:
      - data_array.bin
      - encoder_2_input_libcell_type_ids.bin
      - ASAP7_libcell_type_embeddings.bin
      - encoder_2_output_avail_libcell_num.bin
      - labels.bin
    """
    def __init__(self, design_dirs):
        """
        Args:
            design_dirs (list of str): List of directories, one per design.
        """
        self.design_dirs = design_dirs

    def __len__(self):
        return len(self.design_dirs)

    def __getitem__(self, idx):
        design_dir = self.design_dirs[idx]

        # Construct full paths for each file
        data_array_path = os.path.join(design_dir, "data_array.bin")
        type_ids_path = os.path.join(design_dir, "encoder_2_input_libcell_type_ids.bin")
        embeddings_path = os.path.join(design_dir, "ASAP7_libcell_type_embeddings.bin")
        avail_libcell_num_path = os.path.join(design_dir, "encoder_2_output_avail_libcell_num.bin")
        labels_path = os.path.join(design_dir, "labels.bin")

        # Load each file
        data1 = load_3d_tensor(data_array_path)
        type_ids = load_2d_int_tensor(type_ids_path)
        type_embeddings = load_embeddings(embeddings_path)
        encoder2_avail = load_2d_int_tensor(avail_libcell_num_path)
        labels = load_2d_int_tensor(labels_path).long()

        # Process type_ids and generate data2:
        #   - Replace invalid type_ids (-1) with 0 for indexing
        #   - Then zero-out embeddings where type_ids were originally -1
        mask = (type_ids == -1)
        safe_ids = type_ids.clone()
        safe_ids[mask] = 0
        data2 = type_embeddings[safe_ids]
        data2[mask.unsqueeze(-1).expand_as(data2)] = 0

        # Convert encoder2_avail to long (if needed)
        safe_encoder2_avail = encoder2_avail.clone().long()

        # Return a dictionary of tensors.
        # The training code can later use the keys "data1", "data2", "labels",
        # "encoder2_avail", and "type_ids" as needed.
        sample = {
            "data1": data1,
            "data2": data2,
            "labels": labels,
            "encoder2_avail": safe_encoder2_avail,
            "type_ids": type_ids,
        }
        return sample

# -----------------------------------------------------------------------------
# Helper functions to create DataLoaders for training and validation.
# -----------------------------------------------------------------------------

def create_dataloader(design_dirs, batch_size=1, shuffle=True, num_workers=0):
    """
    Creates a DataLoader from a list of design directories.
    """
    dataset = TransSizerDataset(design_dirs)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

def create_train_val_dataloaders(train_dirs, val_dirs,
                                 batch_size=1, shuffle=True, num_workers=0):
    """
    Given a base directory for training designs and another for validation designs,
    this function collects the individual design directories and creates DataLoaders.

    Args:
        train_dirs (list): List containing training design folders.
        val_dirs (list): List containing validation design folders.
        batch_size (int): Batch size (each sample is a design).
        shuffle (bool): Whether to shuffle the training designs.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        train_loader, val_loader: Two DataLoaders for training and validation.
    """
    # List all subdirectories (designs) in the provided base paths
    #train_design_dirs = [os.path.join(train_base_dir, d) for d in os.listdir(train_base_dir) if os.path.isdir(os.path.join(train_base_dir, d))]
    #val_design_dirs = [os.path.join(val_base_dir, d) for d in os.listdir(val_base_dir) if os.path.isdir(os.path.join(val_base_dir, d))]

    train_loader = create_dataloader(train_dirs,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers)
    val_loader = create_dataloader(val_dirs,
                                   batch_size=batch_size,
                                   shuffle=False,  # Typically no need to shuffle validation data
                                   num_workers=num_workers)
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# Training function with validation evaluation
# -----------------------------------------------------------------------------

def train_model(train_loader, val_loader, num_epochs=5, warmup_ratio=0.1, device="cpu"):
    # -------------------------------------------------------------------------
    # 1. Model Configuration (using your hyper-parameters)
    # -------------------------------------------------------------------------
    D_in = 786   # e.g., 768 + 18
    D_out = 50   # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2
    )
    model.to(device)
    model.train()

    # -------------------------------------------------------------------------
    # 2. Loss & Optimizer
    # -------------------------------------------------------------------------
    criterion = CustomCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Total training steps (one step per batch)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
    for epoch in range(num_epochs):
        # -----------------------------
        # Training Loop
        # -----------------------------
        for batch in train_loader:
            data1 = batch["data1"].to(device)
            data2 = batch["data2"].to(device)
            labels = batch["labels"].to(device)
            encoder2_avail = batch["encoder2_avail"].to(device)
            type_ids = batch["type_ids"].to(device)

            optimizer.zero_grad()
            out = model(data1, data2)  # expected shape: [bsz, L2, D_out]
            loss = criterion(out, labels, encoder2_avail)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            preds = out.argmax(dim=2)  # [bsz, L2]
            correct = (preds == labels).sum().item()
            total = labels.numel()
            accuracy = correct / total

            # Accuracy excluding padding tokens (where type_ids == -1)
            mask = (type_ids != -1)
            valid_count = mask.sum().item()
            valid_correct = (preds[mask] == labels[mask]).sum().item()
            valid_accuracy = valid_correct / valid_count if valid_count > 0 else 0.0

            current_lr = scheduler.get_last_lr()[0]  # Assumes a single param group
            print(f"[Train] Epoch [{epoch+1}/{num_epochs}] Step {global_step} "
                  f"LR={current_lr:.6f} Loss={loss.item():.4f} "
                  f"Acc={accuracy*100:.2f}% Valid Acc={valid_accuracy*100:.2f}%")

        # -----------------------------
        # Validation Evaluation Step
        # -----------------------------
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        val_valid_correct = 0
        val_valid_count = 0

        with torch.no_grad():
            for batch in val_loader:
                data1 = batch["data1"].to(device)
                data2 = batch["data2"].to(device)
                labels = batch["labels"].to(device)
                encoder2_avail = batch["encoder2_avail"].to(device)
                type_ids = batch["type_ids"].to(device)

                out = model(data1, data2)
                loss = criterion(out, labels, encoder2_avail)
                # Multiply by the number of samples in this batch for weighted average.
                batch_size = data1.size(0)
                val_loss_total += loss.item() * batch_size

                preds = out.argmax(dim=2)
                batch_correct = (preds == labels).sum().item()
                batch_total = labels.numel()
                val_correct += batch_correct
                val_total += batch_total

                mask = (type_ids != -1)
                batch_valid_correct = (preds[mask] == labels[mask]).sum().item()
                batch_valid_count = mask.sum().item()
                val_valid_correct += batch_valid_correct
                val_valid_count += batch_valid_count

        # Compute average validation loss and accuracy
        avg_val_loss = val_loss_total / len(val_loader.dataset)
        overall_val_accuracy = val_correct / val_total if val_total > 0 else 0
        valid_val_accuracy = val_valid_correct / val_valid_count if val_valid_count > 0 else 0

        print(f"[Validation] Epoch [{epoch+1}/{num_epochs}] "
              f"Avg Loss: {avg_val_loss:.4f} "
              f"Overall Acc: {overall_val_accuracy*100:.2f}% "
              f"Valid Acc (excl. padding): {valid_val_accuracy*100:.2f}%")

        # Set model back to training mode for next epoch
        model.train()

    return model


def example_train():
    # Hyperparams (example)
    D_in = 786 # 768 + 18  #17
    D_out = 50 # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    # Create model
    model = TwoEncoderTransformer(D_in, D_out, D_emb, D_model, FF_hidden_dim, num_heads,
                                  num_encoder_layers, num_encoder_layers_2)
    model.train()

    # Dummy data: (batch_size=4, L=8, D_in=17)
    # and second input (batch_size=4, L2=4, D_emb=768)
    # L2 = L//2 in your scheme
    bsz = 4
    L = 8
    L2 = L // 2

    # data1 = torch.randn(bsz, L, D_in)    # [4, 8, 786]
    # data2 = torch.randn(bsz, L2, D_emb) # [4, 4, 768]
    # Load data1: 3D tensor [bsz, L, D_in]
    data1 = load_3d_tensor("data_array.bin")
    bsz, L, D_in = data1.shape
    L2 = L // 2

    # Load data2: Process 2D IDs to 3D embeddings
    type_ids = load_2d_int_tensor("encoder_2_input_libcell_type_ids.bin")
    type_embeddings = load_embeddings("ASAP7_libcell_type_embeddings.bin")

    # Convert IDs to embeddings with -1 handling
    mask = (type_ids == -1)
    safe_ids = type_ids.clone()
    safe_ids[mask] = 0  # Replace -1 with 0 index
    data2 = type_embeddings[safe_ids]  # Shape [bsz, L2, D_emb]
    data2[mask.unsqueeze(-1).expand_as(data2)] = 0  # Zero out -1 embeddings

    # Print shapes of data1 and data2
    print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")

    # Fake “labels” just for demonstration—assume output shape is [4,4,50]
    # we do an MSE to some random target
    # target = torch.randn(bsz, L2, D_out)  # [4, 4, 50]

    # Load target: 2D tensor [bsz, L2] target has integer class labels
    target = load_2d_int_tensor("labels.bin")
    target = target.long()  # Convert to long tensor

    # Print target shape
    print(f"Target shape: {target.shape}")


    # Simple MSE loss
    #criterion = nn.MSELoss()

    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test model output shape
    out = model(data1, data2)
    print(f"Model output shape: {out.shape}")

    # One pass
    for step in range(10):
        optimizer.zero_grad()
        out = model(data1, data2)

        # Transpose the output to be compatible with CrossEntropyLoss
        out = out.transpose(1, 2)  # Change shape from [bsz, L2, num_classes] to [bsz, num_classes, L2]

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

    return model

def example_train_2(num_epochs=5, warmup_ratio=0.1):
    """
    Trains the TwoEncoderTransformer model for 'num_epochs' epochs.

    Args:
        num_epochs (int): Number of epochs to train for.
        warmup_ratio (float): Fraction of total steps used for linear warmup.

    Returns:
        model: Trained model.
    """
    # -------------------------------------------------------------------------
    # 1. Model Configuration
    # -------------------------------------------------------------------------
    D_in = 786   # 768 + 18
    D_out = 50   # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    # Create model
    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2
    )
    model.train()

    # -------------------------------------------------------------------------
    # 2. Data Loading (Single Batch Example)
    #    Replace this section with a DataLoader if you have multiple batches.
    # -------------------------------------------------------------------------
    # Load data1: 3D tensor [bsz, L, D_in]
    data1 = load_3d_tensor("data_array.bin")       # e.g. shape [4, 8, 786]
    bsz, L, D_in = data1.shape
    L2 = L // 2

    # Load data2 from type IDs -> embeddings
    type_ids = load_2d_int_tensor("encoder_2_input_libcell_type_ids.bin")
    type_embeddings = load_embeddings("ASAP7_libcell_type_embeddings.bin")
    mask = (type_ids == -1)
    safe_ids = type_ids.clone()
    safe_ids[mask] = 0  # replace -1 with 0 index
    data2 = type_embeddings[safe_ids]  # shape [bsz, L2, D_emb]
    data2[mask.unsqueeze(-1).expand_as(data2)] = 0  # zero out invalid embeddings

    # Load target: 2D tensor [bsz, L2] of integer class labels
    target = load_2d_int_tensor("labels.bin").long()

    # Print shapes of data1, data2, and target
    print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}, Target shape: {target.shape}")

    # Print the share of padding tokens in data2
    print(f"Share of padding tokens in data2: {mask.float().mean().item():.2f}")
    # Average number of padding tokens
    print(f"Average number of padding tokens per batch: {mask.sum(dim=1).float().mean().item():.2f}")

    # -------------------------------------------------------------------------
    # 3. Loss Function & Optimizer
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # -------------------------------------------------------------------------
    # 4. Learning Rate Scheduler: Linear Warmup + Linear Decay
    # -------------------------------------------------------------------------
    # We'll assume a total of (num_epochs * 1) "steps" if we only do
    # one step per epoch. Adjust if you have multiple steps per epoch.
    total_steps = num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        """
        1) Warmup from LR=0 -> LR=1 (relative scaling) for 'warmup_steps'.
        2) Then linearly decay from 1 -> 0 from 'warmup_steps' to 'total_steps'.
        """
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------------------------------------------------------
    # 5. Training Loop by Epoch
    # -------------------------------------------------------------------------
    global_step = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(data1, data2)  # shape [bsz, L2, D_out]

        # For CrossEntropyLoss, we need [bsz, num_classes, L2]
        out = out.transpose(1, 2)  # => [bsz, D_out, L2]

        # Compute loss
        loss = criterion(out, target)  # target shape [bsz, L2]

        # Backprop
        loss.backward()
        optimizer.step()

        # Scheduler step (update learning rate)
        scheduler.step()
        global_step += 1

        # ---------------------------------------------------------------------
        # Compute Accuracy
        # out.shape = [bsz, D_out, L2]
        # argmax over dimension=1 (class dimension) => shape [bsz, L2]
        # Compare with target [bsz, L2]
        # ---------------------------------------------------------------------
        preds = out.argmax(dim=1)  # [bsz, L2]
        correct = (preds == target).sum().item()
        total = target.numel()
        accuracy = correct / total

        # Use mask to exclude padding tokens when computing accuracy
        mask = (type_ids != -1)
        valid_count = mask.sum().item()
        valid_correct = (preds[mask] == target[mask]).sum().item()
        valid_accuracy = valid_correct / valid_count if valid_count > 0 else 0.0

        # ---------------------------------------------------------------------
        # Print epoch metrics
        # ---------------------------------------------------------------------
        current_lr = scheduler.get_last_lr()[0]  # If you have only one param group
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Step {global_step} "
            f"LR={current_lr:.6f} "
            f"Loss={loss.item():.4f} "
            f"Acc={accuracy*100:.2f}% "
            f"Valid Acc={valid_accuracy*100:.2f}%"  # Accuracy excluding padding tokens
        )

    return model

def example_train_3(
    batch_size=4,
    num_epochs=5,
    warmup_ratio=0.1,
    lr=1e-3
):
    """
    Train the TwoEncoderTransformer model, ignoring padding tokens (-1)
    for both loss and accuracy.
    """
    # -------------------------------------------------------------------------
    # 1. Model Configuration
    # -------------------------------------------------------------------------
    D_in = 786  # 768 + 18
    D_out = 50  # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2
    )
    model.train()

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # Load data1: shape [N, L, D_in]
    data1 = load_3d_tensor("data_array.bin")
    N, L, D_in_check = data1.shape
    assert D_in_check == D_in, f"Expected D_in={D_in}, got {D_in_check}"

    # L2 = L // 2
    L2 = L // 2

    # Load data2 from type IDs -> embeddings
    type_ids = load_2d_int_tensor("encoder_2_input_libcell_type_ids.bin")  # shape [N, L2]
    type_embeddings = load_embeddings("ASAP7_libcell_type_embeddings.bin") # shape [num_types, D_emb]
    mask = (type_ids == -1)
    safe_ids = type_ids.clone()
    safe_ids[mask] = 0  # replace -1 with 0 index
    data2 = type_embeddings[safe_ids]  # shape [N, L2, D_emb]
    data2[mask.unsqueeze(-1).expand_as(data2)] = 0  # zero out invalid embeddings

    # Load target: shape [N, L2], some may be -1 to indicate padding
    target = load_2d_int_tensor("labels.bin").long()

    #print(f"data1 shape: {data1.shape}")
    #print(f"data2 shape: {data2.shape}")
    #print(f"target shape: {target.shape}")

    # Print shapes of data1, data2, and target
    print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}, Target shape: {target.shape}")

    # Print the share of padding tokens in data2
    print(f"Share of padding tokens in data2: {mask.float().mean().item():.2f}")
    # Average number of padding tokens
    print(f"Average number of padding tokens per batch: {mask.sum(dim=1).float().mean().item():.2f}")


    # Create a Dataset & DataLoader
    train_dataset = TensorDataset(data1, data2, target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------------------------------------------
    # 3. Loss Function & Optimizer
    #    Use ignore_index=-1 to exclude padded labels from the loss.
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------------------------------------------------
    # 4. Learning Rate Scheduler (Linear Warmup + Decay)
    # -------------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_count = 0
        total_count = 0

        for batch_data1, batch_data2, batch_target in train_loader:
            # Forward
            out = model(batch_data1, batch_data2)  # shape [bsz, L2, D_out]
            out = out.transpose(1, 2)  # => [bsz, D_out, L2]

            # Compute loss (ignore_index=-1 in criterion)
            loss = criterion(out, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Accumulate total loss
            bsz = batch_data1.size(0)
            epoch_loss += loss.item() * bsz  # multiply by batch size for average

            # Compute accuracy, ignoring -1
            # ---------------------------------------------------
            #   preds shape: [bsz, L2]
            #   batch_target shape: [bsz, L2]
            #   valid_mask excludes positions where batch_target == -1
            # ---------------------------------------------------
            preds = out.argmax(dim=1)
            valid_mask = (batch_target != -1)

            # Flatten everything for counting if you prefer
            valid_preds = preds[valid_mask]          # Only valid positions
            valid_labels = batch_target[valid_mask]  # Exclude padding

            correct_count += (valid_preds == valid_labels).sum().item()
            total_count += valid_labels.numel()

        # Average loss & accuracy for the epoch
        epoch_loss /= len(train_dataset)
        epoch_acc = correct_count / total_count if total_count > 0 else 0.0

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Step {global_step} "
            f"LR={current_lr:.6f} "
            f"Loss={epoch_loss:.4f} "
            f"Acc={100*epoch_acc:.2f}%"
        )

    return model

def example_train_4(
    batch_size=4,
    num_epochs=5,
    warmup_ratio=0.1,
    lr=1e-3,
    device=None
):
    """
    Train the TwoEncoderTransformer model with the following features:
    - Configurable batch size
    - Epoch-based training
    - Linear warmup + stable decay learning rate scheduling
    - Loss and accuracy logging each epoch
    - Exclude padding tokens (-1 indexes) from loss and accuracy
    - Configurable device (CPU/GPU)

    Args:
        batch_size (int): Batch size for DataLoader.
        num_epochs (int): Number of epochs to train for.
        warmup_ratio (float): Fraction of total steps used for linear warmup.
        lr (float): Initial learning rate.
        device (str or torch.device): Device to train on ('cpu', 'cuda', etc.).
                                        If None, automatically selects 'cuda' if available.
    Returns:
        model: Trained model.
    """
    # -------------------------------------------------------------------------
    # 1. Device Configuration
    # -------------------------------------------------------------------------
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Model Configuration
    # -------------------------------------------------------------------------
    D_in = 786   # 768 + 18
    D_out = 50   # Maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    # Instantiate the model and move it to the specified device
    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2
    ).to(device)
    model.train()

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    # Load data1: shape [N, L, D_in]
    data1 = load_3d_tensor("data_array.bin")  # Example: [N, L, D_in]
    N, L, D_in_check = data1.shape
    assert D_in_check == D_in, f"Expected D_in={D_in}, got {D_in_check}"

    # Calculate L2 based on L
    L2 = L // 2

    # Load data2 from type IDs -> embeddings
    type_ids = load_2d_int_tensor("encoder_2_input_libcell_type_ids.bin")  # Shape: [N, L2]
    type_embeddings = load_embeddings("ASAP7_libcell_type_embeddings.bin")  # Shape: [num_types, D_emb]

    # Handle padding tokens (-1)
    mask = (type_ids == -1)  # Shape: [N, L2]
    safe_ids = type_ids.clone()
    safe_ids[mask] = 0  # Replace -1 with 0 index to avoid indexing errors
    data2 = type_embeddings[safe_ids]  # Shape: [N, L2, D_emb]
    data2[mask.unsqueeze(-1).expand_as(data2)] = 0  # Zero out embeddings where type_ids == -1

    # Load target labels: shape [N, L2], where -1 indicates padding
    target = load_2d_int_tensor("labels.bin").long()  # Convert to long tensor for CrossEntropyLoss

    # Print data shapes and padding statistics
    print(f"Data1 shape: {data1.shape}")
    print(f"Data2 shape: {data2.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Share of padding tokens in target: {mask.float().mean().item():.2f}")
    print(f"Average number of padding tokens per sample: {mask.sum(dim=1).float().mean().item():.2f}")

    # Move data to device
    data1 = data1.to(device)
    data2 = data2.to(device)
    target = target.to(device)

    # Create a Dataset & DataLoader
    train_dataset = TensorDataset(data1, data2, target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------------------------------------------
    # 4. Loss Function & Optimizer
    # -------------------------------------------------------------------------
    # Use ignore_index=-1 to exclude padded labels from the loss
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------------------------------------------------
    # 5. Learning Rate Scheduler (Linear Warmup + Decay)
    # -------------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        """
        Linear warmup followed by linear decay.
        """
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_count = 0
        total_count = 0

        for batch_data1, batch_data2, batch_target in train_loader:
            # Move batch data to device
            batch_data1 = batch_data1.to(device)
            batch_data2 = batch_data2.to(device)
            batch_target = batch_target.to(device)

            # Forward pass
            out = model(batch_data1, batch_data2)  # Shape: [bsz, L2, D_out]
            out = out.transpose(1, 2)  # Shape: [bsz, D_out, L2] for CrossEntropyLoss

            # Compute loss (ignore_index=-1)
            loss = criterion(out, batch_target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Accumulate loss (multiply by batch size to get total loss)
            epoch_loss += loss.item() * batch_data1.size(0)

            # Compute accuracy, ignoring padding tokens
            preds = out.argmax(dim=1)  # Shape: [bsz, L2]
            valid_mask = (batch_target != -1)  # Shape: [bsz, L2]

            # Apply mask to predictions and targets
            valid_preds = preds[valid_mask]
            valid_labels = batch_target[valid_mask]

            correct_count += (valid_preds == valid_labels).sum().item()
            total_count += valid_labels.numel()

        # Calculate average loss and accuracy for the epoch
        epoch_loss /= len(train_dataset)
        epoch_acc = correct_count / total_count if total_count > 0 else 0.0

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]  # Assumes a single parameter group

        # Log epoch metrics
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Step {global_step} "
            f"LR={current_lr:.6f} "
            f"Loss={epoch_loss:.4f} "
            f"Acc={100 * epoch_acc:.2f}%"
        )

    return model

class CustomCrossEntropyLoss_draft0(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets, available_k):
        """
        Args:
            logits (torch.Tensor): Shape [bsz, L2, D_out]
            targets (torch.Tensor): Shape [bsz, L2]
            available_k (torch.Tensor): Shape [bsz, L2], specifies the number of valid classes for each position
        Returns:
            torch.Tensor: Computed loss
        """
        D_out = logits.size(2)
        # Create indices tensor [0, 1, ..., D_out-1]
        indices = torch.arange(D_out, device=logits.device).view(1, 1, D_out)
        # Expand available_k to [bsz, L2, 1] for broadcasting
        available_k_expanded = available_k.unsqueeze(-1)  # [bsz, L2, 1]
        # Create mask where indices >= available_k (invalid classes)
        mask = indices >= available_k_expanded  # [bsz, L2, D_out]
        # Mask invalid logits to -inf
        masked_logits = logits.masked_fill(mask, -float('inf'))
        # Compute loss using the standard CrossEntropyLoss
        loss = self.criterion(masked_logits.view(-1, D_out), targets.view(-1))
        return loss

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, available_k):
        """
        Args:
            logits: [bsz, L2, D_out]
            targets: [bsz, L2]
            available_k: [bsz, L2] (number of valid classes)
        """
        D_out = logits.size(2)

        # print(f'D_out: {D_out}')
        # print(f'logits.shape: {logits.shape}')
        # print(f'targets.shape: {targets.shape}')
        # print(f'available_k.shape: {available_k.shape}')
        # # Print example values
        # print(f"logits[0, 0]: {logits[0, 0]}")
        # print(f"targets[0, 0]: {targets[0, 0]}")
        # print(f"available_k[0, 0]: {available_k[0, 0]}")



        # 1. Mask logits beyond available_k
        indices = torch.arange(D_out, device=logits.device).view(1, 1, D_out)
        available_k_expanded = available_k.unsqueeze(-1)  # [bsz, L2, 1]
        mask = indices >= available_k_expanded  # [bsz, L2, D_out]
        # Instead of using -inf, we can use a very large negative value
        masked_logits = logits.masked_fill(mask, -1e9)
        # masked_logits = logits.masked_fill(mask, -float('inf'))

        # Print shapes
        # print(f"indices.shape: {indices.shape}")
        # print(f"available_k_expanded.shape: {available_k_expanded.shape}")
        # print(f"mask.shape: {mask.shape}")
        # print(f"masked_logits.shape: {masked_logits.shape}")

        # Print example values
        # print(f"indices[0, 0]: {indices[0, 0]}")
        # print(f"available_k_expanded[0, 0]: {available_k_expanded[0, 0]}")
        # print(f"mask[0, 0]: {mask[0, 0]}")
        # print(f"masked_logits[0, 0]: {masked_logits[0, 0]}")


        # 2. Compute valid positions (available_k > 0 and target < available_k)
        valid_mask = (available_k > 0) & (targets < available_k)  # [bsz, L2]

        # # Print shapes
        # print(f"valid_mask.shape: {valid_mask.shape}")

        # # Print example values
        # print(f"valid_mask[0, 0]: {valid_mask[0, 0]}")

        # 3. Compute log softmax and gather target log probabilities
        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)

        # # Print shapes
        # print(f"log_probs.shape: {log_probs.shape}")

        # # Print example values
        # print(f"log_probs[0, 0]: {log_probs[0, 0]}")

        nll_loss = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [bsz, L2]

        # # Print shapes
        # print(f"nll_loss.shape: {nll_loss.shape}")

        # # Print example values
        # print(f"nll_loss[0, 0]: {nll_loss[0, 0]}")

        # 4. Mask out invalid positions and compute mean loss

        # # Print shapes
        # print(f"valid_mask.shape: {valid_mask.shape}")
        # print(f"nll_loss.shape: {nll_loss.shape}")

        # # # Print example values
        # print(f"valid_mask[0]: {valid_mask[0]}")
        # print(f"nll_loss[0]: {nll_loss[0]}")
        # print(f"valid_mask[0, 63]: {valid_mask[0, 63]}")
        # print(f"nll_loss[0, 63]: {nll_loss[0, 63]}")

        nll_loss = nll_loss * valid_mask.float()

        # # Print shapes
        # print(f"nll_loss.shape: {nll_loss.shape}")

        # # Print example values
        # print(f'nll_loss[0]: {nll_loss[0]}')
        # print(f"nll_loss[0, 0]: {nll_loss[0, 0]}")

        # print(f'valid_mask[0]: {valid_mask[0]}')
        # print(f"valid_mask[0, 0]: {valid_mask[0, 0]}")

        valid_count = valid_mask.sum().float()

        # # Print shapes
        # print(f"valid_count.shape: {valid_count.shape}")

        # # Print example values
        # print(f"valid_count: {valid_count}")

        # When summing nll_loss consider the mask as it includes nan
        # nll_loss shape is [bsz, L2] either nan or valid values
        # valid_mask shape is [bsz, L2] either True or False
        # Convert valid_mask to float to multiply with nll_loss to zero out nans
        # masked_nll_loss_sum = (nll_loss * valid_mask.float()).sum()

        # # Test masking operation
        # print(f'nll_loss[0]: {nll_loss[0]}')
        # print(f'valid_mask[0]: {valid_mask[0]}')
        # print(f'nll_loss[0] * valid_mask[0]: {nll_loss[0] * valid_mask[0].float()}')

        # # Print shapes
        # print(f"masked_nll_loss_sum.shape: {masked_nll_loss_sum.shape}")

        # # Print example values
        # print(f"masked_nll_loss_sum: {masked_nll_loss_sum}")



        #loss = masked_nll_loss_sum / valid_count.clamp(min=1e-6)  # Avoid division by zero

        loss = nll_loss.sum() / valid_count.clamp(min=1e-6)  # Avoid division by zero


        # # Print nll_loss.sum() and valid_count
        # print(f"nll_loss.sum(): {nll_loss.sum()}")
        # print(f'masked_nll_loss_sum: {masked_nll_loss_sum}')
        # print(f"valid_count: {valid_count}")

        # # Print final loss
        # print(f"loss: {loss.item()}")


        return loss

def example_train_5(base_path="./", num_epochs=5, warmup_ratio=0.1):
    """
    Trains the TwoEncoderTransformer model for 'num_epochs' epochs.

    Args:
        num_epochs (int): Number of epochs to train for.
        warmup_ratio (float): Fraction of total steps used for linear warmup.

    Returns:
        model: Trained model.
    """
    # -------------------------------------------------------------------------
    # 1. Model Configuration
    # -------------------------------------------------------------------------
    D_in = 786   # 768 + 18
    D_out = 50   # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    # Create model
    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2
    )
    model.train()

    # -------------------------------------------------------------------------
    # 2. Data Loading (Single Batch Example)
    #    Replace this section with a DataLoader if you have multiple batches.
    # -------------------------------------------------------------------------
    # Load data1: 3D tensor [bsz, L, D_in]
    data1 = load_3d_tensor(os.path.join(base_path, "data_array.bin"))       # e.g. shape [4, 8, 786]
    bsz, L, D_in = data1.shape
    L2 = L // 2

    # Load data2 from type IDs -> embeddings
    type_ids = load_2d_int_tensor(os.path.join(base_path, "encoder_2_input_libcell_type_ids.bin")) # shape [bsz, L2]
    type_embeddings = load_embeddings(os.path.join(base_path, "ASAP7_libcell_type_embeddings.bin")) # shape [num_types, D_emb]
    encoder_2_output_avail_libcell_num = load_2d_int_tensor(os.path.join(base_path, "encoder_2_output_avail_libcell_num.bin")) # shape [bsz, L2]
    mask = (type_ids == -1)
    safe_ids = type_ids.clone()
    safe_ids[mask] = 0  # replace -1 with 0 index
    data2 = type_embeddings[safe_ids]  # shape [bsz, L2, D_emb]
    data2[mask.unsqueeze(-1).expand_as(data2)] = 0  # zero out invalid embeddings

    # Load target: 2D tensor [bsz, L2] of integer class labels
    target = load_2d_int_tensor(os.path.join(base_path, "labels.bin")).long()

    safe_encoder_2_output_avail_libcell_num = (encoder_2_output_avail_libcell_num.clone()).long()

    # Print shapes of data1, data2, and target
    print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}, Target shape: {target.shape}")

    # Print the share of padding tokens in data2
    print(f"Share of padding tokens in data2: {mask.float().mean().item():.2f}")
    # Average number of padding tokens
    print(f"Average number of padding tokens per batch: {mask.sum(dim=1).float().mean().item():.2f}")

    # -------------------------------------------------------------------------
    # 3. Loss Function & Optimizer
    # -------------------------------------------------------------------------
    criterion = CustomCrossEntropyLoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # -------------------------------------------------------------------------
    # 4. Learning Rate Scheduler: Linear Warmup + Linear Decay
    # -------------------------------------------------------------------------
    # We'll assume a total of (num_epochs * 1) "steps" if we only do
    # one step per epoch. Adjust if you have multiple steps per epoch.
    total_steps = num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        """
        1) Warmup from LR=0 -> LR=1 (relative scaling) for 'warmup_steps'.
        2) Then linearly decay from 1 -> 0 from 'warmup_steps' to 'total_steps'.
        """
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------------------------------------------------------
    # 5. Training Loop by Epoch
    # -------------------------------------------------------------------------
    global_step = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(data1, data2)  # shape [bsz, L2, D_out]

        # Compute loss using custom loss function
        loss = criterion(out, target, safe_encoder_2_output_avail_libcell_num)

        # Backprop
        loss.backward()
        optimizer.step()

        # Scheduler step (update learning rate)
        scheduler.step()
        global_step += 1

        # ---------------------------------------------------------------------
        # Compute Accuracy
        # out.shape = [bsz, D_out, L2]
        # argmax over dimension=1 (class dimension) => shape [bsz, L2]
        # Compare with target [bsz, L2]
        # ---------------------------------------------------------------------
        preds = out.argmax(dim=2)  # [bsz, L2]
        correct = (preds == target).sum().item()
        total = target.numel()
        accuracy = correct / total

        # Use mask to exclude padding tokens when computing accuracy
        mask = (type_ids != -1)
        valid_count = mask.sum().item()
        valid_correct = (preds[mask] == target[mask]).sum().item()
        valid_accuracy = valid_correct / valid_count if valid_count > 0 else 0.0

        # ---------------------------------------------------------------------
        # Print epoch metrics
        # ---------------------------------------------------------------------
        current_lr = scheduler.get_last_lr()[0]  # If you have only one param group
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Step {global_step} "
            f"LR={current_lr:.6f} "
            f"Loss={loss.item():.4f} "
            f"Acc={accuracy*100:.2f}% "
            f"Valid Acc={valid_accuracy*100:.2f}%"  # Accuracy excluding padding tokens
        )
    # Print Metrics
    print(f"Number of non-padding tokens: {valid_count}")
    print(f"Number of correct predictions: {valid_correct}")
    print(f"Accuracy: {valid_accuracy*100:.2f}%")
    return model

# -------------------------------------------------------------------------
# 2. Custom Dataset to hold *all* data from multiple base_paths
# -------------------------------------------------------------------------
class MultiPathDataset(Dataset):
    def __init__(self, base_paths):
        """
        Loads and concatenates data from each path in 'base_paths'.
        """
        all_data1 = []
        all_libcell_ids = []
        all_targets = []
        all_type_ids = []
        all_safe_encoder_2 = []

        # We load the shared type embeddings once (assuming the same file for each base_path).
        # If each base_path has a separate embedding file, you can adapt below.
        # Here we assume the same path name "ASAP7_libcell_type_embeddings.bin" in each folder
        # but you can change logic if needed.
        # We'll load the first available embedding file as a demonstration:
        #   type_embeddings = load_embeddings(os.path.join(base_paths[0], "ASAP7_libcell_type_embeddings.bin"))
        # If each base_path has the same embedding, you can do this once outside the loop.
        # If each base_path has different embeddings, adjust accordingly.

        # Just load from the first base_path for demonstration:
        first_path = base_paths[0]
        embedding_file = os.path.join(first_path, "ASAP7_libcell_type_embeddings.bin")
        print(f"Loading type embeddings from: {embedding_file}")
        self.type_embeddings = load_embeddings(embedding_file)  # shape [num_types, D_emb]


        # --- Load data from each path and store temporarily ---
        for bp in base_paths:
            print(f"\nLoading data from base_path={bp}:")
            try: 
                # Load data
                # -------------------------
                # data1: 3D float tensor
                # -------------------------
                data1_path = os.path.join(bp, "data_array.bin")
                data1 = load_3d_tensor(data1_path)  # shape [bsz, L, D_in]

                # -------------------------
                # libcell_ids: 2D int tensor
                # -------------------------
                libcell_ids_path = os.path.join(bp, "libcell_ids.bin")
                libcell_ids = load_2d_int_tensor(libcell_ids_path).long()  # shape [bsz, L]

                # -------------------------
                # type_ids: 2D int tensor
                # -------------------------
                type_ids_path = os.path.join(bp, "encoder_2_input_libcell_type_ids.bin")
                type_ids = load_2d_int_tensor(type_ids_path).long()  # shape [bsz, L2]

                # -------------------------
                # safe_encoder_2_output_avail_libcell_num
                # -------------------------
                avail_libcell_num_path = os.path.join(bp, "encoder_2_output_avail_libcell_num.bin")
                safe_encoder_2 = load_2d_int_tensor(avail_libcell_num_path).long()  # shape [bsz, L2]

                # -------------------------
                # target: 2D tensor [bsz, L2]
                # -------------------------
                label_path = os.path.join(bp, "labels.bin")
                target = load_2d_int_tensor(label_path).long()

                # --- Basic Shape Consistency Check ---
                bsz = data1.shape[0]
                L = data1.shape[1]
                L2 = type_ids.shape[1]
                if not (libcell_ids.shape == (bsz, L) and
                        target.shape == (bsz, L2) and
                        safe_encoder_2.shape == (bsz, L2)):
                    print(f"Warning: Shape mismatch detected in {bp}!")
                    print(f"  data1: {data1.shape}")
                    print(f"  libcell_ids: {libcell_ids.shape}")
                    print(f"  type_ids: {type_ids.shape}")
                    print(f"  target: {target.shape}")
                    print(f"  safe_encoder_2: {safe_encoder_2.shape}")
                    # Decide how to handle: skip this batch, raise error, etc.
                    # For now, we'll try to continue, but padding might behave unexpectedly
                    # if dimensions other than L/L2 mismatch. It's better to fix the data generation.


                # Accumulate original tensors
                all_data1.append(data1)
                all_libcell_ids.append(libcell_ids)
                all_targets.append(target)
                all_type_ids.append(type_ids)
                all_safe_encoder_2.append(safe_encoder_2)

                # Print the shapes of data1, data2, targets, type_ids, and safe_encoder_2
                print(f"Data1 shape: {data1.shape}, Libcell IDs shape: {libcell_ids.shape}, Target shape: {target.shape}")
                print(f"Type IDs shape: {type_ids.shape}, Safe Encoder 2 shape: {safe_encoder_2.shape}")

                # Try to check the statistics of the input data, especially the padding tokens
                valid_data1_mask = (data1 != 0.0)
                valid_libcell_ids_mask = (libcell_ids != -1)
                valid_type_ids_mask = (type_ids != -1)
                # In the current iteration of MLGateSizer.cc labels/targets are initialized to 0, this makes it difficult to distinguish padding tokens from valid tokens
                # As some valid tokens may also be 0, so the share of padding tokens for target is not that reliable
                valid_target_mask = (target != 0) 

                # Pre-calculate padding masks (logical NOT of valid masks)
                padding_data1_mask = ~valid_data1_mask
                padding_libcell_ids_mask = ~valid_libcell_ids_mask
                padding_type_ids_mask = ~valid_type_ids_mask
                padding_target_mask = ~valid_target_mask

                # Calculate counts and totals once
                num_padding_data1 = padding_data1_mask.sum()
                total_data1 = data1.numel()

                num_padding_libcell = padding_libcell_ids_mask.sum()
                total_libcell = libcell_ids.numel()

                num_padding_type = padding_type_ids_mask.sum()
                total_type = type_ids.numel()

                num_padding_target = padding_target_mask.sum()
                total_target = target.numel()

                # Print share information using pre-calculated values
                print(f"Share of padding tokens in data1 (encoder_1 input): {(num_padding_data1 / total_data1).item():.2f} ({num_padding_data1} / {total_data1})")
                print(f"Share of padding tokens in libcell_ids (encoder_1 input): {(num_padding_libcell / total_libcell).item():.2f} ({num_padding_libcell} / {total_libcell})")
                print(f"Share of padding tokens in type_ids (encoder_2 input): {(num_padding_type / total_type).item():.2f} ({num_padding_type} / {total_type})")
                print(f"Share of padding tokens in target (encoder_2 output): {(num_padding_target / total_target).item():.2f} ({num_padding_target} / {total_target})") # Using consistent padding count

                # Calculate sequence lengths (sum of valid tokens per sequence)
                seq_lengths_enc1 = valid_libcell_ids_mask.sum(dim=1).float()
                seq_lengths_enc2 = valid_type_ids_mask.sum(dim=1).float()

                # Calculate and print sequence length stats
                avg_len_enc1 = seq_lengths_enc1.mean().item()
                min_len_enc1 = seq_lengths_enc1.min().item()
                max_len_enc1 = seq_lengths_enc1.max().item()
                print(f"Average, Minimum, Maximum input sequence length of encoder 1 (padding tokens excluded): {avg_len_enc1:.2f}, {min_len_enc1}, {max_len_enc1}")

                avg_len_enc2 = seq_lengths_enc2.mean().item()
                min_len_enc2 = seq_lengths_enc2.min().item()
                max_len_enc2 = seq_lengths_enc2.max().item()
                print(f"Average, Minimum, Maximum input sequence length of encoder 2 (padding tokens excluded): {avg_len_enc2:.2f}, {min_len_enc2}, {max_len_enc2}")

                # Also keep track of average, minimum, maximum safe_encoder_2 values (for non-padding tokens)
                avg_avail_libcell_num = safe_encoder_2[valid_type_ids_mask].float().mean().item()
                min_avail_libcell_num = safe_encoder_2[valid_type_ids_mask].min().item()
                max_avail_libcell_num = safe_encoder_2[valid_type_ids_mask].max().item()
                print(f"Average, Minimum, Maximum safe_encoder_2 values (for non-padding tokens): {avg_avail_libcell_num:.2f}, {min_avail_libcell_num}, {max_avail_libcell_num}")


                # --- Generate and Save Histogram for Encoder 1 Sequence Lengths ---
                if seq_lengths_enc1.numel() > 0: # Only plot if there's data
                    plt.figure(figsize=(10, 6))
                    # Convert tensor to numpy array for matplotlib
                    lengths_np = seq_lengths_enc1.cpu().numpy()
                    plt.hist(lengths_np, bins=min(50, int(max_len_enc1 - min_len_enc1 + 1)), edgecolor='black') # Adjust bins as needed

                    plt.title(f'Encoder 1 Valid Sequence Length Distribution\n(Base Path: {os.path.basename(bp)})')
                    plt.xlabel('Valid Sequence Length (from libcell_ids != -1)')
                    plt.ylabel('Frequency (Number of Sequences)')

                    # Prepare statistics text
                    stats_text = f'Average: {avg_len_enc1:.2f}\nMinimum: {min_len_enc1}\nMaximum: {max_len_enc1}'

                    # Add text box with statistics to the plot
                    # Adjust x, y coordinates (0 to 1 relative to axes) and alignment as needed
                    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

                    plt.grid(axis='y', alpha=0.75)

                    # Define save path and save the figure
                    hist_filename = "encoder1_valid_seq_len_histogram.png"
                    save_path = os.path.join(bp, hist_filename)
                    try:
                        plt.savefig(save_path, bbox_inches='tight')
                        print(f"Histogram saved to: {save_path}\n")
                    except Exception as e:
                        print(f"Error saving histogram to {save_path}: {e}\n")

                    # Close the plot to free memory
                    plt.close()
                else:
                    print("Skipping histogram generation for Encoder 1: No sequence length data found.")

            except FileNotFoundError as e:
                print(f"Error loading data from {bp}: {e}. Skipping this path.")
            except Exception as e:
                print(f"An unexpected error occurred while loading data from {bp}: {e}. Skipping this path.")
        
        if not all_data1: # Check if any data was loaded successfully
            raise ValueError("No data loaded successfully. Check base_paths and file existence.")
        
        # --- Determine Maximum Sequence Lengths ---
        # Max length for encoder 1 inputs (data1, libcell_ids) - dimension 1
        max_len_L = 0
        for t in all_data1:
            max_len_L = max(max_len_L, t.shape[1])
        # Double check with libcell_ids (should ideally be the same)
        for t in all_libcell_ids:
            max_len_L = max(max_len_L, t.shape[1])
        print(f"Determined maximum sequence length L (Encoder 1): {max_len_L}")

        # Max length for encoder 2 inputs/outputs (type_ids, targets, safe_encoder_2) - dimension 1
        max_len_L2 = 0
        for t in all_type_ids:
            max_len_L2 = max(max_len_L2, t.shape[1])
        # Double check with targets and safe_encoder_2
        for t in all_targets:
            max_len_L2 = max(max_len_L2, t.shape[1])
        for t in all_safe_encoder_2:
            max_len_L2 = max(max_len_L2, t.shape[1])
        print(f"Determined maximum sequence length L2 (Encoder 2): {max_len_L2}")


        # --- Pad Tensors and Store in New Lists ---
        padded_data1 = []
        padded_libcell_ids = []
        padded_targets = []
        padded_type_ids = []
        padded_safe_encoder_2 = []

        pad_value_float = 0.0
        pad_value_int = -1 # Use -1 for padding integer IDs, matching mask logic
        pad_value_target = 0 # Use 0 for padding target labels and safe_encoder_2(avail_libcell_num)

        for i in range(len(all_data1)):
            # Pad tensors related to L
            current_L = all_data1[i].shape[1]
            pad_amount_L = max_len_L - current_L
            # F.pad format: (pad_left, pad_right, pad_top, pad_bottom, ...) applied from last dim backwards
            # For data1 (B, L, D), pad L (dim 1): needs (0, 0 for D, 0, pad_amount_L for L)
            if pad_amount_L > 0:
                padded_data1.append(F.pad(all_data1[i], (0, 0, 0, pad_amount_L), mode='constant', value=pad_value_float))
                # For libcell_ids (B, L), pad L (dim 1): needs (0, pad_amount_L for L)
                padded_libcell_ids.append(F.pad(all_libcell_ids[i], (0, pad_amount_L), mode='constant', value=pad_value_int))
            else: # No padding needed if already max length
                padded_data1.append(all_data1[i])
                padded_libcell_ids.append(all_libcell_ids[i])


            # Pad tensors related to L2
            current_L2 = all_type_ids[i].shape[1]
            pad_amount_L2 = max_len_L2 - current_L2
            if pad_amount_L2 > 0:
                # For type_ids, targets, safe_encoder_2 (B, L2), pad L2 (dim 1): needs (0, pad_amount_L2 for L2)
                padded_type_ids.append(F.pad(all_type_ids[i], (0, pad_amount_L2), mode='constant', value=pad_value_int))
                padded_targets.append(F.pad(all_targets[i], (0, pad_amount_L2), mode='constant', value=pad_value_target))
                padded_safe_encoder_2.append(F.pad(all_safe_encoder_2[i], (0, pad_amount_L2), mode='constant', value=pad_value_target))
            else: # No padding needed
                padded_type_ids.append(all_type_ids[i])
                padded_targets.append(all_targets[i])
                padded_safe_encoder_2.append(all_safe_encoder_2[i])


        # ----------------------------------------------------
        # Concatenate PADDED tensors along dimension=0 (the batch dimension)
        # ----------------------------------------------------
        self.data1 = torch.cat(padded_data1, dim=0)     # shape [sum(bsz), L_max, D_in], encoder_1_numerical data
        self.libcell_ids = torch.cat(padded_libcell_ids, dim=0)     # shape [sum(bsz), L_max], encoder_1_libcell_ids
        self.targets = torch.cat(padded_targets, dim=0) # shape [sum(bsz), L2_max]
        self.type_ids = torch.cat(padded_type_ids, dim=0) # shape [sum(bsz), L2_max], encoder_2_input_libcell_type_ids
        self.safe_encoder_2 = torch.cat(padded_safe_encoder_2, dim=0) # shape [sum(bsz), L2_max], encoder_2_output_avail_libcell_num

        # Just to confirm final shapes:
        print(f"\n--- After Padding and Concatenation ---")
        print(f"Final combined data1 shape: {self.data1.shape}")
        print(f"Final combined libcell_ids shape: {self.libcell_ids.shape}")
        print(f"Final combined targets shape: {self.targets.shape}")
        print(f"Final combined type_ids shape: {self.type_ids.shape}")
        print(f"Final combined safe_encoder_2 shape: {self.safe_encoder_2.shape}")

        

    def __len__(self):
        return self.data1.size(0)  # total number of "samples" across all base_paths

    def __getitem__(self, idx):
        """
        Return a single sample: (data1, libcell_ids, target, type_ids, safe_encoder_2).
        """
        return (
            self.data1[idx],
            self.libcell_ids[idx],
            self.targets[idx],
            self.type_ids[idx],
            self.safe_encoder_2[idx]
        )

# =====================================================================
# Helper function for plotting
# =====================================================================
def plot_training_progress(
    batch_steps, batch_losses,
    epoch_steps, epoch_train_losses, epoch_train_accuracies,
    epoch_val_losses, epoch_val_accuracies,
    save_path=None
):
    """
    Plots batch training loss, epoch training/validation loss, and
    epoch training/validation accuracy with improved clarity and dual x-axis
    (Global Step and Epoch).

    Uses solid lines for training metrics and dashed lines for validation metrics.
    Uses blue/cyan colors for loss and red/orange colors for accuracy.

    Args:
        batch_steps (list): List of global steps corresponding to each batch loss.
        batch_losses (list): List of training losses for each batch.
        epoch_steps (list): List of global steps at the end of each epoch.
                            Should have the same length as epoch_* lists.
        epoch_train_losses (list): List of average training losses for each epoch.
        epoch_train_accuracies (list): List of training accuracies for each epoch.
        epoch_val_losses (list): List of average validation losses for each epoch.
        epoch_val_accuracies (list): List of validation accuracies for each epoch.
        save_path (str, optional): Path to save the plot image. If None, displays the plot.
    """
    if not epoch_steps:
        print("Warning: epoch_steps list is empty. Cannot plot epoch data or epoch axis.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7)) # Increased figure size slightly

    # --- Primary X-axis: Global Step ---
    ax1.set_xlabel('Global Step')

    # --- Loss Plotting (Left Y-axis) ---
    color_loss_train = 'tab:blue'
    color_loss_val = 'tab:cyan'
    ax1.set_ylabel('Loss', color=color_loss_train)
    ax1.tick_params(axis='y', labelcolor=color_loss_train)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7, color=color_loss_train) # Grid for loss

    # Plot batch loss (smoothed trend) - Use training loss color with low alpha
    # Plot this first so it's in the background
    if batch_steps and batch_losses:
      ax1.plot(batch_steps, batch_losses, color=color_loss_train, alpha=0.2, label='Batch Training Loss (Trend)')

    # Plot epoch training loss - Solid line
    line1, = ax1.plot(epoch_steps, epoch_train_losses, 'o-', color=color_loss_train, linewidth=2, label='Train Loss')

    # Plot epoch validation loss - Dashed line
    line2 = None
    if epoch_val_losses:
        line2, = ax1.plot(epoch_steps, epoch_val_losses, 's--', color=color_loss_val, linewidth=2, label='Validation Loss')

    # --- Accuracy Plotting (Right Y-axis) ---
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_acc_train = 'tab:red'
    color_acc_val = 'tab:orange'
    ax2.set_ylabel('Accuracy', color=color_acc_train)
    ax2.tick_params(axis='y', labelcolor=color_acc_train)
    ax2.set_ylim(min(0, min(epoch_train_accuracies or [0]), min(epoch_val_accuracies or [0])) - 0.05, 1.05) # Dynamic lower bound slightly below min acc

    # Plot epoch training accuracy - Solid line
    line3, = ax2.plot(epoch_steps, epoch_train_accuracies, '^-', color=color_acc_train, linewidth=2, label='Train Accuracy')

    # Plot epoch validation accuracy - Dashed line
    line4 = None
    if epoch_val_accuracies:
        line4, = ax2.plot(epoch_steps, epoch_val_accuracies, 'd--', color=color_acc_val, linewidth=2, label='Validation Accuracy')

    # --- Secondary X-axis: Epoch ---
    ax_epoch = ax1.twiny() # Create a shared y-axis but independent x-axis

    # Position the secondary x-axis below the primary one
    ax_epoch.spines["bottom"].set_position(("axes", -0.15)) # Adjust spacing as needed
    # Hide the top spine and ticks/labels
    ax_epoch.spines["top"].set_visible(False)
    ax_epoch.xaxis.set_ticks_position('bottom')
    ax_epoch.xaxis.set_label_position('bottom')

    # Set the ticks and labels for the epoch axis
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_xlim(ax1.get_xlim()) # Match the limits of the primary x-axis
    ax_epoch.set_xticks(epoch_steps) # Set tick positions based on global steps at epoch end
    ax_epoch.set_xticklabels([i + 1 for i in range(len(epoch_steps))]) # Label ticks with epoch numbers (1, 2, ...)

    # --- Final Touches ---
    fig.suptitle('Training Progress (Loss & Accuracy)', fontsize=16)

    # Combine legends from both y-axes
    lines = [line1]
    if line2: lines.append(line2)
    lines.append(line3)
    if line4: lines.append(line4)
    # Add batch loss line to legend if plotted
    if batch_steps and batch_losses:
        batch_line = ax1.get_lines()[0] # Get the first line plotted (batch loss)
        lines.append(batch_line)

    labels = [l.get_label() for l in lines]
    # Use ax1 for the legend placement, adjust location if needed
    ax1.legend(lines, labels, loc='center right')

    # Adjust layout to prevent labels overlapping, especially the new x-axis
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust bottom margin for epoch axis label

    if save_path:
        plt.savefig(save_path, bbox_inches='tight') # bbox_inches ensures labels aren't cut off
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory

# -------------------------------------------------------------------------
# 3. Example Training Function using DataLoader
# -------------------------------------------------------------------------
def train_multiple_paths(
    base_paths,
    val_paths=None,
    num_epochs=5,
    warmup_ratio=0.1,
    batch_size=2,
    shuffle=True,
    device="cpu",
    plot_save_path="training_plot.png"
):
    """
    Demonstration of how to train using data from multiple base_paths with mini-batching.
    """
    # --- Data storage for plotting ---
    batch_losses = []
    batch_steps = []
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_val_losses = []
    epoch_val_accuracies = []
    epoch_steps = []
    # --------------------------------

    # ---------------------------------------------------------------------
    # A) Create model (same as your single-batch code)
    # ---------------------------------------------------------------------
    D_in = 18   # 768 + 18
    D_out = 50   # maximum number of classes
    D_emb = 768
    D_model = 128
    FF_hidden_dim = 4 * D_model  # 512
    num_heads = 8
    num_encoder_layers = 6
    num_encoder_layers_2 = 2

    # Load libcell and libcell type embeddings
    print("Loading libcell and libcell type embeddings...")
    libcell_type_embeddings = load_embeddings(os.path.join(base_paths[0], "ASAP7_libcell_type_embeddings.bin"))
    libcell_embeddings = load_embeddings(os.path.join(base_paths[0], "ASAP7_libcell_embeddings.bin"))

    model = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim,
        num_heads, num_encoder_layers, num_encoder_layers_2,
        libcell_embedding=libcell_embeddings,
        libcell_type_embedding=libcell_type_embeddings
    )
    model.to(device)
    model = torch.compile(model)
    model.train()

    # ---------------------------------------------------------------------
    # B) Create the Dataset + DataLoader
    # ---------------------------------------------------------------------
    print("Loading Training Data...")
    dataset = MultiPathDataset(base_paths=base_paths)
    # The type embeddings are stored in dataset.type_embeddings
    # if you want them in your training loop for anything else.

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    # Create a validation dataset if provided
    print("Loading Validation Data...")
    val_dataset = MultiPathDataset(base_paths=val_paths) if val_paths else None

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    # ---------------------------------------------------------------------
    # C) Define Loss and Optimizer
    # ---------------------------------------------------------------------
    criterion = CustomCrossEntropyLoss()  # or nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ---------------------------------------------------------------------
    # D) LR Scheduler: Linear Warmup & Decay (per-step)
    # ---------------------------------------------------------------------
    total_steps = num_epochs * len(dataloader)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print("Training started.")
    # ---------------------------------------------------------------------
    # E) Training Loop
    # ---------------------------------------------------------------------
    global_step = 0
    for epoch in range(num_epochs):
        # Optionally track epoch-level metrics
        epoch_loss = 0.0

        epoch_padding_excluded_correct = 0
        epoch_padding_excluded_total = 0

        # Wrap the training dataloader with a tqdm progress bar.
        train_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for step, (data1_batch, libcell_ids_batch, target_batch, type_ids_batch, safe_encoder_2_batch) in enumerate(train_bar):
            # Move data to device
            data1_batch = data1_batch.to(device)
            libcell_ids_batch = libcell_ids_batch.to(device)
            target_batch = target_batch.to(device)
            type_ids_batch = type_ids_batch.to(device)
            safe_encoder_2_batch = safe_encoder_2_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            out = model(data1_batch, libcell_ids_batch, type_ids_batch)  # shape [bsz, L2, D_out]

            # Compute loss
            loss = criterion(out, target_batch, safe_encoder_2_batch)

            # --- Store batch loss and step for plotting ---
            current_loss = loss.item()
            batch_losses.append(current_loss)
            batch_steps.append(global_step)
            # ---------------------------------------------

            loss.backward()
            optimizer.step()
            scheduler.step()   # Update LR per-step



            epoch_loss += current_loss
            global_step += 1

            # -----------------------------------------------------------------
            # Accuracy (excluding padding)
            # The padding tokens at end of each sequence are excluded from accuracy calculation
            # They can be identified as the type_ids == -1 for the second encoder
            # -----------------------------------------------------------------
            with torch.no_grad():
                preds = out.argmax(dim=2)  # shape [bsz, L2]

                # Use type_ids to compute accuracy excluding padding tokens
                mask = (type_ids_batch != -1) # Shape: [bsz, L2], valid positions
                valid_correct = (preds[mask] == target_batch[mask]).sum().item()
                valid_total = mask.sum().item()
                valid_accuracy = valid_correct / valid_total if valid_total > 0 else 0.0
                epoch_padding_excluded_correct += valid_correct
                epoch_padding_excluded_total += valid_total

            # Update progress bar with current loss and learning rate
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")



        # Print epoch summary
        epoch_padding_excluded_acc = epoch_padding_excluded_correct / epoch_padding_excluded_total if epoch_padding_excluded_total > 0 else 0.0
        avg_epoch_loss = epoch_loss / len(dataloader)

        # --- Store epoch training metrics for plotting ---
        epoch_train_losses.append(avg_epoch_loss)
        epoch_train_accuracies.append(epoch_padding_excluded_acc)
        epoch_steps.append(global_step) # Record step count at the end of the epoch
        # ------------------------------------------------

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Global Step={global_step} | "
            f"LR={scheduler.get_last_lr()[0]:.6f} | "
            f"Loss={avg_epoch_loss:.4f} | "
            f"Accuracy={epoch_padding_excluded_acc*100:.2f}%"  # Accuracy excluding padding tokens
        )
        # Validation loop (if provided)
        avg_val_loss = float('nan') # Default if no validation
        val_padding_excluded_acc = float('nan')
        if val_dataloader:
            model.eval()
            val_loss_total = 0.0
            val_padding_excluded_correct = 0
            val_padding_excluded_total = 0
            with torch.no_grad():

                for val_data1, val_data1_libcell_id, val_target, val_type_ids, val_safe_encoder_2 in val_dataloader:
                    val_data1 = val_data1.to(device)
                    val_data1_libcell_id = val_data1_libcell_id.to(device)
                    val_target = val_target.to(device)
                    val_type_ids = val_type_ids.to(device)
                    val_safe_encoder_2 = val_safe_encoder_2.to(device)

                    val_out = model(val_data1, val_data1_libcell_id, val_type_ids)
                    val_loss = criterion(val_out, val_target, val_safe_encoder_2)

                    val_loss_total += val_loss.item()

                    val_preds = val_out.argmax(dim=2)

                    val_mask = (val_type_ids != -1)
                    val_valid_correct = (val_preds[val_mask] == val_target[val_mask]).sum().item()
                    val_valid_total = val_mask.sum().item()
                    val_padding_excluded_correct += val_valid_correct
                    val_padding_excluded_total += val_valid_total
            # Compute validation metrics
            val_padding_excluded_acc = val_padding_excluded_correct / val_padding_excluded_total if val_padding_excluded_total > 0 else 0.0
            avg_val_loss = val_loss_total / len(val_dataloader)

            # --- Store epoch validation metrics for plotting ---
            epoch_val_losses.append(avg_val_loss)
            epoch_val_accuracies.append(val_padding_excluded_acc)
            # ----------------------------------------------------

            # Print validation metrics
            print(
                f"Validation | "
                f"Loss={avg_val_loss:.4f} | "
                f"Accuracy={val_padding_excluded_acc*100:.2f}%"  # Accuracy excluding padding tokens
            )
            model.train() # Set back to train mode
        else:
            # If no validation, append NaN or skip to keep lists aligned if needed,
            # but our plotting function handles potentially empty val lists.
            epoch_val_losses.append(float('nan'))
            epoch_val_accuracies.append(float('nan'))


    print("Training complete.")

    # --- Plotting after training ---
    print("Generating training plot...")
    plot_training_progress(
        batch_steps, batch_losses,
        epoch_steps, epoch_train_losses, epoch_train_accuracies,
        epoch_val_losses if val_dataloader else [], # Pass empty lists if no validation
        epoch_val_accuracies if val_dataloader else [],
        save_path=plot_save_path
    )
    # -------------------------------

    return model

if __name__ == "__main__":
    #trained_model = example_train()
    #type_embeddings = load_embeddings("/home/kmcho/2_Project/ML_GateSizing_OpenROAD/dev_repo/test_scripts/pytorch_transsizer_training_code/ASAP7_libcell_type_embeddings.bin")
    #trained_model = example_train_2(num_epochs=100, warmup_ratio=0.2)
    #trained_model = example_train_3(batch_size=4, num_epochs=100, warmup_ratio=0.2, lr=1e-3)
    #trained_model = example_train_4(batch_size=4, num_epochs=10, warmup_ratio=0.2, lr=1e-3, device='cuda')
    # Added loading encoder_2_output_avail_libcell_num from example_train_5 onwards
    # This masks out libcell's that exceed the maximum number of available libcells for that particular libcell type
    #trained_model = example_train_5(base_path= "./NV_NVDLA_partition_m", num_epochs=1, warmup_ratio=0.2)
    torch.set_float32_matmul_precision('high')

    print("Test models with padding")
    # test_attention_masking()
    # test_two_encoder_transformer_masking()

    train_base_dir = ["./NV_NVDLA_partition_m"]#["./ariane136", "./NV_NVDLA_partition_m", "./NV_NVDLA_partition_p", "./mempool_tile_wrap"]#["./train_data/NV_NVDLA_partition_m"]
#["./train_data/NV_NVDLA_partition_p", "./train_data/ariane136", "./train_data/aes_256"]#,"./train_data/NV_NVDLA_partition_m"] #["./train_data/NV_NVDLA_partition_m"] #["./train_data/NV_NVDLA_partition_p", "./train_data/ariane136", "./train_data/aes_256"]#, "./train_data/mempool_tile_wrap"]
    val_base_dir = ["./NV_NVDLA_partition_m"]#["./aes_256"]#["./train_data/NV_NVDLA_partition_m"]
    trained_model = train_multiple_paths(
        base_paths=train_base_dir,
        val_paths=val_base_dir,
        num_epochs=10,#200,#2,
        warmup_ratio=0.2,
        batch_size=512,
        shuffle=True,
        device="cuda:0"
    )

    """
    # Create Dataloaders
    train_loader, val_loader = create_train_val_dataloaders(
        train_base_dir, val_base_dir,
        batch_size=16,
        shuffle=True,
        num_workers=0)

    device =  "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_model(
        train_loader, val_loader,
        num_epochs=10,
        warmup_ratio=0.2,
        device=device
    )
    """



    print("Training completed.")
    # Print model summary
    print(trained_model)
    # Print parameters
    for name, param in trained_model.named_parameters():
        print(name, param.size())
    # Print total number of parameters
    print(f"Total number of parameters: {sum(p.numel() for p in trained_model.parameters())}")
    # Export the trained model
    export_model_weights_robust(trained_model, "transformer_params.bin")
