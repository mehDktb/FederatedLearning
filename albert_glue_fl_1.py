# !pip install nvidia-ml-py3
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
import csv
import os
import json
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AutoConfig, AdamW
from tqdm import tqdm
from threading import Thread
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo, nvmlShutdown
import time
from torch.optim import AdamW
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from threading import Thread
import matplotlib.pyplot as plt
import json
import os
from collections import Counter

class LoRALayer(nn.Module):
    """
    A LoRA layer that modifies the attention mechanism by introducing low-rank matrices.
    """

    def __init__(self, original_layer, rank=8):
        super().__init__()

        # Store original layer
        self.original_layer = original_layer

        # Check if the attention layer has query, key, and value attributes
        if hasattr(original_layer, 'query') and hasattr(original_layer, 'key') and hasattr(original_layer, 'value'):
            query_dim = original_layer.query.weight.size(1)
            key_dim = original_layer.key.weight.size(1)
            value_dim = original_layer.value.weight.size(1)
        else:
            raise ValueError("Original layer does not contain query, key, or value attributes")

        # Define LoRA parameters (low-rank adaptation)
        self.rank = rank
        self.lora_q = nn.Linear(query_dim, rank, bias=False)
        self.lora_k = nn.Linear(key_dim, rank, bias=False)
        self.lora_v = nn.Linear(value_dim, rank, bias=False)

        # Initialize LoRA parameters
        nn.init.normal_(self.lora_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_v.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # Get the original outputs (query, key, value)
        query = self.original_layer.query(hidden_states)
        key = self.original_layer.key(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = self.original_layer.value(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        # Apply LoRA modifications
        query = query + self.lora_q(query)
        key = key + self.lora_k(key)
        value = value + self.lora_v(value)

        # Forward pass through original attention mechanism
        return self.original_layer.attention(query, key, value, attention_mask=attention_mask)
# calculates the number of clients (in this test just returns 2)
def get_number_of_clients():
    return 2


def tokenize_function(examples, tokenizer, max_length):
    # For tasks with a single sentence (like SST-2)
    if "sentence" in examples:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    # For tasks with sentence pairs (e.g., MRPC)
    elif "sentence1" in examples and "sentence2" in examples:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=max_length)
    else:
        raise ValueError("Expected keys 'sentence' or 'sentence1'/'sentence2' not found in examples.")

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def compute_metrics(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    return {"accuracy": (predictions == labels).mean()}

def measure_function_energy(func, args=(), kwargs=None, interval=1, output_file=""):
    if kwargs is None:
        kwargs = {}

    print("we are here")
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    energy_joules = 0
    memory_usage_mb = 0

    func_thread = Thread(target=func, args=args, kwargs=kwargs)

    def monitor(thread):
        nonlocal energy_joules
        nonlocal memory_usage_mb
        while thread.is_alive():
            power_mw = nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            energy_joules += power_w * interval

            mem_info = nvmlDeviceGetMemoryInfo(handle)
            memory_usage_mb += mem_info.used / (1024 * 1024)

            time.sleep(interval)

    func_thread.start()

    monitor_thread = Thread(target=monitor, args=(func_thread,))
    monitor_thread.start()

    func_thread.join()
    monitor_thread.join()

    nvmlShutdown()

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([energy_joules, memory_usage_mb])

# preparing the dataset for each client
# I have added the test dataset to each client
def prepare_and_split_dataset(task, client_count, SEED, PARENT_FOLDER):
    os.makedirs(PARENT_FOLDER, exist_ok=True)
    random.seed(SEED)

    print(f"Downloading dataset {task}")
    dataset = load_dataset("glue", task)['train']
    print(f"Length of {task} Train Dataset: {len(dataset)}")


    test_dataset = load_dataset("glue", task)['validation']

    label_0 = [item for item in dataset if item['label'] == 0]
    label_1 = [item for item in dataset if item['label'] == 1]

    random.shuffle(label_0)
    random.shuffle(label_1)

    label_0_test = [item for item in test_dataset if item['label'] == 0]
    label_1_test = [item for item in test_dataset if item['label'] == 1]

    random.shuffle(label_0_test)
    random.shuffle(label_1_test)

    split_0 = [label_0[i::client_count] for i in range(client_count)]
    split_1 = [label_1[i::client_count] for i in range(client_count)]

    split_0_test = [label_0_test[i::client_count] for i in range(client_count)]
    split_1_test = [label_1_test[i::client_count] for i in range(client_count)]

    for client_id in range(client_count):
        client_data = split_0[client_id] + split_1[client_id]
        random.shuffle(client_data)

        client_data_test = split_0_test[client_id] + split_1_test[client_id]
        random.shuffle(client_data_test)

        client_file = os.path.join(PARENT_FOLDER, f"train_{client_id}.json")
        client_file_test = os.path.join(PARENT_FOLDER, f"test_{client_id}.json")

        with open(client_file, 'w', encoding='utf-8') as f:
            json.dump(client_data, f, ensure_ascii=False, indent=2)

        print(f"Saved data for client {client_id} with {len(client_data)} samples.")

        with open(client_file_test, 'w', encoding='utf-8') as f:
            json.dump(client_data_test, f, ensure_ascii=False, indent=2)

        print(f"Saved train dataset for client {client_id} with {len(client_data)} samples.")
        print(f"Saved test dataset for client {client_id} with {len(client_data_test)} samples.")



    print(f"Dataset prepared and split into {client_count} parts in folder '{PARENT_FOLDER}'.")


def evaluate(model, data_loader, loss_fn):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            loss = outputs.loss
            logits = outputs.logits
            batch_size = batch['labels'].size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_examples += batch_size
    avg_loss = total_loss / total_examples if total_examples > 0 else 0
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    model.train()  # Switch back to train mode
    return avg_loss, accuracy


def perturbed_inference_train(model, train_loader, num_epochs, learning_rate, perturbation_strength):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    gradient_accumulation_steps = 4

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss = 0
        batch_count = 0
        optimizer.zero_grad()

        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for i, batch in enumerate(train_progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            logits = outputs.logits
            labels = batch["labels"]

            loss_fn = torch.nn.CrossEntropyLoss()
            current_loss = loss_fn(logits, labels)
            loss_value = current_loss.item()
            total_loss += loss_value
            batch_count += 1
            train_progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            if i % 2 == 0:
                perturbed_logits = logits + torch.randn_like(logits) * (perturbation_strength / 2)
                perturbed_loss = loss_fn(perturbed_logits, labels)
                total_loss_batch = (current_loss + perturbed_loss) / 2
            else:
                total_loss_batch = current_loss

            total_loss_batch = total_loss_batch / gradient_accumulation_steps
            total_loss_batch.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            del outputs, logits, current_loss, total_loss_batch
            if 'perturbed_logits' in locals():
                del perturbed_logits
            if 'perturbed_loss' in locals():
                del perturbed_loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / batch_count:.4f}")

    return model

def bp_train(model, train_loader, test_loader, num_epochs, learning_rate, training_steps, warmup_steps, client_id):
    print(">>>> Start normal fine tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.01)
    # Set up the learning rate scheduler with warmup steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

    loss_fn = torch.nn.CrossEntropyLoss()
    gradient_accumulation_steps = 4

    model.train()
    metrics = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])

            loss = outputs.loss
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()
            total_batches += 1

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()

            del outputs, loss, scaled_loss
            torch.cuda.empty_cache()

            if total_batches > 0:
                avg_loss = total_loss / total_batches
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        train_epoch_loss, train_epoch_accuracy = evaluate(model, train_loader, loss_fn)
        test_epoch_loss, test_epoch_accuracy = evaluate(model, test_loader, loss_fn)

        print(f"Epoch {epoch+1} - "
              f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.4f} | "
              f"Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_accuracy:.4f}")

        metrics.append({
            "Round": epoch + 1,
            "Train Accuracy": train_epoch_accuracy,
            "Train Loss": train_epoch_loss,
            "Test Accuracy": test_epoch_accuracy,
            "Test Loss": test_epoch_loss
        })

    filename = f"client_metrics_{client_id}.csv"
    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = ["Round", "Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print(f"Metrics appended to {filename}")

    return model

def check_label_balance(data, threshold=0.1):
        """
        Checks and prints the label distribution for the dataset.

        Args:
            data (list): List of dataset instances (each a dict with a 'label' key).
            threshold (float): Acceptable relative difference for balanced labels.
                            For two labels, balanced if the smaller count is at least
                            (1 - threshold) of the larger count.
        """
        labels = [instance['label'] for instance in data]
        label_counts = Counter(labels)
        total = sum(label_counts.values())

        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} instances ({(count / total) * 100:.2f}%)")

        if len(label_counts) > 1:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            if min_count / max_count >= (1 - threshold):
                print("=> Dataset is balanced.\n")
            else:
                print("=> Dataset is unbalanced.\n")
        else:
            print("=> Only one label present in the dataset.\n")

def train_for_client(client_id, model_path, tokenizer, batch_size, num_epochs, learning_rate,
                     perturbation_strength, max_length, training_steps, warmup_steps):

    print(f">>>>Start training client {client_id}")
    client_file_index = client_id - 1
    with open(f"datasets/train_{client_file_index}.json", "r", encoding="utf-8") as f:
        raw_data_train = json.load(f)
    print(f"Client {client_id} TRAIN DATA:")
    check_label_balance(raw_data_train)

    train_dataset = Dataset.from_list(raw_data_train)
    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
    train_loader = DataLoader(tokenized_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    with open(f"datasets/test_{client_file_index}.json", "r", encoding="utf-8") as tf:
        raw_data_test = json.load(tf)
    print(f"Client {client_id} TEST DATA:")
    check_label_balance(raw_data_test)

    test_dataset = Dataset.from_list(raw_data_test)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
    test_loader = DataLoader(tokenized_test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    # Load configuration from the saved model path and set dropout parameters
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    config.classifier_dropout_prob = 0.1

    if os.path.exists(model_path):
        print(f"Loading global model")
        model = AlbertForSequenceClassification.from_pretrained(model_path, config=config)
    else:
        print(f"Global model not found. Starting from base pre-trained model.")
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", config=config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pass training_steps and warmup_steps to bp_train
    measure_function_energy(
        bp_train,
        args=(model, train_loader, test_loader, num_epochs, learning_rate, training_steps, warmup_steps, client_id),
        interval=1,
        output_file=f"energy_log_client_{client_id}.csv"
    )

    # Return state dict
    return model.state_dict()

def evaluate_global_model(task, model, tokenizer, device, batch_size, max_length):
    print(">>>>Start evaluation for global model")
    eval_dataset = load_dataset("glue", task)["validation"]
    eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)

    eval_loader = DataLoader(eval_dataset, batch_size, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    loss_fn = torch.nn.CrossEntropyLoss()

    eval_progress_bar = tqdm(eval_loader, desc="Evaluating Global Model", leave=False)
    with torch.no_grad():
        for batch in eval_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            labels = batch["labels"]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(eval_loader)
    metrics = compute_metrics(all_predictions, all_labels)
    print(f">>>>Done evaluation for global model: Accuracy: {metrics['accuracy']:.4f}, Loss: {average_loss:.4f}")
    return metrics['accuracy'], average_loss

def structured_pruning(model: AlbertForSequenceClassification, prune_rate: float):
    count_parameters(model, False)
    print(">>>>Start structured pruning ...")
    albert_model = model.albert
    for group in albert_model.encoder.albert_layer_groups:
        for layer in group.albert_layers:
            attention = layer.attention

            if hasattr(attention, "self"):
                # Old format (attention.self contains query/key/value)
                attn = attention.self
            else:
                # Newer format (attention directly contains query/key/value)
                attn = attention

            num_heads = attn.num_attention_heads
            head_size = attn.attention_head_size

            heads_to_prune = int(num_heads * prune_rate)

            if heads_to_prune > 0:
                # Compute importance scores per head (L1 norm of query weights per head)
                head_scores = []
                for head_idx in range(num_heads):
                    start = head_idx * head_size
                    end = (head_idx + 1) * head_size
                    head_weight = attn.query.weight[start:end]
                    head_scores.append(torch.norm(head_weight, p=1).item())

                # Sort heads by importance and prune the least important ones
                heads_sorted = sorted(range(num_heads), key=lambda x: head_scores[x])
                heads_to_remove = heads_sorted[:heads_to_prune]

                # Zero-out query/key/value weights for the pruned heads
                with torch.no_grad():
                    for head in heads_to_remove:
                        start = head * head_size
                        end = (head + 1) * head_size
                        attn.query.weight.data[start:end] = 0
                        attn.key.weight.data[start:end] = 0
                        attn.value.weight.data[start:end] = 0
    print(">>>>Done structerd pruning.")
    count_parameters(model, True)
    return model

def unstructured_pruning(model, prune_rate: float):
    count_parameters(model, False)
    print(">>>>Start unstructured pruning ...")
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            all_weights.append(param.data.view(-1))

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights.abs(), prune_rate)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                mask = param.abs() >= threshold
                param.data *= mask

    print(">>>>Done unstructerd pruning.")
    count_parameters(model, True)
    return model

def add_lora_to_albert_model(model, rank):
    albert_model = model.albert
    # Iterate through each layer group and layer in ALBERT's encoder
    for group in albert_model.encoder.albert_layer_groups:
        for layer in group.albert_layers:
            attention = layer.attention

            # Check if attention has 'query', 'key', and 'value' attributes
            if hasattr(attention, 'query') and hasattr(attention, 'key') and hasattr(attention, 'value'):
                original_attention = attention
                # Replace the attention layer with the LoRA-modified attention
                layer.attention = LoRALayer(original_attention, rank)
    return model

def count_parameters(model, active):
    if active:
        active_params = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight_mask'):  # Check if pruning mask exists
                    active_params += int(module.weight_mask.sum())  # Count non-zero weights
                else:
                    active_params += module.weight.numel()
        print(f">>>>**Active parameters** in model {model} is {active_params}")
    else:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>>>**All parameters** in model {model} is {params}")


#creating model
def create_model(model_name, model_path):
    from transformers import AutoConfig  # Ensure AutoConfig is imported
    # Load configuration and set dropout hyperparameters
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    config.hidden_dropout_prob = 0.1              # ALBERT DR: dropout for ALBERT layers
    config.attention_probs_dropout_prob = 0.1       # ALBERT DR: dropout in attention probabilities
    config.classifier_dropout_prob = 0.1            # Classifier DR: dropout in the classification head

    # Load model and tokenizer with the updated configuration
    global_model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    global_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f">>>>{model_name} saved in {model_path}")


def main():
    round_number = 40
    max_length = 128
    batch_size = 16
    seed = 42
    num_epochs = 1
    learning_rate = 2e-5  # Static learning rate as requested
    perturbation_strength = 0.1
    training_steps = 20935  # TS (Training Steps)
    warmup_steps = 1252     # WS (Warmup Steps)

    model_name = "albert-base-v2"
    model_path = "./global_models"
    dataset_train_path = "datasets"
    task = "sst2"

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    create_model(model_name, model_path)
    global_model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AlbertTokenizer.from_pretrained(model_path)

    num_clients = get_number_of_clients()
    prepare_and_split_dataset(task, num_clients, seed, dataset_train_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up CSV file for tracking metrics
    with open("federated_training_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy", "Loss"])

    # Store initial model for checking significant deviations
    initial_state_dict = {k: v.clone().detach().cpu() for k, v in global_model.state_dict().items()}

    for round_num in range(1, round_number+1):
        print(f">>>>>Start round {round_num}")

        # List to collect client models
        client_model_states = []

        for client_id in range(1, num_clients + 1):
            # Pass training_steps and warmup_steps to train_for_client
            client_model_state = train_for_client(
                client_id, model_path, tokenizer, batch_size, num_epochs,
                learning_rate, perturbation_strength, max_length, training_steps, warmup_steps
            )

            # Convert to CPU for consistent aggregation
            client_model_states.append({k: v.cpu() for k, v in client_model_state.items()})


        # Inside main() loop aggregation:
        global_model_state = {}
        for key in client_model_states[0].keys():
            # Ensure all client tensors are CPU and float32
            tensors = [client_state[key].float().cpu() for client_state in client_model_states]

            # Check for NaN/Inf in individual client models
            valid_tensors = []
            for t in tensors:
                if not (torch.isnan(t).any() and not (torch.isinf(t)).any()):
                    valid_tensors.append(t)

            if len(valid_tensors) == 0:
                print(f"All clients have NaN/Inf in {key}, using initial value")
                global_model_state[key] = initial_state_dict[key].clone()
                continue

            # Average valid tensors
            avg_tensor = torch.stack(valid_tensors).mean(dim=0)

            # Check averaged tensor
            if torch.isnan(avg_tensor).any() or torch.isinf(avg_tensor).any():
                print(f"NaN/Inf detected in {key} after averaging, using initial value")
                global_model_state[key] = initial_state_dict[key].clone()
            else:
                global_model_state[key] = avg_tensor

        # Move state to model's device before loading
        device = next(global_model.parameters()).device
        global_model_state = {k: v.to(device) for k, v in global_model_state.items()}
        global_model.load_state_dict(global_model_state)

        # Save checkpoint
        os.makedirs("global_models", exist_ok=True)
        global_model.save_pretrained(f"./global_models")
        print(f"Global Model saved for Round {round_num}")

        # Evaluate the model
        accuracy, loss = evaluate_global_model(task, global_model, tokenizer, device, batch_size, max_length)

        print(f"Round {round_num} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        # Write results to CSV
        with open("federated_training_results.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([round_num, accuracy, loss])

    # Final evaluation and summary
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
