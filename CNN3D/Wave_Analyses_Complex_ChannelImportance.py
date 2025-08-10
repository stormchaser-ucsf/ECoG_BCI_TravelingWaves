# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 18:54:09 2025

@author: nikic
"""


from iAE_utils_models import *

#%% (MAIN MAIN) CHANNEL ABLATION EXPERIMENTS 


# get baseline classification loss from the model 
r,i,decodes = test_model_complex(model, Xtest)
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 
decodes =  torch.from_numpy(decodes).to(device).float()
test_labels_torch = torch.from_numpy(labels_test).to(device).float()
classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')# input. target
baseline_loss = classif_criterion(torch.squeeze(decodes),test_labels_torch)

Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
num_batches = math.ceil(Xtest_real.shape[0]/2048)
idx = (np.arange(Xtest_real.shape[0]))
idx_split = np.array_split(idx,num_batches)

num_layers = round(sum(1 for m in model.encoder.modules() if isinstance(m, nn.Conv3d))/2)
results = []
for layer in range(1, num_layers+1):  # conv1 to conv6
    print(layer)
    num_channels = getattr(model.encoder, f"conv{layer}").real_conv.out_channels
    for ch in range(num_channels):
        # have to loop over batches here
        score=[]
        for batch in range(num_batches):
            
            samples = idx_split[batch]
            Xtest_real_batch = torch.from_numpy(Xtest_real[samples,:]).to(device).float()
            Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples,:]).to(device).float()
            labels_batch = torch.from_numpy(labels_test[samples]).to(device).float()
                
            tmp = ablate_encoder_channel_complex(model, 
                        Xtest_real_batch, Xtest_imag_batch, layer, ch,labels_batch)
            score.append(tmp.item())
        
        score = sum(score)/Xtest.shape[0]
        score = score/baseline_loss.item()
        score = score/num_channels
        results.append((layer, ch, score))

torch.cuda.empty_cache()
torch.cuda.ipc_collect() 


# getting stats
layer_4_results = [(ch, score) for lyr, ch, score in results if lyr == 4]
layer_4_scores = [score for lyr, ch, score in results if lyr == 1]
important_channels = [(layer, ch, score) for (layer, ch, score) in results if score > 1.1]

# plotting
import collections
layer_dict = collections.defaultdict(list)
for layer, ch, score in results:
    layer_dict[layer].append(score)

# Plot per layer
for layer in sorted(layer_dict.keys()):
    plt.figure()
    plt.bar(range(len(layer_dict[layer])), layer_dict[layer])
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title(f"Ablation Scores for Layer {layer}")
    plt.xlabel("Channel")
    plt.ylabel("Score (Ablated Loss / Baseline Loss)")
    plt.show()


#%% LOOKING AT ACTIVATION PER CHANNEL LAYER TO SEE IF OL/CL ACTIVATES CERTAIN CHANNELS/LAYERS

results_act = []
#num_layers =  len(list(model.encoder.children()))
num_layers = round(sum(1 for m in model.encoder.modules() if isinstance(m, nn.Conv3d))/2)
elu = nn.ELU()
for layer in range(1, num_layers+1):  # conv1 to conv6
    print(f"Layer {layer}")
    num_channels = getattr(model.encoder, f"conv{layer}").real_conv.out_channels
    
    for ch in range(num_channels):
        act_class_0 = []
        act_class_1 = []

        for batch in range(num_batches):
            samples = idx_split[batch]

            Xtest_real_batch = torch.from_numpy(Xtest_real[samples, :]).to(device).float()
            Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples, :]).to(device).float()
            labels_batch = torch.from_numpy(labels_test[samples]).to(device).float()

            # Forward manually up to current layer
            a, b = Xtest_real_batch.clone(), Xtest_imag_batch.clone()
            for i in range(1, layer + 1):
                conv = getattr(model.encoder, f"conv{i}")
                a, b = conv(a, b)
                a,b = elu(a),elu(b)

            # Compute magnitude
            mag = torch.sqrt(a**2 + b**2)  # shape: [B, C, H, W, T]
            mag_ch = mag[:, ch]  # shape: [B, H, W, T]

            # Mean over spatial/temporal dims
            mag_mean = mag_ch.view(mag_ch.size(0), -1).mean(dim=1)  # shape: [B]

            # Split by class
            act_class_0.extend(mag_mean[labels_batch == 0].tolist())
            act_class_1.extend(mag_mean[labels_batch == 1].tolist())

        mean_act_0 = sum(act_class_0) / len(act_class_0) if act_class_0 else 0.0
        mean_act_1 = sum(act_class_1) / len(act_class_1) if act_class_1 else 0.0

        results_act.append((layer, ch, mean_act_0, mean_act_1))


# Organize results per layer
from collections import defaultdict
layer_to_diffs = defaultdict(list)

for layer, ch, mean_0, mean_1 in results_act:
    diff = abs(mean_0 - mean_1)
    layer_to_diffs[layer].append((ch, diff))

# Plot layer by layer
num_layers = len(layer_to_diffs)
fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 3 * num_layers))

if num_layers == 1:
    axes = [axes]  # make it iterable if only one layer

for i, layer in enumerate(sorted(layer_to_diffs.keys())):
    ch_ids, diffs = zip(*sorted(layer_to_diffs[layer]))
    ax = axes[i]
    ax.bar(ch_ids, diffs)
    ax.set_title(f"Layer {layer}: Activation Difference (|class0 - class1|)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Activation Difference")
    ax.set_xticks(ch_ids)

plt.tight_layout()
plt.show()


#%% PLOTTING ABLATION VS ACTIVATION STRENGTH DIFFERENCE



# Convert to dict for easy lookup
act_diff_dict = {(l, ch): abs(m0 - m1) for l, ch, m0, m1 in results_act}
ablation_dict = {(l, ch): score for l, ch, score in results}

# Make sure keys match in both
common_keys = sorted(set(act_diff_dict.keys()) & set(ablation_dict.keys()))

# Extract paired values
act_diffs = [act_diff_dict[k] for k in common_keys]
ablation_scores = [ablation_dict[k] for k in common_keys]
labels = [f"L{l}_C{ch}" for l, ch in common_keys]

# Scatter plot
plt.figure(figsize=(7, 6))
plt.scatter(act_diffs, ablation_scores, alpha=0.7, edgecolor='k')

# Optionally label points
for i, label in enumerate(labels):
    plt.text(act_diffs[i] + 0.002, ablation_scores[i] + 0.002, label, fontsize=8)

plt.xlabel("Activation Magnitude Difference (|Class0 - Class1|)")
plt.ylabel("Classification Loss Ratio (Ablation / Baseline)")
plt.title("Channel-wise: Activation Difference vs Classification Importance")
plt.grid(True)
plt.show()

#%% EXAMINING GRADIENTS WRT TEST LOSS AT INDIVIDUAL CHANNELS/LAYERS

# ==== 1. Storage ====
activations = {}
gradients_sum = defaultdict(lambda: 0.0)
num_batches_seen = defaultdict(lambda: 0)

# Hook to store activations and enable grad tracking
def get_hook(name):
    def hook(module, input, output):
        activations[name] = output
        output.retain_grad()
    return hook

# ==== 2. Register hooks ====
hook_handles = []
for idx, layer in enumerate(model.encoder.children()):
    # For complex conv layers with separate real and imaginary convs
    if hasattr(layer, "real_conv") and hasattr(layer, "imag_conv"):
        hook_handles.append(layer.real_conv.register_forward_hook(get_hook(f"layer{idx+1}_real")))
        hook_handles.append(layer.imag_conv.register_forward_hook(get_hook(f"layer{idx+1}_imag")))
    elif isinstance(layer, nn.Conv3d):  # Fallback if it's just a regular conv
        hook_handles.append(layer.register_forward_hook(get_hook(f"layer{idx+1}")))

# ==== 3. Loop over batches ====
model.encoder.eval()
model.decoder.eval()
model.classifier.train()
classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')

Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
num_batches = math.ceil(Xtest_real.shape[0]/2048)
idx = (np.arange(Xtest_real.shape[0]))
idx_split = np.array_split(idx,num_batches)


for batch_idx in range(num_batches):
    samples = idx_split[batch_idx]

    Xtest_real_batch = torch.from_numpy(Xtest_real[samples, :]).to(device).float()
    Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples, :]).to(device).float()
    labels_batch = torch.from_numpy(labels_test[samples]).to(device).float()

    # Forward pass
    r,i,logits = model(Xtest_real_batch, Xtest_imag_batch)
    loss = classif_criterion(logits.squeeze(), labels_batch)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Accumulate absolute gradients per channel
    for name, act in activations.items():
        grad = act.grad  # shape: [B, C, ...]
        ch_importance = grad.abs().mean(dim=(0, 2, 3, 4))  # mean |grad| per channel
        gradients_sum[name] += ch_importance.detach().cpu().numpy()
        num_batches_seen[name] += 1

# ==== 4. Combine real + imag into a single magnitude ====
combined_importance = defaultdict(list)
for name in gradients_sum:
    if "_real" in name:
        base = name.replace("_real", "")
        real_vals = gradients_sum[name] / num_batches_seen[name]
        imag_vals = gradients_sum[name.replace("_real", "_imag")] / num_batches_seen[name.replace("_real", "_imag")]
        combined_mag = (real_vals**2 + imag_vals**2)**0.5
        combined_importance[base] = combined_mag

# ==== 5. Plot per-layer ====
for layer_name, ch_vals in combined_importance.items():
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(ch_vals)), ch_vals)
    plt.title(f"{layer_name}: Gradient Magnitude per Channel (real+imag combined)")
    plt.xlabel("Channel")
    plt.ylabel("Mean sqrt(real_grad^2 + imag_grad^2)")
    plt.show()

# ==== 6. Remove hooks ====
for h in hook_handles:
    h.remove()


torch.cuda.empty_cache()
torch.cuda.ipc_collect() 


