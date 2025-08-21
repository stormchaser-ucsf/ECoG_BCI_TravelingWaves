# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 18:54:09 2025

@author: nikic
"""



#%% (MAIN MAIN) CHANNEL ABLATION EXPERIMENTS 


from iAE_utils_models import *
# get baseline classification loss from the model 
r,i,decodes = test_model_complex(model, Xtest)
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 
decodes =  torch.from_numpy(decodes).to(device).float()
test_labels_torch = torch.from_numpy(labels_test).to(device).float()
classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')# input. target
baseline_loss = classif_criterion(torch.squeeze(decodes),test_labels_torch)

Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
num_batches = math.ceil(Xtest_real.shape[0]/128)
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

from iAE_utils_models import *
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
                #a,b = elu(a),elu(b)
                z = ((a**2) + (b**2))**0.5
                a,b = a*elu(z)/z, b*elu(z)/z

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


from iAE_utils_models import *

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


from iAE_utils_models import *

torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

# ==== 1. Storage ====
activations = {}
gradients_sum = defaultdict(lambda: 0.0)
num_batches_seen = defaultdict(lambda: 0)
total_samples_seen = defaultdict(lambda: 0)

# Hook to store activations and enable grad tracking
def get_hook(name):
    def hook(module, input, output):
        activations[name] = output
        output.retain_grad()
    return hook

# # ==== 2. Register hooks ====
hook_handles = []
for idx, layer in enumerate(model.encoder.children()):
    # For complex conv layers with separate real and imaginary convs
    if hasattr(layer, "real_conv") and hasattr(layer, "imag_conv"):
        hook_handles.append(layer.real_conv.register_forward_hook(get_hook(f"layer{idx+1}_real")))
        hook_handles.append(layer.imag_conv.register_forward_hook(get_hook(f"layer{idx+1}_imag")))
    elif isinstance(layer, nn.Conv3d):  # Fallback if it's just a regular conv
        hook_handles.append(layer.register_forward_hook(get_hook(f"layer{idx+1}")))

# ==== 2. Register hooks decoder layers ====
# hook_handles = []
# for idx, layer in enumerate(model.decoder.children()):
#     # For complex conv layers with separate real and imaginary convs
#     if hasattr(layer, "real_deconv") and hasattr(layer, "imag_deconv"):
#         hook_handles.append(layer.real_deconv.register_forward_hook(get_hook(f"layer{idx+1}_real")))
#         hook_handles.append(layer.imag_deconv.register_forward_hook(get_hook(f"layer{idx+1}_imag")))
#     elif isinstance(layer, nn.Conv3d):  # Fallback if it's just a regular conv
#         hook_handles.append(layer.register_forward_hook(get_hook(f"layer{idx+1}")))



# ==== 3. Loop over batches ====
model.encoder.eval()
model.decoder.eval()
model.classifier.train()
classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')
recon_criterion = nn.MSELoss(reduction='mean')

Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
Ytest_real,Ytest_imag = Ytest.real,Ytest.imag
num_batches = math.ceil(Xtest_real.shape[0]/256)
idx = (np.arange(Xtest_real.shape[0]))
idx_split = np.array_split(idx,num_batches)


for batch_idx in range(num_batches):
    samples = idx_split[batch_idx]

    Xtest_real_batch = torch.from_numpy(Xtest_real[samples, :]).to(device).float()
    Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples, :]).to(device).float()
    Ytest_real_batch = torch.from_numpy(Ytest_real[samples, :]).to(device).float()
    Ytest_imag_batch = torch.from_numpy(Ytest_imag[samples, :]).to(device).float()
    labels_batch = torch.from_numpy(labels_test[samples]).to(device).float()

    # Forward pass
    r,i,logits = model(Xtest_real_batch, Xtest_imag_batch)
    loss = classif_criterion(logits.squeeze(), labels_batch)
    #loss1 = recon_criterion(r,Ytest_real_batch)
    #loss2 = recon_criterion(i,Ytest_imag_batch)
    #loss=1*(loss1+loss2)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Accumulate absolute gradients per channel
    for name, act in activations.items():
        grad = act.grad  # shape: [B, C, H,W,T]
        #print(grad.shape)
        #print(name)
        #ch_importance = grad.abs().sum(dim=(0, 2, 3, 4))  # mean |grad| per channel
        ch_importance = grad.abs().mean(dim=( 2, 3, 4))  # mean |grad| over dimensions, gets mean gradient per channel
        ch_importance = ch_importance.sum(dim=0) # sum over all the batch samples as batch samples not uniform per batch
        gradients_sum[name] += ch_importance.detach().cpu().numpy()
        num_batches_seen[name] += 1
        total_samples_seen[name] += len(samples)

# ==== 4. Combine real + imag into a single magnitude ====
combined_importance = defaultdict(list)
for name in gradients_sum:
    if "_real" in name:
        base = name.replace("_real", "")
        #real_vals = gradients_sum[name] / num_batches_seen[name]
        real_vals = gradients_sum[name] / total_samples_seen[name]
        #imag_vals = gradients_sum[name.replace("_real", "_imag")] / num_batches_seen[name.replace("_real", "_imag")]
        imag_vals = gradients_sum[name.replace("_real", "_imag")] / total_samples_seen[name.replace("_real", "_imag")]
        combined_mag = (real_vals**2 + imag_vals**2)**0.5
        combined_importance[base] = combined_mag

# # ==== 5. Plot per-layer ====
# for layer_name, ch_vals in combined_importance.items():
#     plt.figure(figsize=(8, 3))
#     plt.bar(range(len(ch_vals)), ch_vals)
#     plt.title(f"{layer_name}: Gradient Magnitude per Channel (real+imag combined)")
#     plt.xlabel("Channel")
#     plt.ylabel("Mean sqrt(real_grad^2 + imag_grad^2)")
#     plt.show()

# # ==== 5a. Plot all at once ====
# n_layers = len(combined_importance)
# fig, axes = plt.subplots( n_layers, 1,figsize=(8 ,3* n_layers), sharey=True,sharex=False)

# # Make sure axes is iterable even if there’s only 1 layer
# if n_layers == 1:
#     axes = [axes]

# for ax, (layer_name, ch_vals) in zip(axes, combined_importance.items()):
#     ax.bar(range(len(ch_vals)), ch_vals)
#     ax.set_title(f"{layer_name}")
#     ax.set_xlabel("Channel")
#     ax.set_ylabel("Mean sqrt(real^2 + imag^2)")

# plt.tight_layout()
# plt.show()


# ==== 6. Remove hooks ====
for h in hook_handles:
    h.remove()


torch.cuda.empty_cache()
torch.cuda.ipc_collect() 


# Prepare a combined x-axis index for plotting
all_channels = []
all_values = []
layer_boundaries = []
offset = 0

for layer_name, ch_vals in combined_importance.items():
    ch_vals = np.array(ch_vals)
    all_channels.extend(range(offset, offset + len(ch_vals)))
    all_values.extend(ch_vals)
    offset += len(ch_vals)
    layer_boundaries.append(offset)  # store where layers end

# Create one figure
plt.figure(figsize=(14, 4))
plt.bar(all_channels, all_values, color='steelblue')

# Add vertical lines to mark layer boundaries
for boundary in layer_boundaries[:-1]:  # skip final boundary
    plt.axvline(boundary - 0.5, color='red', linestyle='--', alpha=0.7)

# Add labels
plt.xlabel("Channel (across all layers)")
plt.ylabel("Mean sqrt(real_grad² + imag_grad²)")
plt.title("Gradient Magnitude per Channel")

# Add layer names in the middle of each layer's range
midpoints = [0] + layer_boundaries[:-1]
for i, name in enumerate(combined_importance.keys()):
    midpoint = (midpoints[i] + layer_boundaries[i] - 1) / 2
    plt.text(midpoint, max(all_values) * 0.99, name,
             ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.show()

#%% GRAD CAM ANALYSES USING SAME CODE STRUCTURE AS ABOVE


# ==== 6. Remove hooks ====
for h in hook_handles:
    h.remove()


torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

# ===== Storage =====
activations = {}
per_channel_importance_real = defaultdict(lambda: 0.0)
per_channel_importance_imag = defaultdict(lambda: 0.0)
total_samples_seen_real = defaultdict(int)
total_samples_seen_imag = defaultdict(int)

def get_hook(name):
    """Store activations and retain gradients."""
    def hook(module, input, output):
        activations[name] = output
        output.retain_grad()
    return hook

# ===== User setting =====
target_layer_base = "layer2"  # <-- base name, no _real/_imag needed

# ===== Register hooks =====
hook_handles = []
for idx, layer in enumerate(model.encoder.children()):
    layer_name_base = f"layer{idx+1}"
    # print(layer_name_base)
    # if idx==7:
    #     print(layer)        

    if layer_name_base == target_layer_base:
        # Complex conv/deconv case
        if hasattr(layer, "real_deconv") and hasattr(layer, "imag_deconv"):
            hook_handles.append(layer.real_deconv.register_forward_hook(get_hook(f"{layer_name_base}_real")))
            hook_handles.append(layer.imag_deconv.register_forward_hook(get_hook(f"{layer_name_base}_imag")))
        elif hasattr(layer, "real_conv") and hasattr(layer, "imag_conv"):
            hook_handles.append(layer.real_conv.register_forward_hook(get_hook(f"{layer_name_base}_real")))
            hook_handles.append(layer.imag_conv.register_forward_hook(get_hook(f"{layer_name_base}_imag")))
        elif isinstance(layer, nn.Conv3d):
            hook_handles.append(layer.register_forward_hook(get_hook(layer_name_base)))

# ===== Grad-CAM computation =====
model.encoder.eval()
model.decoder.eval()
model.classifier.train()

idx = np.where(labels_test==0)[0]
Xtest_real, Xtest_imag = Xtest.real, Xtest.imag
Ytest_real, Ytest_imag = Ytest.real, Ytest.imag
Xr = Xtest_real[idx,:]
Xi = Xtest_imag[idx,:]
Yr = Ytest_real[idx,:]
Yi = Ytest_imag[idx,:]

num_batches = math.ceil(Xr.shape[0] / 256)
idx_split = np.array_split(np.arange(Xr.shape[0]), num_batches)

gradcam_sum_real = None
gradcam_sum_imag = None
total_samples = 0

for batch_idx in range(num_batches):
    samples = idx_split[batch_idx]

    X_real_batch = torch.from_numpy(Xr[samples]).to(device).float()
    X_imag_batch = torch.from_numpy(Xi[samples]).to(device).float()
    Y_real_batch = torch.from_numpy(Yr[samples]).to(device).float()
    Y_imag_batch = torch.from_numpy(Yi[samples]).to(device).float()

    # Forward
    r, i, logits = model(X_real_batch, X_imag_batch)
    #score = logits.mean()
    s1 = torch.square(Y_real_batch - r)
    s2 = torch.square(Y_imag_batch - i)
    s = torch.sqrt(s1+s2)
    score = s.mean()

    # Backward
    model.zero_grad()
    score.backward(retain_graph=True)

    # ---- Real & Imag parts ----
    real_name = f"{target_layer_base}_real"
    imag_name = f"{target_layer_base}_imag"

    if real_name in activations and imag_name in activations:
        # ----- REAL -----
        
        act_real = activations[real_name]
        grad_real = activations[real_name].grad
        w_real = grad_real.mean(dim=(2, 3, 4), keepdim=True)
        cam_real = (w_real * act_real).sum(dim=1)  # sum over channels
        ch_mag_real = (w_real * act_real).abs().mean(dim=(2, 3, 4))  # per-channel
        per_channel_importance_real[target_layer_base] += ch_mag_real.sum(dim=0).detach().cpu().numpy()
        total_samples_seen_real[target_layer_base] += ch_mag_real.shape[0]

        cam_real = F.relu(cam_real)
        cam_real = cam_real / (cam_real.max() + 1e-8)
        cam_real_resized = F.interpolate(
            cam_real.unsqueeze(1), size=X_real_batch.shape[2:], mode='trilinear', align_corners=False
        ).squeeze(1)

        # ----- IMAG -----
        act_imag = activations[imag_name]
        grad_imag = act_imag.grad
        w_imag = grad_imag.mean(dim=(2, 3, 4), keepdim=True)
        cam_imag = (w_imag * act_imag).sum(dim=1)
        ch_mag_imag = (w_imag * act_imag).abs().mean(dim=(2, 3, 4))
        per_channel_importance_imag[target_layer_base] += ch_mag_imag.sum(dim=0).detach().cpu().numpy()
        total_samples_seen_imag[target_layer_base] += ch_mag_imag.shape[0]

        cam_imag = F.relu(cam_imag)
        cam_imag = cam_imag / (cam_imag.max() + 1e-8)
        cam_imag_resized = F.interpolate(
            cam_imag.unsqueeze(1), size=X_real_batch.shape[2:], mode='trilinear', align_corners=False
        ).squeeze(1)

    else:
        # Single real-valued case
        name = target_layer_base
        act = activations[name]
        grad = act.grad
        w = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam_real = (w * act).sum(dim=1)
        ch_mag_real = (w * act).abs().mean(dim=(2, 3, 4))
        per_channel_importance_real[target_layer_base] += ch_mag_real.sum(dim=0).detach().cpu().numpy()
        total_samples_seen_real[target_layer_base] += ch_mag_real.shape[0]

        cam_real = F.relu(cam_real)
        cam_real = cam_real / (cam_real.max() + 1e-8)
        cam_real_resized = F.interpolate(
            cam_real.unsqueeze(1), size=X_real_batch.shape[2:], mode='trilinear', align_corners=False
        ).squeeze(1)
        cam_imag_resized = None  # not used

    # ---- Accumulate ----
    cam_real_np = cam_real_resized.detach().cpu().numpy()
    if gradcam_sum_real is None:
        gradcam_sum_real = np.sum(cam_real_np, axis=0)
    else:
        gradcam_sum_real += np.sum(cam_real_np, axis=0)

    if cam_imag_resized is not None:
        cam_imag_np = cam_imag_resized.detach().cpu().numpy()
        if gradcam_sum_imag is None:
            gradcam_sum_imag = np.sum(cam_imag_np, axis=0)
        else:
            gradcam_sum_imag += np.sum(cam_imag_np, axis=0)

    total_samples += cam_real_np.shape[0]

# cleaning up
del score,r,i,logits,activations,X_real_batch,X_imag_batch, Y_real_batch,Y_imag_batch
del s1,s2,s
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ===== Average spatial Grad-CAM =====
gradcam_avg_real = gradcam_sum_real / total_samples
gradcam_avg_imag = gradcam_sum_imag / total_samples if gradcam_sum_imag is not None else None

# ===== Average per-channel magnitudes =====
per_channel_avg_real = per_channel_importance_real[target_layer_base] / total_samples_seen_real[target_layer_base]
per_channel_avg_imag = per_channel_importance_imag[target_layer_base] / total_samples_seen_imag[target_layer_base]

# ===== Visualize REAL Grad-CAM =====

# Plotting
x  = (gradcam_avg_real**2 + gradcam_avg_imag**2)**0.5
x1 = np.moveaxis(x, -1, 0)  # Shape: (40, 11, 23)
fig, ax = plt.subplots()
im = ax.imshow(x1[0], cmap='jet', animated=True)
title = ax.set_title("Time: 0", fontsize=12)
#ax.set_title("Optimized Input Over Time")
ax.axis('off')

def update(frame):
    im.set_array(x1[frame])    
    title.set_text(f"Time: {frame}/{x1.shape[0]}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=x1.shape[0], interval=100, blit=False)

# Show the animation
plt.show()
# save the animation
filename = 'Grad_CAM_'  + target_layer_base + 'OL_Mag_Recon_v2_ROI.gif'
ani.save(filename, writer="pillow", fps=6)



# phasor animation
scaler = MaxAbsScaler()
xreal = gradcam_avg_real;
ximag = gradcam_avg_imag;
# xc = xreal + 1j*ximag
# xc_abs = np.abs(xc)
xreal = xreal
ximag = ximag

# xreal = 2 * ((xreal - xreal.min()) / (xreal.max() - xreal.min())) - 1
# ximag = 2 * ((ximag - ximag.min()) / (ximag.max() - ximag.min())) - 1
fig, ax = plt.subplots(figsize=(6, 6))

def update(t):
    #plot_phasor_frame_time(xreal, ximag, t, ax)
    plot_phasor_frame(xreal, ximag, t, ax)
    return []

#ani = animation.FuncAnimation(fig, update, frames=xreal.shape[0], blit=False)
ani = animation.FuncAnimation(fig, update, frames=xreal.shape[2], blit=False)

plt.show()

# save the animation
filename = 'Grad_CAM_'  + target_layer_base + 'OL_Phasor_Recon_v2_ROI.gif'
ani.save(filename, writer="pillow", fps=4)

plt.plot(xreal[0,0,:])
plt.plot(ximag[0,0,:])

plt.show();



#%%
fig, ax = plt.subplots()

im = ax.imshow(x1[0, :, :], cmap='magma', origin='lower')

def update(frame):
    im.set_array(gradcam_avg_real[frame, :, :])
    ax.set_title(f"REAL - Slice {frame}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=x1.shape[0], interval=100, blit=True)
plt.show()
# save the animation
ani.save("Grad_CAM_Layer4_OL.gif", writer="pillow", fps=6)

# ===== Visualize IMAG Grad-CAM (if exists) =====
if gradcam_avg_imag is not None:
    fig, ax = plt.subplots()
    im = ax.imshow(gradcam_avg_imag[0, :, :], cmap='jet', origin='lower')

    def update_imag(frame):
        im.set_array(gradcam_avg_imag[frame, :, :])
        ax.set_title(f"IMAG - Slice {frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update_imag, frames=gradcam_avg_imag.shape[0], interval=100, blit=True)
    plt.show()

# ===== Print per-channel importance =====
print(f"Per-channel REAL Grad-CAM magnitudes for {target_layer_base}:")
print(per_channel_avg_real)
if gradcam_avg_imag is not None:
    print(f"Per-channel IMAG Grad-CAM magnitudes for {target_layer_base}:")
    print(per_channel_avg_imag)


#%% DO PCA TO EXAMINE ACTIVATIONS OF INDIVIDUAL CHANNELS OF A LAYER

# PRELIMS
from iAE_utils_models import *
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

# get the CNN architecture model

from iAE_utils_models import *

if 'model' in locals():
    del model 
 
model = model_class(ksize,num_classes,input_size,lstm_size).to(device)
model.load_state_dict(torch.load(nn_filename))


# GET THE ACTIVATIONS FROM A CHANNEL LAYER OF INTEREST
layer_name = 'layer3'
channel_idx = 7
batch_size=256

activations_real, activations_imag = get_channel_activations(model, Xtest, Ytest,
                                    labels_test,device,layer_name,
                                    channel_idx,batch_size)

activations = activations_real + 1j*activations_imag

# RUN COMPLEX PCA
eigvals, eigmaps, Z = complex_pca(activations,5)

# plot phasors of the eigenmaps
pc_idx=1;
H,W = eigmaps.shape[:2]
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
U = eigmaps[:,:,pc_idx].real
V = eigmaps[:,:,pc_idx].imag
plt.figure()
plt.quiver(X,Y,U,V,angles='xy')
plt.xlim(X.min()-1,X.max()+1)
plt.ylim(Y.min()-1,Y.max()+1)

# plot phase map
ph = np.angle(eigmaps[:,:,pc_idx])
ph = np.cos(ph)
plt.figure()
plt.imshow(ph,vmin=-1, vmax=1)
plt.colorbar()

# plot activations
scores = Z[100,:,pc_idx]
# a = np.angle(scores)
# a = np.cos(a)
a = np.abs(scores)
plt.figure()
plt.plot(a)





