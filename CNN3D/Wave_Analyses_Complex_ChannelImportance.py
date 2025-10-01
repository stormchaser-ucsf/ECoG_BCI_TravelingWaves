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

#%% (MAIN) EXAMINING GRADIENTS WRT TEST LOSS AT INDIVIDUAL CHANNELS/LAYERS


from iAE_utils_models import *

torch.cuda.empty_cache()
torch.cuda.ipc_collect() 



if 'model' in locals():
    del model 
 
model = model_class(ksize,num_classes,input_size,lstm_size).to(device)
model.load_state_dict(torch.load(nn_filename))


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

# Xtest_real,Xtest_imag = Xtest.real,Xtest.imag
# Ytest_real,Ytest_imag = Ytest.real,Ytest.imag
idx = np.where(labels_val==0)[0]
Xtest_real,Xtest_imag = Xval[idx,:].real,Xval[idx,:].imag
Ytest_real,Ytest_imag = Yval[idx,:].real,Yval[idx,:].imag


num_batches = math.ceil(Xtest_real.shape[0]/256)
idx = (np.arange(Xtest_real.shape[0]))
idx_split = np.array_split(idx,num_batches)


for batch_idx in range(num_batches):
    samples = idx_split[batch_idx]

    Xtest_real_batch = torch.from_numpy(Xtest_real[samples, :]).to(device).float()
    Xtest_imag_batch = torch.from_numpy(Xtest_imag[samples, :]).to(device).float()
    Ytest_real_batch = torch.from_numpy(Ytest_real[samples, :]).to(device).float()
    Ytest_imag_batch = torch.from_numpy(Ytest_imag[samples, :]).to(device).float()
    #labels_batch = torch.from_numpy(labels_test[samples]).to(device).float()
    labels_batch = torch.from_numpy(labels_val[samples]).to(device).float()

    # Forward pass
    r,i,logits = model(Xtest_real_batch, Xtest_imag_batch)
    #loss = classif_criterion(logits.squeeze(), labels_batch)
    loss1 = recon_criterion(r,Ytest_real_batch)
    loss2 = recon_criterion(i,Ytest_imag_batch)
    loss=1*(loss1+loss2)

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
target_layer_base = "layer3"  # <-- base name, no _real/_imag needed

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

idx = np.where(labels_test==1)[0]
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
    score = logits.mean()
    #s1 = torch.square(Y_real_batch - r)
    #s2 = torch.square(Y_imag_batch - i)
    #s = torch.sqrt(s1+s2)
    #score = s.mean()

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
        
        #get only the channel you care about
        act_real = act_real[:,11,:,:,:][:,None,:,:,:]
        grad_real = grad_real[:,11,:,:,:][:,None,:,:,:]
        
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
        
        #get only the channel you care about
        act_imag = act_imag[:,11,:,:,:][:,None,:,:,:]
        grad_imag = grad_imag[:,11,:,:,:][:,None,:,:,:]
        
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
#del s1,s2,s
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
filename = 'Grad_CAM_'  + target_layer_base + 'CL_Mag_Class_v2_ROI.gif'
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
filename = 'Grad_CAM_'  + target_layer_base + 'CL_Phasor_Class_v2_ROI.gif'
ani.save(filename, writer="pillow", fps=4)

plt.plot(xreal[0,0,:])
plt.plot(ximag[0,0,:])

plt.show();

#%% (MAIN) LOOK AT DIFFERENCES BETWEEN OL AND CL SEPARATELY AFTER PROJECTING ONTO THE LAYER
# PCA, DAY BY DAY


#PC0: much more trial to trial variability, phase noise, but higher activation levels
# higher amplitude and more variability in activation, less precise

# NOTES: Layer 3, Ch 11:
    # PC0: higher trial to trial variability in OL than CL, but also greater amplitude modulation
           #this also tends to characterize a standing wave pattern 

# PRELIMS
from iAE_utils_models import *
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

if 'model' in locals():
    del model 

num_classes=1    
input_size=384*2
lstm_size=32
ksize=2;
model_class = Autoencoder3D_Complex_deep
nn_filename = 'i3DAE_B3_Complex_New.pth' 
model = model_class(ksize,num_classes,input_size,lstm_size).to(device)
model.load_state_dict(torch.load(nn_filename))

# GET THE ACTIVATIONS FROM A CHANNEL LAYER OF INTEREST
layer_name = 'layer4'
channel_idx = 14
batch_size=256

# init variables
OL=[]
CL=[]
noise_stats=[]
var_stats=[]
mean_stats = []
mean_statsA=[]
mean_statsB=[]
var_statsA=[]
var_statsB=[]
for day_idx in np.arange(10)+1:
    
    
    idx_days = np.where(labels_test_days == day_idx)[0]
    tmp_labels = labels_test[idx_days]
    tmp_ydata = Ytest[idx_days,:]
    tmp_xdata = Xtest[idx_days,:]
    
    activations_real, activations_imag = get_channel_activations(model, tmp_xdata, tmp_ydata,
                                        tmp_labels,device,layer_name,
                                        channel_idx,batch_size)
    activations = activations_real + 1j*activations_imag
    
    # RUN COMPLEX PCA on OL
    idx = np.where(tmp_labels==0)[0]
    activations_ol = activations[idx,:]
    eigvals, eigmaps, Zproj , VAF,eigvecs,_ = complex_pca(activations_ol,15)
    #plt.stem(VAF)
    
    # RUN COMPLEX PCA on CL
    idx = np.where(tmp_labels==1)[0]
    activations_cl = activations[idx,:]
    eigvals1, eigmaps1, Zproj1 , VAF1,eigvecs1 ,_= complex_pca(activations_cl,15)
    #plt.stem(VAF1)
    
    
    # PLOT EIGMAPS AS PHASORS
    pc_idx=2
    H,W = eigmaps.shape[:2]
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    U = eigmaps[:,:,pc_idx].real
    V = eigmaps[:,:,pc_idx].imag
    Z1 = U+1j*V
    plt.figure()
    plt.quiver(X,Y,U,V,angles='xy')
    plt.gca().invert_yaxis()
    OL.append(Z1)
    plt.xlim(X.min()-1,X.max()+1)
    plt.ylim(Y.min()-1,Y.max()+1)
    plt.title('OL Day ' + str(day_idx))
    
    
    #pc_idx=1
    H,W = eigmaps1.shape[:2]
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    U = eigmaps1[:,:,pc_idx].real
    V = eigmaps1[:,:,pc_idx].imag
    Z2 = U+1j*V
    plt.figure()
    plt.quiver(X,Y,U,V,angles='xy')
    plt.gca().invert_yaxis()
    CL.append(Z2)
    plt.xlim(X.min()-1,X.max()+1)
    plt.ylim(Y.min()-1,Y.max()+1)
    plt.title('CL Day ' + str(day_idx))
    
    # get the phase gradients 
    # pm,pd = phase_gradient_complex_multiplication(Z2)
    
    # # plot the phase gradients
    # U = np.cos(pd)
    # V =  np.sin(pd)    
    # plt.figure()
    # plt.quiver(X,Y,U,V,angles='xy')
    # plt.gca().invert_yaxis()
    
    #### LOOK AT PHASE NOISE 
    A = Zproj[:,:,pc_idx]
    B = Zproj1[:,:,pc_idx]
    ampA,ampB,noiseA,noiseB = get_phase_statistics(A,B)
    noise_diff = (np.mean(noiseA)-np.mean(noiseB))/np.mean(noiseA) * 100
    var_stats.append((np.std(ampA)-np.std(ampB))/np.std(ampA) * 100)
    mean_stats.append((np.mean(ampA)-np.mean(ampB))/np.mean(ampA) * 100)
    noise_stats.append(noise_diff)
    
    mean_statsA.append(np.mean(ampA))
    mean_statsB.append(np.mean(ampB))
    var_statsA.append(np.std(ampA))
    var_statsB.append(np.std(ampB))

var_stats = np.array(var_stats)
noise_stats = np.array(noise_stats)
mean_stats = np.array(mean_stats)


# plot example activation
tmp = Zproj[112,:,pc_idx]
plt.figure()
plt.plot(tmp.real)
plt.plot(tmp.imag)
plt.plot(abs(tmp))
plt.legend(('Real','Imag','Abs'))
plt.ylabel('mu wave')
plt.xlabel('Time')
plt.title('Single Trial PC 1 Projection')

plt.figure()
plt.boxplot(noise_stats);plt.hlines(0,0.5,1.5)
plt.title('Phase noise')
plt.ylabel('Open Loop minus Closed loop (%)')
plt.xticks([])

plt.figure()
plt.boxplot(var_stats);plt.hlines(0,0.5,1.5)
plt.title('Trial to Trial Variance')
plt.ylabel('Open Loop minus Closed loop (%)')
plt.xticks([])

plt.figure()
plt.boxplot(mean_stats);plt.hlines(0,0.5,1.5)
plt.title('Mean amplitude')
plt.ylabel('Open Loop minus Closed loop (%)')
plt.xticks([])


print(stats.ttest_1samp(noise_stats,0))
pvalue,boot_stat = bootstrap_test(noise_stats,'mean')
pvalue,boot_stat = bootstrap_test(var_stats,'mean')
pvalue,boot_stat = bootstrap_test(mean_stats,'mean')
print(stats.wilcoxon(noise_stats))
print(stats.wilcoxon(var_stats))
print(stats.wilcoxon(mean_stats))

# plotting
mean_statsA = np.array(mean_statsA)
mean_statsB = np.array(mean_statsB)
var_statsA = np.array(var_statsA)
var_statsB = np.array(var_statsB)
plt.figure()
plt.plot(mean_statsA,var_statsA,'.b')
plt.plot(mean_statsB,var_statsB,'.r')
plt.show()

plt.figure()
plt.plot(mean_stats,var_stats,'.b')
plt.show()


CL = np.array(CL)
OL = np.array(OL)
#filename = 'Eigmaps_layer4Ch14PC2.mat'
filename = 'Eigmaps_' +  str(layer_name) + 'Ch' + str(channel_idx)+ 'PC'+str(pc_idx) + '.mat'
savemat(filename, {"OL": OL, "CL": CL}, long_field_names=True)

#Eigmaps_layer3Ch11PC2 is good

# have to examine the activations of the PCs
# unwrap phase and look at variability in slope
# amplitude modulation
# wave speed


# KEY STEPS
# AMPLITUDE DIFFERENCES BETWEEN THE TWO CONDITIONS IN PC ACTIVITY
# FREQ OF AMPLITUDE FLUCTUATIONS (PAC)
# PHASE NOISE IE VARIABILITY IN PHASE IN PROJECTED ACTIVITY


print((np.mean(ampA)-np.mean(ampB))/np.mean(ampA) * 100)

print(stats.mannwhitneyu(ampA,ampB))

print((np.mean(noiseA)-np.mean(noiseB))/np.mean(noiseA) * 100)

print(stats.mannwhitneyu(noiseA,noiseB))

# make a movie of the PC through time
pc_idx=2
recon = Zproj[10,:,pc_idx][:,None] @ eigvecs[:,pc_idx].T.conj()[None,:]
recon = recon.T
recon = recon.reshape(H,W,-1)

# Transpose so shape = (time, height, width)
frames = np.moveaxis(recon, -1, 0)  # (28, 8, 20)
frames=np.imag(frames)
fig,ax = plt.subplots()
im = ax.imshow(frames[0],cmap='viridis',aspect='auto')

def update(i):
    im.set_array(frames[i])
    ax.set_title(f"Frame {i}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames.shape[0],interval=100,blit=True)
filename1 = 'Eigmaps_'+str(layer_name)+'Ch'+str(channel_idx)+'PC'+str(pc_idx)+'TrialExample_imag.gif'
ani.save(filename1, writer="pillow", fps=5)


#filename1 = 'Eigmaps_layer3Ch11PC2_imag' + '_trialExample.mp4'


#from matplotlib.animation import FFMpegWriter

# Define writer with codec 'libx264'
#writer = FFMpegWriter(fps=5, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
#ani.save("matrix_movie.mp4", writer=writer)


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


# # compute the curl of the phasor field
# Z = np.angle(Z2)
# dphi_dy, dphi_dx = np.gradient(Z,edge_order=2)
# plt.figure()
# plt.quiver(X,Y,dphi_dx,dphi_dy,angles='xy')
# plt.gca().invert_yaxis()
# plt.xlim(X.min()-1,X.max()+1)
# plt.ylim(Y.min()-1,Y.max()+1)

#%% CONTINUATION FROM ABOVE, CONTRASTIVE COMPLEX VALUED PCA

from scipy.linalg import eigh
A = activations_ol
B = activations_cl
# compute covariance matrix
eigvals, eigmaps, Z , VAF,eigvecs,Ca = complex_pca(A,15)
eigvals, eigmaps, Z , VAF,eigvecs,Cb = complex_pca(B,15)

alp = 1e-1
M = Cb + alp * np.eye(Cb.shape[0])

eigvals, eigvecs = eigh(Ca, M)
    
# Sort descending by eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

B, W, H, T = A.shape
n_components=eigvecs.shape[0]
eigmaps = eigvecs.reshape(W,H,n_components)    

# plot phasor maps
for pc_idx in np.arange(10):
        
    H,W = eigmaps.shape[:2]
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    U = eigmaps[:,:,pc_idx].real
    V = eigmaps[:,:,pc_idx].imag
    Z2 = U+1j*V
    plt.figure()
    plt.quiver(X,Y,U,V,angles='xy')
    plt.gca().invert_yaxis()
        


#%% COMMON PCA ON ALL


# PRELIMS
from iAE_utils_models import *
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

if 'model' in locals():
    del model 

model = model_class(ksize,num_classes,input_size,lstm_size).to(device)
model.load_state_dict(torch.load(nn_filename))

# GET THE ACTIVATIONS FROM A CHANNEL LAYER OF INTEREST
layer_name = 'layer3'
channel_idx = 11
batch_size=256
activations_real, activations_imag = get_channel_activations(model, Xval, Yval,
                                    labels_val,device,layer_name,
                                    channel_idx,batch_size)
activations = activations_real + 1j*activations_imag

eigvals, eigmaps, Z , VAF,eigvecs = complex_pca(activations,15)

# PLOT EIGMAPS AS PHASORS
pc_idx=5
H,W = eigmaps.shape[:2]
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
U = eigmaps[:,:,pc_idx].real
V = eigmaps[:,:,pc_idx].imag
plt.figure()
plt.quiver(X,Y,U,V,angles='xy')
plt.xlim(X.min()-1,X.max()+1)
plt.ylim(Y.min()-1,Y.max()+1)

#%% DO PCA TO EXAMINE ACTIVATIONS OF INDIVIDUAL CHANNELS OF A LAYER (v0)

# PRELIMS
from iAE_utils_models import *
torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

# get the CNN architecture model



if 'model' in locals():
    del model 
 
model = model_class(ksize,num_classes,input_size,lstm_size).to(device)
model.load_state_dict(torch.load(nn_filename))


# GET THE ACTIVATIONS FROM A CHANNEL LAYER OF INTEREST
layer_name = 'layer3'
channel_idx = 11
batch_size=256

activations_real, activations_imag = get_channel_activations(model, Xval, Yval,
                                    labels_val,device,layer_name,
                                    channel_idx,batch_size)

activations = activations_real + 1j*activations_imag

# RUN COMPLEX PCA
eigvals, eigmaps, Z , VAF,eigvecs = complex_pca(activations,15)

# PLOT EIGMAPS AS PHASORS
pc_idx=1
H,W = eigmaps.shape[:2]
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
U = eigmaps[:,:,pc_idx].real
V = eigmaps[:,:,pc_idx].imag
plt.figure()
plt.quiver(X,Y,U,V,angles='xy')
plt.xlim(X.min()-1,X.max()+1)
plt.ylim(Y.min()-1,Y.max()+1)

# PLOT EIGMAPS AS IMAGE
ph = np.angle(eigmaps[:,:,pc_idx])
ph = np.sin(ph)
plt.figure()
plt.imshow(ph,vmin=-1, vmax=1)
plt.colorbar()

# EXAMINING THE PHASE VARIANCE BETWEEN OL AND CL
# phase diffusion constant 
var_ol=[]
var_cl=[]
for trial in np.arange(len(labels_val)):   
    tmp = Z[trial,:,pc_idx]
    ph = np.angle(tmp)
    ph = np.unwrap(ph)
    delta_ph = np.diff(ph)
    if labels_val[trial]==0:
        var_ol.append(np.var(delta_ph))
        
    if labels_val[trial]==1:
        var_cl.append(np.var(delta_ph))

var_ol = np.array(var_ol)      
var_cl = np.array(var_cl)      
plt.boxplot((var_ol,var_cl))

# EXAMINING SLOPES, VARIANCE AND ACTIVATIONS
slopes_ol=[]
slopes_cl=[]
var_ol=[]
var_cl=[]
act_ol=[]
act_cl=[]
for trial in np.arange(len(labels_val)):   
    tmp = Z[trial,:,pc_idx]
    amp = np.mean(np.abs(tmp))    
    ph = np.angle(tmp)
    
    #ph = np.mod(ph, 2*np.pi)  # shift to [0, 2π]
    ph = np.unwrap(ph)
    slope, intercept = np.polyfit(np.arange(len(ph)), ph, 1)
    phat = intercept + slope*np.arange(len(ph))
    err = (phat-ph)[:,None]
    err = err.T @ err
    err = np.var(np.diff(ph))

    if labels_val[trial]==0:
        slopes_ol.append(slope)
        var_ol.append(err)
        act_ol.append(amp)
        
    if labels_val[trial]==1:
        slopes_cl.append(slope)
        var_cl.append(err)
        act_cl.append(amp)

slopes_ol = np.array(slopes_ol)      
slopes_cl = np.array(slopes_cl)    
act_ol = np.array(act_ol)      
act_cl = np.array(act_cl)    

var_ol = np.array(var_ol).squeeze()    
var_cl = np.array(var_cl).squeeze()  

print(var_cl.mean())
print(var_ol.mean())


plt.boxplot((var_ol,var_cl))  
plt.figure()
plt.boxplot((slopes_ol,slopes_cl))


slope, intercept = np.polyfit(np.arange(len(ph)), ph, 1)
phat = intercept + slope*np.arange(len(ph))
plt.figure()
plt.plot(ph)
plt.plot(phat,'--')
plt.show

# PLOT PC SINGLE TRIAL ACTIVATIONS AS PHASOR ANIMATION
trial=2;
tmp = Z[trial,:,pc_idx][:,None]
filename = 'trial0_PC_phasor.gif'
ani = plot_1D_phasor_movie(tmp,filename)


# plot raw activations
scores = Z[0,:,pc_idx]
a = np.angle(scores)
a = np.cos(a)
# a = np.abs(scores)
plt.figure()
plt.plot(a)

#### recon single trial
pc_idx=0
z = Z[100,:,pc_idx][:,None]
pc = eigvecs[:,pc_idx][:,None]
Xhat1 = z @ (pc.conj().T)
Xhat1 = np.reshape(Xhat1,(Xhat1.shape[0],H,W))

# filename = 'temp_recon_Layer'
# make_movie(Xhat1,filename)

#x1=Xhat1.real
x1 = np.sin(np.angle(Xhat1))
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
filename = 'temp_recon_Layer.gif'
ani.save(filename, writer="pillow", fps=6)

#### plot phasor
xreal = Xhat1.real
ximag = Xhat1.imag

# xreal = 2 * ((xreal - xreal.min()) / (xreal.max() - xreal.min())) - 1
# ximag = 2 * ((ximag - ximag.min()) / (ximag.max() - ximag.min())) - 1
fig, ax = plt.subplots(figsize=(6, 6))

def update(t):
    plot_phasor_frame_time(xreal, ximag, t, ax)
    #plot_phasor_frame(xreal, ximag, t, ax)
    return []

#ani = animation.FuncAnimation(fig, update, frames=xreal.shape[0], blit=False)
ani = animation.FuncAnimation(fig, update, frames=xreal.shape[0], blit=False)

plt.show()

# save the animation
filename = 'temp_recon_Layer_phasor.gif'
ani.save(filename, writer="pillow", fps=4)

plt.plot(xreal[0,0,:])
plt.plot(ximag[0,0,:])

plt.show()



#### TO plot movie of activations as on the the grid VECTORIZED
z =Z[:,:,pc_idx].T
z = z[:,:,None]
pc = eigvecs[:,pc_idx][:,None]
pc = pc[:,:,None]
Xhat = z @ pc.conj().T

### reshape into grid

### make movie per trial 
# plot VAF
plt.figure()
plt.stem(VAF)

ol_day = np.where(labels_val==0)[0]
cl_day = np.where(labels_val==1)[0]
Z_ol = Z[ol_day,:]
Z_cl = Z[cl_day,:]

a = np.median(np.abs(Z_ol[:,:,pc_idx]),axis=1)
b = np.median(np.abs(Z_cl[:,:,pc_idx]),axis=1)

# looking ar variance within each trial to see if there are differences
var_ol=[]
for i in np.arange(Z_ol.shape[0]):
    tmp = Z_ol[i,:,pc_idx][:,None]
    u = np.mean(tmp,axis=0)
    v = (tmp - u).conj().T @ (tmp-u)
    d = v.real
    var_ol.append(d)
    
var_cl=[]
for i in np.arange(Z_cl.shape[0]):
    tmp = Z_cl[i,:,pc_idx][:,None]
    u = np.mean(tmp,axis=0)
    v = (tmp - u).conj().T @ (tmp-u)
    d = v.real
    var_cl.append(d)

var_ol = np.log(np.array(var_ol).squeeze())
var_cl = np.log(np.array(var_cl).squeeze())
plt.figure()
plt.boxplot((var_ol,var_cl))
plt.xticks((1,2),labels=('OL','CL'))
res=stats.wilcoxon(var_ol,var_cl)
print(res.pvalue)

# look at variance in the activations themselves
var_ol=[]
act_ol = activations[ol_day,:]
for i in np.arange(act_ol.shape[0]):
    tmp = np.array(act_ol[i,:])
    h,w,t=tmp.shape
    tmp = (np.reshape(tmp,(h*w,t))).T
    m = np.mean(tmp,axis=0)[None,:]
    z = tmp-m
    C = ((1/(t-1)) * (z.conj().T) @ z).real
    var_ol.append(lin.diagonal(C))
    
var_cl=[]
act_cl = activations[cl_day,:]
for i in np.arange(act_cl.shape[0]):
    tmp = np.array(act_cl[i,:])
    h,w,t=tmp.shape
    tmp = (np.reshape(tmp,(h*w,t))).T
    m = np.mean(tmp,axis=0)[None,:]
    z = tmp-m
    C = ((1/(t-1)) * (z.conj().T) @ z).real
    var_cl.append(lin.diagonal(C))
    
var_ol = np.array(var_ol)
var_cl = np.array(var_cl)

# condition specific variance described by a mode
dataA = activations[ol_day,:]
dataB = activations[cl_day,:]


# looking at the strength of the activations of OL and CL on each day for PC1
act_str_ol=[]
act_str_cl=[]
for day in np.arange(10)+1:
    #day = 1
    #labels_day = labels_test[np.where(labels_test_days==day)[0]]
    ol_day = np.where(labels_day==0)[0]
    cl_day = np.where(labels_day==1)[0]    
    Z_ol = Z[ol_day,:]
    Z_cl = Z[cl_day,:]
    
    # plt.figure()
    # plt.plot(Z_ol[0,:,pc_idx].real)
    # plt.plot(Z_ol[0,:,pc_idx].imag)
    # plt.plot(np.abs(Z_ol[0,:,0]))
    # plt.show()
    
    
    a = np.mean(np.abs(Z_ol[:,:,pc_idx]),axis=1)
    b = np.mean(np.abs(Z_cl[:,:,pc_idx]),axis=1)
    #plt.boxplot((a,b))
    statistic, p_value = stats.ks_2samp(a,b)
    
    act_str_ol.append(np.mean(a))
    act_str_cl.append(np.mean(b))
    
plt.figure()
plt.plot(act_str_ol)
plt.plot(act_str_cl)
plt.legend(('OL','CL'))
plt.show()

plt.figure()
plt.boxplot((act_str_ol,act_str_cl))

