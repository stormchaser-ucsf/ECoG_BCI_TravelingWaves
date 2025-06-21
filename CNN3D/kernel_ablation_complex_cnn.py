
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==== USER DEFINED IMPORTS ====
# from your_model import ComplexAutoencoder3D, classifier
# from your_dataset import YourDatasetClass

# ==== DEVICE ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== LOAD MODEL ====
# model = ComplexAutoencoder3D(...)
# model.load_state_dict(torch.load('your_trained_model.pth'))
# model = model.to(device)
# model.eval()

# ==== LOAD DATASET ====
# dataset = YourDatasetClass(...)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# ==== ABLATION HOOK ====
def make_channel_ablation_hook(indices):
    def hook(module, input, output):
        output[:, indices] = 0
        return output
    return hook

# ==== FIND CONV LAYERS ====
def find_conv_layers_encoder_real(model):
    conv_layers_real = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ('real' in name):
            conv_layers_real[name] = module
    return conv_layers_real

def find_conv_layers_encoder_imag(model):
    conv_layers_imag = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ( 'imag' in name):
            conv_layers_imag[name] = module
    return conv_layers_imag



def find_conv_layers_decoder(model):
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d) and ('real' in name or 'imag' in name):
            conv_layers[name] = module
    return conv_layers


# ==== ABLATION LOOP ====
def ablate_kernels(model, dataloader):
    conv_layers_real = find_conv_layers_encoder_real(model)
    conv_layers_imag = find_conv_layers_encoder_imag(model)
    a=iter(conv_layers_real.items())
    b=iter(conv_layers_imag.items())
    results = {}
    
    for i in np.arange(len(conv_layers_real)):
        real_layer_name,real_layer = next(a)
        imag_layer_name,imag_layer = next(b)        
        lay_name = 'conv' + str(i+1)
        results[lay_name] = {}
        num_kernels = imag_layer.out_channels
        
        for k in range(num_kernels):
            print(f"Ablating complex {lay_name}[{k}]...")
            hook_real = real_layer.register_forward_hook(make_channel_ablation_hook([k]))
            hook_imag = imag_layer.register_forward_hook(make_channel_ablation_hook([k]))
            
            

    for layer_name, layer in conv_layers_real.items():
        #print(layer_name, layer.out_channels)
        
        num_kernels = layer.out_channels
        results[layer_name] = {}

        for k in range(num_kernels):
            print(f"Ablating {layer_name}[{k}]...")
            hook = layer.register_forward_hook(make_channel_ablation_hook([k]))

            total_loss = 0
            model.eval()
            with torch.no_grad():
                for real, imag, _ in dataloader:
                    real, imag = real.to(device), imag.to(device)
                    out_r, out_i, *_ = model(real, imag)
                    loss = F.mse_loss(out_r, real) + F.mse_loss(out_i, imag)
                    total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            results[layer_name][k] = avg_loss
            print(f"{layer_name}[{k}] -> Recon Loss: {avg_loss:.4f}")

            hook.remove()
    
    return results

# ==== RUN ABLATION ====
# results = ablate_kernels(model, dataloader)

# ==== SAVE RESULTS ====
# import json
# with open('ablation_results.json', 'w') as f:
#     json.dump(results, f, indent=2)

# ==== OPTIONAL: PLOT ====
# import matplotlib.pyplot as plt
# for layer, losses in results.items():
#     plt.figure()
#     plt.bar(losses.keys(), losses.values())
#     plt.title(f"Ablation Impact - {layer}")
#     plt.xlabel('Kernel Index')
#     plt.ylabel('Reconstruction Loss')
#     plt.show()
