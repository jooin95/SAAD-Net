import torch
import json

with open('result_v2_imp_FIXED_full_lr5e5/history.json', 'r') as f:
    history = json.load(f)

best_epoch_idx = 39  
class_acc = history['class_acc_history'][best_epoch_idx]
print("Per-class accuracy at best epoch:")
for cls_idx, acc in class_acc.items():
    print(f"  Class {cls_idx}: {acc:.2f}%")

ckpt = torch.load('result_v2_imp_FIXED_full_lr5e5/best_model.pth', map_location='cpu')
state = ckpt['model_state_dict']

# LAA alpha
alpha = state['model.anomaly_module.alpha'].item()
print(f"LAA alpha: {alpha:.4f}")

# S-SAA gamma  
gamma = state['model.saa.gamma'].item()
print(f"S-SAA gamma: {gamma:.4f}")