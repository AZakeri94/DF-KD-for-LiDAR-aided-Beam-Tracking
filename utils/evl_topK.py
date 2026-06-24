    #================================================== Evaluation/Test ==============================
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def evaluate_model(model, val_loader, num_classes, device):
        model.eval()
        total = np.zeros((4,))
        top1_correct = np.zeros((4,))
        top2_correct = np.zeros((4,))
        top3_correct = np.zeros((4,))
        top5_correct = np.zeros((4,))
        val_loss = 0

        with torch.no_grad():
            for lidar_img, beam, label in val_loader:
                lidar_img = torch.swapaxes(lidar_img, 0, 1)
                lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)
                label = torch.cat([beam[..., -1:], label], dim=-1)
                #print(label.shape)
                label = torch.swapaxes(label, 0, 1)
                lidar_img, label = lidar_img.to(device), label.to(device)
                #print(lidar_img.shape)


                h = model.initHidden(lidar_img.shape[1])
                if h is not None: h = h.to(device)
                logits, features = model(lidar_img, h)

                logits = logits[-4:]
                #print(logits.shape)
                label = label[-4:]
                #print(label.shape)
                val_loss += F.cross_entropy(logits.reshape(-1, num_classes), label.flatten(), reduction='sum').item()

                total += torch.sum(label != -100, dim=-1).cpu().numpy()
                pred = torch.argmax(logits, dim=-1)
                top1_correct += torch.sum(pred == label, dim=-1).cpu().numpy()

                _, topk = torch.topk(logits, 5, dim=-1)
                topk = topk.cpu().numpy()
                label_np = label.cpu().numpy()

                for i in range(label_np.shape[0]):
                    for j in range(label_np.shape[1]):
                        if label_np[i, j] == -100:
                            continue
                        top2_correct[i] += np.isin(label_np[i, j], topk[i, j, :2])
                        top3_correct[i] += np.isin(label_np[i, j], topk[i, j, :3])
                        top5_correct[i] += np.isin(label_np[i, j], topk[i, j, :5])

        val_loss /= total.sum()
        val_top1 = top1_correct / total
        val_top2 = top2_correct / total
        val_top3 = top3_correct / total
        val_top5 = top5_correct / total

        #print(f"Val_Loss: {val_loss:.4f}")
        #print("Top-k Accuracies (per step):")
        print("Top1=", val_top1)
        #print("Top-2:", val_top2)
        #print("Top-3:", val_top3)
        #print("Top5=", val_top5)
        return val_top1, val_top5, val_loss

    #=============================================== END evaluation ---