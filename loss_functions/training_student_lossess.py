# different lossess for the training of the student 
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% ========= Student KD Loss ==========
def distillation_loss( fake_inputs, teacher, student, T=2.0 ):
    batch_size = fake_inputs.size(1) #print(batch_size)

    teacher.eval()
    with torch.no_grad():
        h = teacher.initHidden(batch_size).to(device)
        t_logits, _ = teacher(fake_inputs, h)                    # logits: all 11 steps [11, B, 64]

    h = student.initHidden(batch_size).to(device)
    s_logits, _ = student(fake_inputs, h)                        # logits: all 11 steps

    s_logits = s_logits.reshape(-1, s_logits.size(-1)) # [11*B, 64] [-4:]
    t_logits = t_logits.reshape(-1, t_logits.size(-1))

    kd_loss = F.kl_div(                # this is an important cost function here!
        F.log_softmax( s_logits / T, dim=-1 ),
        F.softmax( t_logits / T, dim=-1 ),
        reduction='batchmean'
    ) * (T * T)

    loss = kd_loss

    #loss = F.mse_loss( t_logits, s_logits ) #print(kd_loss_MSE) # 2


    return loss
