import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, os
import numpy as np
from torch.utils.data import DataLoader

from model import GruModel, student_model, gru_cnn_teacher
from utils.evl_topK import evaluate_model
from utils.data_feed import DataFeed
from torch.utils.data import DataLoader
from loss_functions.data_generator_lossess import loss_generator
from loss_functions.training_student_lossess import distillation_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

#%% ========= Generator Model ==========
input_dim = 500 # input random data length; to define this globally
seq_len = 8

#'''  # 1h
class generator_model(nn.Module):
    def __init__( self, output_dim=seq_len * 216, hidden_dim=128 ): # also requires input-size define above
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, seq_len, 216)
#'''  

       
''' #2h
class generator_model(nn.Module):
    def __init__(self, output_dim=seq_len * 216, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, seq_len, 216)
#'''


#%% GENERATOR: the training function

def train_generator(generator, teacher, num_epochs=500, batch_size=64, lr=1e-3, metadata=None):
    
    teacher.train()
    teacher.dropout.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    generator.train()
    optimizer = optim.Adam( generator.parameters(), lr=lr,  weight_decay=1e-4 )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    if metadata is None:
        metadata = torch.load('./saved_models/metadata.pt')

    for epoch in range(num_epochs):
        z = torch.randn(batch_size, input_dim).to(device)
        fake_inputs = generator(z).permute(1, 0, 2) # outputsize: [8, BS, #feat. 216]
        fake_inputs = torch.cat([fake_inputs, torch.zeros_like(fake_inputs[:3, ...])], dim=0) # zero-padd with length V=3

        loss = loss_generator(fake_inputs, teacher, batch_size, metadata=metadata)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print( f"Epoch {epoch}: Loss = {loss.item():.4f}")




#%% STUDENT: --- training function ---
def train_student_fixed_data(fake_inputs, teacher, student, generator, num_epochs=500, batch_size=64, lr=1e-3,
                              val_loader=None, patience=10, num_classes=64):
    generator.eval()
    teacher.eval()


    optimizer = optim.Adam( student.parameters(),
                            lr=lr, weight_decay=1e-4 )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


    total_samples = fake_inputs.size(1) # = num_samples
    num_batches = total_samples // batch_size

    best_val_acc = -1.0; best_val_loss = 1e10
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        student.train()
        generator.eval()
        teacher.eval()

        epoch_loss = 0.0
        perm = torch.randperm(total_samples)  # shuffle
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_inputs = fake_inputs[:, idx, :].to(device) # size[11, 64, 216]

            loss = distillation_loss(batch_inputs, teacher, student)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()),  1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
      
        #'''
        if val_loader is not None:
            _, _, val_loss = evaluate_model(student, val_loader, num_classes, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
                print("  --> best model saved")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break
        #'''        
             

    if best_state is not None:
        student.load_state_dict(best_state)

if __name__ == "__main__":
    seed = 692

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    val_loader = DataLoader(DataFeed('./utils/val_seqs.csv', seq_len, init_shuffle=False), batch_size=128, shuffle=False)
    test_loader  = DataLoader(DataFeed('./utils/test_seqs.csv',  seq_len), batch_size=128, shuffle=False)
   
    num_classes = 64
    
    num_samples = 5000
    z_fixed = torch.randn(num_samples, input_dim).to(device)

    teacher = gru_cnn_teacher(num_classes).to(device)
    teacher.load_state_dict(torch.load(f'{save_dir}/teacher.pth'))

    student = gru_student(num_classes).to(device)

    generator = LiDARDiffusionGenerator().to(device)
    train_generator(generator, teacher)
    generator.eval()
    with torch.no_grad():
        fake_inputs_lidar = generator(z_fixed).permute(1, 0, 2)
    fake_inputs_fixed = torch.cat([fake_inputs_lidar, torch.zeros_like(fake_inputs_lidar[:3, ...])], dim=0)

    train_student_fixed_data(fake_inputs_fixed, teacher, student, generator,
                             val_loader=val_loader, num_classes=num_classes)

    #print("\n--- Evaluation TEACHER ---")
    t_top1, t_top5, _ = evaluate_model(teacher, test_loader, num_classes, device)

    print("\n--- Evaluation DF-KD STUDENT ---")
    s_top1, s_top5, _ = evaluate_model(student, test_loader, num_classes, device)
