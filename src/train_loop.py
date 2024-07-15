import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, activation_fn_name, device, patience=5):
    # Move model to the specified device
    model.to(device)
    
    # Create directories for storing artifacts
    artifacts_dir = os.path.join('artifacts', activation_fn_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    log_file = os.path.join(artifacts_dir, f'{activation_fn_name}.log')
    best_weights_file = os.path.join(artifacts_dir, 'best_weights.pth')
    top_metrics_file = os.path.join(artifacts_dir, 'top_metrics.csv')
    
    best_loss = float('inf')
    patience_counter = 0
    top_metrics = []
    
    with open(log_file, 'w') as log:
        log.write('Epoch,Train Loss,Val Loss,Train Acc,Val Acc,Train F1,Val F1,Train AUC,Val AUC,Epoch Time\n')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_losses = []
            all_labels = []
            all_preds = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            
            train_loss = np.mean(train_losses)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            train_auc = roc_auc_score(all_labels, outputs.softmax(dim=1).cpu().numpy(), multi_class='ovr')
            
            # Validation phase
            model.eval()
            val_losses = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            val_auc = roc_auc_score(all_labels, outputs.softmax(dim=1).cpu().numpy(), multi_class='ovr')
            
            epoch_time = time.time() - start_time

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Epoch Time: {epoch_time:.2f}s')
            log.write(f'{epoch+1},{train_loss},{val_loss},{train_acc},{val_acc},{train_f1},{val_f1},{train_auc},{val_auc},{epoch_time}\n')
            
            # Check for best validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_weights_file)
            else:
                patience_counter += 1
            
            # Save top metrics
            top_metrics.append((train_acc, val_acc, train_f1, val_f1, train_auc, val_auc))
            top_metrics.sort(key=lambda x: max(x), reverse=True)
            top_metrics = top_metrics[:5]
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Save top metrics to CSV
    df = pd.DataFrame(top_metrics, columns=['Train Acc', 'Val Acc', 'Train F1', 'Val F1', 'Train AUC', 'Val AUC'])
    df.to_csv(top_metrics_file, index=False)
    
    return model


def experiment(activation_fn, activation_fn_name, num_epochs, patience):
    resnet_model = resnet34(activation_fn=activation_fn, num_classes=100)
    model = train_model(resnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, 
                        activation_fn_name=activation_fn_name, device=device, patience=patience)
    return model 

# Example usage:
# Assuming you have train_loader and val_loader already defined
# model = resnet34(activation_fn=nn.PReLU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, activation_fn_name='PReLU', device=device, patience=5)
