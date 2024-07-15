import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.ConvertImageDtype(torch.float32)  # Convert to torch.float32
])


def create_loaders(data_dir = "data", batch_size = 32, num_workers = 4, transform = transform):
    """ Main function to load data and create data loaders.
    Args:
        data_dir (str): Directory with data
        batch_size (int): Number of images per batch
        num_workers (int): Number of workers to use for loading data
        transform (torchvision.transforms): Image transformations
    Returns:
        DataLoader: Training dataloader
        DataLoader: Validation dataloader
        DataLoader: Test dataloader"""
    
    # Create dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Calculate sizes for each set
    total_size = len(dataset)
    test_size = int(0.15 * total_size)
    train_val_size = total_size - test_size
    val_size = int(0.20 * train_val_size)
    train_size = train_val_size - val_size
    
    # Split the dataset
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


