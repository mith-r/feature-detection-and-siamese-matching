import argparse
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from model import SiameseNetwork
from dataset import FeatureMatchingDataset
from loss import ContrastiveLoss
from utils import threshold_sigmoid, threshold_contrastive_loss, visualize_predictions
from tqdm import tqdm

# Hyper Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50


def train(args):
    """
    Train the Siamese network
    
    Args:
        args: Command line arguments
    """
    # Define transformations resize to 256x256
    import torchvision.transforms as transforms
    default_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="train", transform=default_transform)
    print(f"Loaded {len(train_dataset)} training pairs.")

    val_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="test", transform=default_transform)
    print(f"Loaded {len(val_dataset)} validation pairs.")

    # Create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=4,
                                              persistent_workers=True,
                                              pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=4,
                                             persistent_workers=True,
                                             pin_memory=True,
    )
    
    # Initialize model
    siamese_net = SiameseNetwork(args.contra_loss)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    siamese_net = siamese_net.to(device)

    
    # Define loss function
    if args.contra_loss:
        criterion = ContrastiveLoss(margin=args.margin)
        print(f"Using Contrastive Loss with margin={args.margin}")
    else:
        criterion = torch.nn.BCELoss()
        print("Using Binary Cross Entropy Loss")
    
    # Define optimizer
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize lists to track metrics
    train_losses = []
    best_acc = 0.0
    
    # Train the model
    num_epochs = args.epochs
    print(f"Starting training for {num_epochs} epochs...")
    
    # ======================================================================
    # TODO: Implement the training loop
    # Your implementation should:
    # 1. Loop through all epochs
    for epoch in range(num_epochs):
    # 2. For each epoch, iterate through the batches in train_loader
        siamese_net.train()
        epoch_loss = 0.0
        num_batches = 0
        for img1, img2, labels in tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}"):

    # 3. For each batch:
            labels = labels.view(-1, 1).float()

    #    a. Move the data to the appropriate device (CPU/CUDA)
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

    #    b. Zero the parameter gradients using optimizer.zero_grad()
            optimizer.zero_grad()

    #    c. Perform a forward pass through the network
    #    d. Compute the loss (different for contrastive and BCE loss)
            if args.contra_loss:
                out1, out2 = siamese_net(img1, img2)
                loss = criterion(out1, out2, labels)
            else:
                output = siamese_net(img1, img2)
                loss = criterion(output, labels)
  
    #    e. Perform backpropagation using loss.backward()
            loss.backward()

    #    f. Update the model parameters using optimizer.step()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches+= 1

    # 4. Track and print statistics (loss) for each epoch
    # 5. Periodically evaluate the model using the evaluate function
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} loss = {avg_loss:.4f}")
        scheduler.step()
        if (epoch+1) % args.eval_freq == 0:
            acc = evaluate(args, "validation", val_loader, siamese_net, visualize=False)
            if acc > best_acc:
                best_acc = acc
                torch.save(siamese_net.state_dict(), args.model_file)
                print(f"  New best validation accuracy {acc:.2f}% — saved checkpoint to {args.model_file}")

    #
    # Make sure to handle both contrastive loss and BCE loss cases appropriately
    # ======================================================================
    
    # YOUR CODE HERE
    
    # ======================================================================
    # END OF TODO
    # ======================================================================
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Reload the best checkpoint so the returned model is the best one seen during training
    if best_acc > 0.0:
        siamese_net.load_state_dict(torch.load(args.model_file))
        print(f"Best validation accuracy: {best_acc:.2f}% (loaded from {args.model_file})")

    return siamese_net


def evaluate(args, split, data_loader, siamese_net, visualize=False):
    """
    Evaluate the Siamese network
    
    Args:
        args: Command line arguments
        split: Data split ('training' or 'testing')
        data_loader: DataLoader for the split
        siamese_net: Trained Siamese network
        visualize: Whether to visualize predictions
    """
    # Set model to evaluation mode
    siamese_net.eval()
    
    correct = 0.0
    total = 0.0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    all_preds = []
    all_labels = []
    sample_imgs1 = []
    sample_imgs2 = []
    
    with torch.no_grad():
        for img1_set, img2_set, labels in data_loader:
            labels = labels.view(-1, 1).float()
            
            device = next(siamese_net.parameters()).device
            img1_set = img1_set.to(device)
            img2_set = img2_set.to(device)
            labels = labels.to(device)

            
            # Forward pass
            if args.contra_loss:
                output1, output2 = siamese_net(img1_set, img2_set)
                output_labels = threshold_contrastive_loss(output1, output2, args.margin)
            else:
                output_labels_prob = siamese_net(img1_set, img2_set)
                output_labels = threshold_sigmoid(output_labels_prob)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (output_labels == labels).sum().item()

            tp += ((output_labels == 1) & (labels == 1)).sum().item()
            fp += ((output_labels == 1) & (labels == 0)).sum().item()
            fn += ((output_labels == 0) & (labels == 1)).sum().item()
            tn += ((output_labels == 0) & (labels == 0)).sum().item()
            
            # Store predictions for visualization
            if visualize and len(sample_imgs1) < 5:
                # Store a few samples for visualization
                for i in range(min(5, len(labels))):
                    if len(sample_imgs1) < 5:
                        sample_imgs1.append(img1_set[i])
                        sample_imgs2.append(img2_set[i])
                        all_labels.append(labels[i])
                        all_preds.append(output_labels[i])
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f'Accuracy on the {total} {split} images: {accuracy:.2f}%')
    print(f'  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
    print(f'  TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}')
    
    # Visualize some predictions
    if visualize and sample_imgs1:
        visualize_predictions(
            torch.stack(sample_imgs1),
            torch.stack(sample_imgs2),
            torch.stack(all_labels),
            torch.stack(all_preds)
        )
    
    # Return model to training mode
    siamese_net.train()
    
    return accuracy


def test(args, siamese_net=None):
    """
    Test the Siamese network on the test set
    
    Args:
        args: Command line arguments
        siamese_net: Trained Siamese network (if None, load from file)
    """
    # Import transforms here to avoid circular imports
    import torchvision.transforms as transforms
    
    # Load model if not provided
    if siamese_net is None:
        siamese_net = SiameseNetwork(args.contra_loss)
        siamese_net.load_state_dict(torch.load(args.model_file))
        print(f"Loaded model from {args.model_file}")
    
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    siamese_net = siamese_net.to(device)

    
    # Define transformations
    default_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test dataset
    test_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="test", transform=default_transform)
    print(f"Loaded {len(test_dataset)} testing pairs.")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Evaluate on test set
    test_acc = evaluate(args, "testing", test_loader, siamese_net, visualize=True)
    
    return test_acc


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Siamese Network for Feature Matching')
    parser.add_argument('--action', type=str, choices=['train', 'test', 'train_test'], 
                        default='train_test', help='Action to perform')
    parser.add_argument('--data_dir', type=str, default='./', 
                        help='Directory containing training images')
    parser.add_argument('--train_file', type=str, default='./groundtruth.json', 
                        help='File containing training pairs') 
    parser.add_argument('--model_file', type=str, default='siamese_model.pth', 
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--margin', type=float, default=1.0, 
                        help='Margin for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='Use CUDA if available')
    parser.add_argument('--contra_loss', action='store_true', default=False, 
                        help='Use contrastive loss instead of BCE')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Frequency of evaluation during training')
    
    args = parser.parse_args()
    
    print(f"Running with arguments: {args}")
    
    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    elif args.action == 'train_test':
        siamese_net = train(args)
        test(args, siamese_net)


if __name__ == '__main__':
    main()