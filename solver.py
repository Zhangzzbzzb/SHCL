import torch
import numpy as np
import os
from tqdm import tqdm
from model.SHCLVae import SHCLVae
from data_factory.data_loader import get_loader_segment
import csv
import time

        
def kl_divergence(mu_p, log_var_p, mu_n, log_var_n):
    """
    Compute KL divergence between two Gaussian distributions.
    
    Args:
        mu_p, log_var_p: Mean and log variance of the first distribution
        mu_n, log_var_n: Mean and log variance of the second distribution
    
    Returns:
        KL divergence tensor
    """
    return -0.5 * torch.sum(1 + log_var_p - log_var_n - (mu_p - mu_n)**2 / torch.exp(log_var_n) - torch.exp(log_var_p) / torch.exp(log_var_n), dim=-1)


def compute_loss(data_dict):
    """
    Compute total loss combining KL divergence and reconstruction loss.
    
    Args:
        data_dict: Dictionary containing 'single' and 'pair' keys with model outputs
    
    Returns:
        Total loss tensor of shape (B, L)
    """
    loss_total = 0

    for key in ["single", "pair"]:
        xp, xn, mu_p, log_var_p, mu_n, log_var_n = data_dict[key]
        B, L, _ = xp.shape  # B: batch_size, L: time steps

        # Compute KL divergence for each time step
        kl_loss = kl_divergence(
            mu_p.view(B * L, -1), 
            log_var_p.view(B * L, -1), 
            mu_n.view(B * L, -1), 
            log_var_n.view(B * L, -1)
        )
        
        # Compute reconstruction loss (MSE over latent dimension)
        reconstruction_loss = torch.mean((xp - xn) ** 2, dim=-1)

        # Reshape KL loss back to (B, L)
        kl_loss = kl_loss.view(B, L)
        
        # Combine KL divergence and reconstruction loss
        loss = kl_loss + reconstruction_loss

        # Accumulate losses for 'single' and 'pair'
        loss_total += loss
    
    return loss_total

def anomaly_detection_loss(mu_p, mu_n):
    """
    Compute anomaly detection loss based on mean squared difference.
    
    Args:
        mu_p, mu_n: Mean vectors of positive and negative samples
    
    Returns:
        Mean squared difference for anomaly detection
    """
    return torch.mean((mu_p - mu_n) ** 2, dim=-1)


class Solver(object):
    """
    Main solver class for SHCL-VAE training, validation and testing.
    Handles the complete pipeline from data loading to model evaluation.
    """
    DEFAULTS = {}

    def __init__(self, config):
        

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        """
        Initialize SHCL-VAE model and optimizer.
        Prints total parameter count for reference.
        """
        self.model = SHCLVae(win_size=self.win_size, seq_size=self.seq_size, c_in=self.input_c, c_out=self.output_c, d_model=self.d_model, e_layers=self.e_layers, fr=self.fr, tr=self.tr, dev=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')

    def vali(self, vali_loader):
        """
        Validation loop to evaluate model performance on validation set.
        
        Args:
            vali_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()

        loss_list = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)

                data_dict = self.model(input)

                loss = compute_loss(data_dict)
                loss = loss.mean()

                # Add loss as tensor to list
                loss_list.append(loss.detach().cpu())


        return np.average(loss_list)

    def train(self):
        """
        Main training loop with validation and model checkpointing.
        Trains the model for specified number of epochs and saves best checkpoint.
        """

        print("======================TRAIN MODE======================")

        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in tqdm(range(self.num_epochs)):
            loss_list = []
            epoch_time = time.time()
            self.model.train()
            with tqdm(total=train_steps) as pbar:
                for i, (input_data, labels) in enumerate(self.train_loader):

                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)

                    data_dict = self.model(input)

                    loss = compute_loss(data_dict)
                    loss = loss.mean()

                    # Add loss tensor to list for averaging
                    loss_list.append(loss.detach().cpu())
                    pbar.update(1)

                    loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)

            vali_loss = self.vali(self.vali_loader)

            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
            print(
                "Epoch: {0}, Cost time: {1:.3f}s, Steps: {1} | Train Loss: {2:.20f} Vali Loss: {3:.20f} ".format(
                    epoch + 1, time.time() - epoch_time, train_steps, train_loss, vali_loss))

    def test(self):
        """
        Test the trained model and evaluate anomaly detection performance.
        
        1. Finds optimal threshold using threshold loader
        2. Evaluates on test set with standard and adjusted metrics
        3. Saves results to CSV file
        
        Returns:
            Tuple of (accuracy, precision, recall, f_score)
        """
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        # (1) Find the optimal threshold using threshold data
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)

                data_dict = self.model(input)
                loss = compute_loss(data_dict)

                metric = torch.softmax((loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (2) Evaluation on the test set
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)

                data_dict = self.model(input)
                loss = compute_loss(data_dict)

                metric = torch.softmax((loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        

        # Detection adjustment: see https://github.com/thuml/Anomaly-Transformer/issues/14 for details
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
    
        pa_accuracy = accuracy_score(gt, pred)
        pa_precision, pa_recall, pa_f_score, pa_support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Adjusted_Accuracy : {:0.4f}, Adjusted_Precision : {:0.4f}, Adjusted_Recall : {:0.4f}, Adjusted_F-score : {:0.4f} ".format(
                pa_accuracy, pa_precision,
                pa_recall, pa_f_score))
        
        # Save results to CSV
        results = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "describe": self.describe,
            "dataset": self.dataset,
            "win_size": self.win_size,
            "epoch": self.num_epochs,
            "anormly_ratio": self.anormly_ratio,
            "threshold": thresh,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "adjusted_accuracy": pa_accuracy,
            "adjusted_precision": pa_precision,
            "adjusted_recall": pa_recall,
            "adjusted_f_score": pa_f_score
        }
        csv_path = f"result/results.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        return accuracy, precision, recall, f_score


