## similar to github.com/Michaelvll/DeepCCA main
import torch
import torch.nn as nn
import numpy as np
from linear_gcca import linear_gcca
from synth_data import create_synthData
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from models import DeepGCCA
from models_kan import DeepGCCA
# from utils import load_data, svm_classify
import time
import logging
import pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver():
    def __init__(self, model, linear_gcca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = model # nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.7)
        self.device = device

        if linear_gcca is not None:
            self.linear_gcca = linear_gcca()
        else:
            self.linear_gcca = None

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x_list, vx_list=None, tx_list=None, checkpoint='checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        epoch_losses = []
        x_list = [x.to(device) for x in x_list]

        data_size = x_list[0].size(0)

        if vx_list is not None :
            best_val_loss = 0
            vx_list = [vx.to(self.device) for vx in vx_list]

        if tx_list is not None :
            tx_list = [tx.t0(self.device) for tx in tx_list]


        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x = [x[batch_idx, :] for x in x_list]
                #print("batch_x
                # :", type(batch_x))
                output = self.model(batch_x)
                loss = self.loss(output)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            train_loss = np.mean(train_losses)
            epoch_losses.append(train_loss)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx_list is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx_list)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_gcca
        if self.linear_gcca is not None:
            _, outputs_list = self._get_outputs(x_list)
            self.train_linear_gcca(outputs_list)

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx_list is not None:
            loss = self.test(vx_list)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx_list is not None:
            loss = self.test(tx_list)
            self.logger.info('loss on test data: {:.4f}'.format(loss))
        loss_df = pd.DataFrame({'epoch': list(range(1, self.epoch_num + 1)), 'loss': epoch_losses})
        loss_df.to_csv("training_loss_curve.csv", index=False)
        plt.figure()
        plt.plot(loss_df['epoch'], loss_df['loss'])
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("GCCA Training Loss Curve")
        plt.grid(True)
        plt.savefig("training_loss_curve.png")
        plt.close()

    def test(self, x_list, use_linear_gcca=False):
        with torch.no_grad():
            losses, outputs_list = self._get_outputs(x_list)

            if use_linear_gcca:
                if self.linear_gcca is None:
                    raise ValueError("Linear GCCA not initialized.")
                print("Linear CCA started!")
                outputs_list = self.linear_gcca.test(outputs_list)

            return np.mean(losses), outputs_list  # ✅ 总是返回两个值


    def train_linear_gcca(self, x_list):
        self.linear_gcca.fit(x_list, self.outdim_size)

    def _get_outputs(self, x_list):
        with torch.no_grad():
            self.model.eval()
            data_size = x_list[0].size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs_list = []
            for batch_idx in batch_idxs:
                batch_x = [x[batch_idx, :].to(self.device) for x in x_list]
                outputs = self.model(batch_x)
                outputs_list.append([o_j.clone().detach() for o_j in outputs])
                loss = self.loss(outputs)
                losses.append(loss.item())
        outputs_final = []
        for i in range(len(x_list)):
            view = []
            for j in range(len(outputs_list)):
                view.append(outputs_list[j][i].clone().detach())
            view = torch.cat(view, dim=0)
            outputs_final.append(view)
        return losses, outputs_final

    def save(self, name):
        torch.save(self.model, name)

if __name__ == '__main__':
    ############
    # Parameters Section

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the final learned features
    save_name = './DGCCA.model'

    # the parameters for training the network
    learning_rate = 0.002
    epoch_num = 250
    batch_size = 664

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False


    apply_linear_gcca = True
    # end of parameters section
    ############

    protein = torch.tensor(pd.read_csv("../data/exp.csv").values, dtype=torch.float64)
    clinical = torch.tensor(pd.read_csv("../data/clinical.csv").values, dtype=torch.float64)
    dna = torch.tensor(pd.read_csv("../data/meth.csv").values, dtype=torch.float64)
    rna = torch.tensor(pd.read_csv("../data/miRNA.csv").values, dtype=torch.float64)

    views = [dna, rna, protein, clinical]
    #views = [dna, rna, protein]

    outdim_size = 128

    layer_sizes_list = []
    for view in views:
        in_dim = view.shape[1]
        layer_sizes = [256, 256, outdim_size]
        layer_sizes_list.append(layer_sizes)

    # size of the input for view 1 and view 2
    input_shape_list = [view.shape[-1] for view in views]
    
    # Building, training, and producing the new features by DCCA
    print("layer_sizes_list:", layer_sizes_list)
    print("input_shape_list:", input_shape_list)
    model = DeepGCCA(layer_sizes_list, input_shape_list, outdim_size,
                             use_all_singular_values, device=device).double()
    if apply_linear_gcca:
        l_gcca = linear_gcca
    else:
        l_gcca = None

    solver = Solver(model, l_gcca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=device)

    solver.fit(views, checkpoint=save_name)
    _, outputs = solver.test(views, use_linear_gcca=True)

    z = sum(outputs) / len(outputs)  # or torch.cat(outputs, dim=1)
    df_z = pd.DataFrame(z.cpu().numpy())
    df_z.to_csv("fused_multiview_features.csv", index=False)
