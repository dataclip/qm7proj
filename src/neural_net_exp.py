import copy
from timeit import default_timer as timer

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             )
from torch import (nn,
                   optim,
                   )
import neural_net
import neural_preprocess

from utils import (
    weights_init_uniform_rule,
    seed_torch,
)

sns.set_style("whitegrid")
plt.style.use('ggplot')
sns.set(font_scale=1.5)
sns.set_palette("bright")
'''
Import saved npy data
'''
X = np.load('../data/coulomb_matrix.npy')
y = np.load('../data/atomization_energies.npy')
P = np.load('../data/folds.npy')

'''
Randomize coulomb matrix and expand according to Rupp 2012
'''
I = neural_preprocess.Input(X)
X_in = I.forward(X)
x_inp_shape = X_in.shape[1]
X_tensor = torch.from_numpy(X_in)
y_tensor = torch.from_numpy(y)
y_tensor = y_tensor.reshape(-1, 1)

SEED = 234
seed_torch(SEED)

''' call model '''

mlp_model = neural_net.MLPNet(x_inp_shape)

if torch.cuda.is_available():
    mlp_model.cuda()

'''set variables '''
learning_rate = 0.00015
batch_size = 25
epochs = 1000

mlp_model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
init_state = copy.deepcopy(mlp_model.state_dict())
init_state_opt = copy.deepcopy(optimizer.state_dict())


def train_reg(model, epochs, bs=batch_size):

    final_loss_df = pd.DataFrame()
    for split in range(0, 5):

        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)

        ptrain = P[list(range(0, split)) + list(range(split + 1, 5))].flatten()
        ptest = P[split]

        x_train_fold = X_tensor[ptrain]
        y_train_fold = y_tensor[ptrain]
        x_test_fold = X_tensor[ptest]
        y_test_fold = y_tensor[ptest]

        loss_fn = nn.MSELoss()

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        test = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=bs, shuffle=False)

        print(f'Fold {split + 1}')

        loss_values = {
            'train': [],
            'test': []
        }

        for epoch in range(epochs):
            start = timer()
            model.train()
            avg_training_loss = 0.

            for x_batch, y_batch in tqdm.tqdm(train_loader, disable=True):
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_training_loss += loss.item() / len(train_loader)
            loss_values['train'].append(avg_training_loss)

            model.eval()
            avg_test_loss = 0.
            for i, (x_batch, y_batch) in enumerate(test_loader):
                y_pred = model(x_batch).detach()
                avg_test_loss += loss_fn(y_pred, y_batch).item() / len(test_loader)

            loss_values['test'].append(avg_test_loss)

            end = timer()
            elapsed_time = end - start

            if epoch % 8 == 1:
                print('Epoch {}/{} \t loss={:.4f} \t test_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epochs, avg_training_loss, avg_test_loss, elapsed_time))

        loss_df = pd.DataFrame.from_dict(loss_values)
        final_loss_df = final_loss_df.append(loss_df)
        final_loss_df['epoch'] = final_loss_df.index
        final_loss_df = final_loss_df.groupby('epoch').mean().reset_index()
    print('average fold losses: \n', final_loss_df)
    plt.plot(final_loss_df.epoch, final_loss_df.train, label='train')
    plt.plot(final_loss_df.epoch, final_loss_df.test, label='test')
    plt.legend()
    plt.savefig('train_test_plot.jpg')


if __name__ == "__main__":
    print('Input shape:', X_in.shape)
    print('X_tensor shape, mean, std. dev:', X_tensor.shape, X_tensor.mean(), X_tensor.std())
    print('y_tensor shape, mean, std.dev:', y_tensor.shape, y_tensor.mean(), y_tensor.std())

    train_reg(mlp_model, epochs)
    y_pred = mlp_model(X_tensor).detach().numpy()
    y_org = (y.reshape(-1, 1))

    joblib.dump(mlp_model, '../models/neural_net.pkl')
    torch.save(mlp_model.state_dict(), '../models/torch_mlp_model.pt')

    rmse = np.sqrt(mean_squared_error(y_org, y_pred))
    mae = mean_absolute_error(y_org, y_pred)

    metrics = [rmse, mae]
    df = pd.DataFrame(metrics, index=['RMSE', 'MAE'],
                      columns=['values'])
    df.to_csv('../results/nn_results.csv')

    print('RMSE:', rmse)
    print('MAE:', mae)
