import torch
import numpy as np

class Trainer(object):

    def __init__(self, args, model, device, train_loader, validation_loader, in_loader, test_loader, optimizer, D):
        self.model = model
        self.device = device
        self.args = args
        self.epochs = args.epochs
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.in_loader = in_loader
        self.test_loader = test_loader
        self.verbose = self.args.verbose
        self.optimizer = optimizer
        self.reg_decay = self.args.reg_decay

        self.val_losses = []
        self.current_val_losses = []
        self.test_losses = []
        self.current_test_losses = []
        self.in_losses = []
        self.t_mse_losses = []
        self.c_mse_losses = []
        self.predicted_t = []

        self.W = D.W.to(self.device)
        self.adj = self.W.nonzero()
        self.n_pairs = len(self.adj)

        torch.manual_seed(self.args.train_seed)
        torch.cuda.manual_seed(self.args.train_seed)


    def run(self):
        print('START Training ...')
        for epoch in range(self.epochs):
            self.train(epoch=epoch)
            if epoch%self.verbose == 0:
                self.validation()
                self.test(mode='out')
                self.test(mode='in')
                if epoch%200 == 0:
                    print('-------------------------------------------------------------------------')
                    print('Epoch: {}'.format(epoch))
                    print('regs: {}'.format(self.model.alpha))
                    print('Current min val loss: {}'.format(np.min(np.array(self.current_val_losses))))
                    print('Current min test loss: {}'.format(np.min(np.array(self.current_test_losses))))
                    print('Best epoch: {}'.format(np.argsort(np.array(self.val_losses))[0]+1), np.min(np.array(self.val_losses)))
                    print('PEHE OUT',self.test_losses[np.argsort(np.array(self.val_losses))[0]])
                    print('PEHE IN',self.in_losses[np.argsort(np.array(self.val_losses))[0]])
                    print('Treatment MSE',self.t_mse_losses[np.argsort(np.array(self.val_losses))[0]])
                    print('Control MSE',self.c_mse_losses[np.argsort(np.array(self.val_losses))[0]])
                    print('--------------------------------------------------------------------------')
                    self.current_val_losses = []
                    self.current_test_losses = []
            if self.args.report:
                if epoch%20 == 0:
                 print('-------------Epoch: {}-----------------'.format(epoch))

    def train(self, epoch):
        self.model.train()
        for x, node_idx, t , target in self.train_loader:

            """ supervised loss """
            x, node_idx, t, target = x.to(self.device), node_idx.to(self.device), t.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(x, node_idx, t, target)

            """ label propagation loss """
            if epoch > self.args.start_epoch-1:
                if self.args.alpha>0:
                    random_pairs = torch.randint(self.n_pairs, size=(self.args.semi_batch_size,)).long().to(self.device)
                    a_instances1, a_instances2 = self.adj[random_pairs][:,0], self.adj[random_pairs][:,1][:,None]
                    outcome_smoothness_loss = self.model.outcome_smoothness_loss(a_instances1, a_instances2)
                    loss += outcome_smoothness_loss
                if self.args.beta>0:
                    random_pairs = torch.randint(self.n_pairs, size=(self.args.semi_batch_size,)).long().to(self.device)
                    b_instances1, b_instances2 = self.adj[random_pairs][:,0], self.adj[random_pairs][:,1][:,None]
                    treatment_effect_smoothness_loss = self.model.treatment_effect_smoothness_loss(b_instances1, b_instances2)
                    loss += treatment_effect_smoothness_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        """ regularizer decay """
        if epoch > self.args.start_epoch:
            self.model.alpha = self.args.alpha*(self.reg_decay**(epoch-self.args.start_epoch))
            self.model.beta = self.args.beta*(self.reg_decay**(epoch-self.args.start_epoch))

    def validation(self):
        size, loss = 0, 0
        self.model.eval()
        with torch.no_grad():
            for x, node_idx, t , target in self.validation_loader:
                x, node_idx, t, target = x.to(self.device), node_idx.to(self.device), t.to(self.device), target.to(self.device)
                b_loss = self.model(x, node_idx, t, target)
                loss += b_loss*len(x)
                size += len(x)

        """
            Validation loss is computed in terms of MSE using observed outcomes.
        """
        val_loss = torch.sqrt(loss/size)
        if self.args.report:
            print('Validation Loss: {}'.format(val_loss))
        self.val_losses.append(val_loss)
        self.current_val_losses.append(val_loss)

    def test(self, mode):
        self.model.eval()
        test_loss = 0
        treat_mse = 0
        cont_mse = 0
        if mode == 'out':
            with torch.no_grad():
                for x, node_idx, treat_target, cont_target in self.test_loader:
                    x, node_idx, treat_target, cont_target = x.to(self.device), node_idx.to(self.device), treat_target.to(self.device), cont_target.to(self.device)
                    x_treat, x_cont = self.model.predict(node_idx)
                    test_loss += torch.sum(((treat_target-cont_target)-(x_treat-x_cont))**2)
                    treat_mse += torch.sum((treat_target-x_treat)**2)
                    cont_mse += torch.sum((cont_target-x_cont)**2)
            test_loss /= len(self.test_loader.dataset)
            test_loss = torch.sqrt(test_loss)


            treat_mse = torch.sqrt(treat_mse/len(self.test_loader.dataset))
            cont_mse = torch.sqrt(cont_mse/len(self.test_loader.dataset))
            if self.args.report:
                print('PEHE (Out): {}'.format(test_loss))
                print('Treat Mse (Out): {}'.format(treat_mse))
                print('Cont Mse (Out): {}'.format(cont_mse))
            self.test_losses.append(test_loss)
            self.current_test_losses.append(test_loss)
            self.t_mse_losses.append(treat_mse)
            self.c_mse_losses.append(cont_mse)
        elif mode == 'in':
            with torch.no_grad():
                for x, node_idx, treat_target, cont_target in self.in_loader:
                    x, node_idx, treat_target, cont_target = x.to(self.device), node_idx.to(self.device), treat_target.to(self.device), cont_target.to(self.device)
                    x_treat, x_cont = self.model.predict(node_idx)
                    test_loss += torch.sum(((treat_target-cont_target)-(x_treat-x_cont))**2)
                    treat_mse += torch.sum((treat_target-x_treat)**2)
                    cont_mse += torch.sum((cont_target-x_cont)**2)
            test_loss /= len(self.in_loader.dataset)
            test_loss = torch.sqrt(test_loss)
            if self.args.report:
                print('PEHE (In): {}'.format(test_loss))
                print('-------------------------')
            self.in_losses.append(test_loss)
