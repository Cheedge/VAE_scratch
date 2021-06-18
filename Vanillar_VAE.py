import torch.nn as nn

class VanillaVAE(nn.Module):

    """
    Input:
    inout_feature: (int) H*W
    hidden_feature: (int)
    latent_feature: (int)
    x: tensor (N, H*W) batch_size, image_size

    Output:
    mu_logvar: tensor (N, 2 * latent_feature)
    mu: tensor (N, latent_feature)
    log_var: tensor (N, latent_feature)
    x_hat: tensor (N, H*W)
    """

    def __init__(self, inout_feature, hidden_feature, latent_feature):
        super(self, nn.Module).__init__()
        self.latent_feature = latent_feature
        self.encoder = nn.Sequential(
                nn.Linear(inout_feature, hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, 2 * self.latent_feature)
                )
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_feature, hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, inout_feature)
                )


    def Reparam(mu, log_var):
        var = log_var.exp()
        eps = torch.normal(mu, var)
        # z = var.mul(eps).add(mu)
        z = var * eps + mu
        return z

    def forward(self, x):
        mu_logvar = self.encoder(x)
        new_mulogvar = mu_logvar.reshape(-1, self.latent_feature, 2)
        mu = new_mulogvar[:, :, 0]
        log_var = new_mulogvar[:, :, 1]
        z = Reparam(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def BernoulliDecoder():
        pass

    def GaussianDecoder():
        pass


def Loss(x, x_hat, mu, log_var):
    # encode loss: logP(x|z)
    # BCEWithLogitsLoss=Sigmoid + BECLoss
    bcelog_loss = nn.BCEWithLogitsLoss()
    recon_los = bcelog_loss(x, x_hat)
    # KL divergence: -1/2(var**2 + mu**2 - log(var))
    kl_div = -0.5 * (var**2 + mu**2 - torch.logit(var))
    return recon_loss, kl_div
