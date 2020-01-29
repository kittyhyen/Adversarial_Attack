import torch
import torch.distributions


class RandomSignAttack:
    def __init__(self, epsilon=0.3):
        self.noise = torch.distributions.normal.Normal(loc=0, scale=1)
        self.epsilon = epsilon

    def generate(self, x):
        batch_noise = self.noise.sample(x.shape).sign().to(x.device)
        adv_x = torch.clamp(x.detach() + self.epsilon * batch_noise, 0, 1)

        return adv_x


class FGSMAttack:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def generate_v0(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        loss.backward()

        adv_x = torch.clamp(x.detach() + self.epsilon * torch.sign(x.grad.detach()), 0, 1).detach()

        return adv_x

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        adv_x = torch.clamp(x.detach() + self.epsilon * torch.sign(x_grad.detach()), 0, 1).detach()

        return adv_x


class LLFGSMAttack:
    def __init__(self):

    def generate(self, x, y=None):


class RandomFGSMAttack:
    def __init__(self):

    def generate(self, x, y=None):


class IterativeFGSMAttack:
    def __init__(self):

    def generate(self, x, y=None):

