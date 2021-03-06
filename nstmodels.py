import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

device = torch.device("cpu")


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)  # bxcx(hxw)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div_(h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class Styletransfer():
    def __init__(self, cnn, style_img, content_img,
                 content_layers=['conv_3', 'conv_6'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6'],
                 epochs=300, style_weight=10000, content_weight=1):
        self.cnn = copy.deepcopy(cnn)
        self.style_img = style_img
        self.content_img = content_img
        self.input_img = self.content_img.clone()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.epochs = epochs
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.optimizer = optim.LBFGS([self.input_img.requires_grad_()], lr=1)

    def get_model_and_losses(self):
        normalization = Normalization().to(device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            model.add_module(name, layer)
            if name in self.content_layers:
                # add content loss:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            if name in self.style_layers:
                # add style loss:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        # now we trim model after last style/content loss layer
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def convert(self):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_model_and_losses()
        print('Optimizing..')
        run = [0]
        # stopper = []
        # img = []
        while run[0] <= self.epochs:
            def closure():
                # correct the values
                self.input_img.data.clamp_(0, 1)
                self.optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                    # applying weights
                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    # stopper.append((style_score + content_score).item())
                    # img.append(self.input_img.clone())
                return style_score + content_score

            self.optimizer.step(closure)
            # a last correction...
        self.input_img.data.clamp_(0, 1)
        return self.input_img
