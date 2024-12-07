# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.optim as optim

from .base_GAN_custom import BaseGAN
from .utils.config import BaseConfig
from .networks.progressive_conv_net import GNet


class ProgressiveGAN(BaseGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self,
                 dimLatentVector=512,
                 depthScale0=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 **kwargs):
        r"""
        Args:

        Specific Arguments:
            - depthScale0 (int)
            - initBiasToZero (bool): should layer's bias be initialized to
                                     zero ?
            - leakyness (float): negative slope of the leakyRelU activation
                                 function
            - perChannelNormalization (bool): do we normalize the output of
                                              each convolutional layer ?
            - miniBatchStdDev (bool): mini batch regularization for the
                                      discriminator
            - equalizedlR (bool): if True, forces the optimizer to see weights
                                  in range (-1, 1)

        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.depthScale0 = depthScale0
        self.config.initBiasToZero = initBiasToZero
        self.config.leakyReluLeak = leakyness
        self.config.depthOtherScales = [256,256,128,64,32,16]
        self.config.perChannelNormalization = perChannelNormalization
        self.config.alpha = 0
        self.config.miniBatchStdDev = miniBatchStdDev
        self.config.equalizedlR = equalizedlR

        BaseGAN.__init__(self, dimLatentVector, **kwargs)

    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    normalization=self.config.perChannelNormalization,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=3,
                    equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet



    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. The output resolution becomes twice
        bigger.
        """
        self.netG = self.getOriginalG()

        self.netG.addScale(depthNewScale)

        self.config.depthOtherScales.append(depthNewScale)

        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        r"""
        Update the blending factor alpha.

        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.getOriginalG().setNewAlpha(newAlpha)


        if self.avgG:
            self.avgG.module.setNewAlpha(newAlpha)

        self.config.alpha = newAlpha

    def getSize(self):
        r"""
        Get output image size (W, H)
        """
        return self.getOriginalG().getOutputSize()
