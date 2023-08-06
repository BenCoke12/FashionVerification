import re
import numpy as np
import onnxruntime as ort
import ast
#import tensorflow as tf
#from tensorflow import keras
import idx2numpy

#counterexample
counterexample1 = np.array([[0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0], [0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0], [-5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, 0.0, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, 0.0], [0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2], [-5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, 0.0], [-5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -3.5595e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, -4.999986274509804e-2, -4.999986274509804e-2, -5.0e-2, -5.0e-2, 0.0, 4.99997254901961e-2, 2.3529411764705882e-2, 0.0, -5.0e-2, -5.0e-2, -5.000043137254902e-2, 0.0, 1.1764705882352941e-2, -5.00001568627451e-2, 7.84313725490196e-3, -5.0e-2], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.000029411764706e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0000117647058834e-2, -5.000027450980393e-2, 5.0000117647058806e-2, 5.000043137254895e-2, 4.9999862745098045e-2, -5.0e-2, -4.999972549019607e-2, -5.000043137254902e-2, -5.0e-2, 3.92156862745098e-3, 0.0, 0.0, -5.0e-2, -5.0e-2], [0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 4.9999901960784326e-2, -4.999956862745103e-2, 5.0000000000000044e-2, -4.999964705882354e-2, -5.000021568627455e-2, 5.000009803921568e-2, 4.9999529411764665e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.00001568627451e-2, 0.0, -4.999972549019607e-2, 4.999982352941179e-2, 5.000023529411765e-2, -5.0e-2], [0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.000043137254902e-2, -5.0e-2, -5.0e-2, -5.0e-2, 4.7058823529411764e-2, 4.999978431372548e-2, 5.0000000000000044e-2, 5.000047058823531e-2, 4.9999862745098045e-2, 5.000039215686275e-2, 5.000019607843137e-2, -4.999982352941179e-2, 4.999974509803917e-2, -4.999982352941168e-2, -4.9999705882352946e-2, -5.0e-2, -5.0e-2, 0.0, 4.999990196078441e-2, 5.000037254901957e-2, 4.99997254901961e-2, -5.0e-2], [0.0, 7.84313725490196e-3, 1.1764705882352941e-2, 7.84313725490196e-3, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 4.999988235294117e-2, 5.000035294117644e-2, 5.0000058823529425e-2, 4.99997254901961e-2, 4.999980392156861e-2, 5.000029411764706e-2, 4.999968627450979e-2, 5.000017647058819e-2, 5.000023529411768e-2, 4.9999529411764665e-2, 4.999988235294117e-2, 5.000049019607844e-2, -5.000009803921568e-2, 4.9999960784313735e-2, 5.0000117647058806e-2, 5.0000450980392186e-2, 5.000035294117655e-2, 5.000015686274506e-2, -5.0e-2], [-5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, -4.999999999999999e-2, -4.999988235294117e-2, 5.000007843137255e-2, 4.999982352941179e-2, 4.999998039215686e-2, 4.999980392156861e-2, 5.000021568627444e-2, 4.99997647058823e-2, 5.000001960784317e-2, 5.000013725490193e-2, 4.999982352941179e-2, 4.9999666666666664e-2, 4.999994117647055e-2, 5.000049019607844e-2, 4.9999803921568664e-2, 4.99996274509803e-2, 5.000035294117644e-2, 5.000029411764706e-2, 4.999990196078441e-2, 4.9999725490196045e-2, 5.000023529411768e-2, -4.999988235294117e-2], [-5.0e-2, -5.0000019607843144e-2, 5.000009803921571e-2, 5.000041176470588e-2, -5.000015686274506e-2, -5.00003137254903e-2, -4.9999607843137284e-2, -5.0000333333333424e-2, -5.0000333333333424e-2, 5.000043137254906e-2, 4.999982352941179e-2, 5.0000196078431314e-2, 4.999956862745103e-2, 5.0000450980392186e-2, 4.999950980392154e-2, 4.999950980392154e-2, 5.000015686274506e-2, 4.999950980392154e-2, 5.000009803921568e-2, -4.99999019607843e-2, -4.999962745098041e-2, 4.999964705882354e-2, -4.999992156862754e-2, 5.000023529411768e-2, 5.000027450980393e-2, 4.9999725490196045e-2, 5.000035294117644e-2, 4.9999666666666664e-2], [-5.000013725490196e-2, -4.9999666666666664e-2, 5.000043137254906e-2, 4.9999980392156806e-2, 5.000035294117644e-2, -5.000033333333331e-2, -5.000041176470593e-2, -4.999950980392154e-2, -4.999982352941179e-2, 4.9999784313725426e-2, 5.000021568627444e-2, 5.0000000000000044e-2, 4.9999862745098045e-2, 5.0000254901960806e-2, 4.9999803921568664e-2, 4.9999607843137284e-2, 4.999956862745103e-2, 5.000035294117644e-2, -5.000023529411757e-2, -5.000021568627444e-2, -4.999950980392165e-2, -4.999992156862754e-2, -4.99999019607843e-2, 5.0000254901960806e-2, 4.9999980392156806e-2, 4.9999725490196045e-2, 4.999982352941179e-2, -5.000029411764709e-2], [5.000003921568624e-2, 5.000027450980393e-2, 5.000035294117655e-2, 5.000021568627444e-2, -5.000035294117644e-2, -5.000027450980393e-2, -5.000029411764706e-2, -5.000027450980393e-2, -4.9999666666666664e-2, 5.000007843137255e-2, 5.000037254901968e-2, 1.87149803921568e-2, 4.9999529411764665e-2, 4.9999921568627426e-2, 5.000047058823531e-2, 5.0000196078431425e-2, 4.999974509803917e-2, 4.999984313725492e-2, 5.0000450980392186e-2, -4.999988235294117e-2, -4.999970588235292e-2, -4.999999999999993e-2, -5.000029411764706e-2, 5.000001960784317e-2, 5.000013725490193e-2, 5.000001960784317e-2, -3.529411764705881e-2, -4.999976470588241e-2], [4.9999607843137256e-2, -4.999982352941179e-2, 5.00000392156863e-2, 4.999970588235292e-2, -5.00000392156863e-2, -4.999958823529416e-2, -4.999958823529416e-2, -5.000015686274517e-2, -4.9999862745098045e-2, -5.000043137254895e-2, 4.999968627450979e-2, 4.999976470588241e-2, 5.0000196078431425e-2, 4.9999529411764665e-2, 4.9999725490196045e-2, 4.999988235294117e-2, 5.00000392156863e-2, 4.99997647058823e-2, 5.00000392156863e-2, -4.999970588235292e-2, -5.000011764705892e-2, -5.000011764705892e-2, -5.0000450980392075e-2, -4.9999725490196045e-2, 5.0000000000000044e-2, -5.000011764705892e-2, -5.000033333333331e-2, -5.000003921568627e-2], [0.0, 5.0000215686274496e-2, 4.999962745098041e-2, 4.999992156862754e-2, -5.000021568627455e-2, -5.0000392156862694e-2, -4.999970588235292e-2, -4.9999725490196045e-2, -5.0000313725490186e-2, 4.999982352941179e-2, 4.999970588235292e-2, -4.999956862745103e-2, 4.99997647058823e-2, 4.999964705882354e-2, 4.999994117647055e-2, -4.9999921568627426e-2, -4.9999921568627426e-2, -4.999996078431368e-2, 5.000047058823531e-2, 5.0000196078431314e-2, 5.000049019607844e-2, -5.000023529411768e-2, -4.999962745098041e-2, -5.00000392156863e-2, 4.999970588235303e-2, -5.0000000000000044e-2, 5.000049019607844e-2, -5.0000019607843144e-2], [-5.0e-2, -5.0e-2, 4.999996078431371e-2, 5.000035294117655e-2, -5.000047058823531e-2, -4.99999019607843e-2, -4.999999999999993e-2, -4.999982352941179e-2, 4.999988235294117e-2, -4.9999921568627426e-2, -5.00000392156863e-2, -4.9999607843137284e-2, -5.000001960784317e-2, 4.999996078431368e-2, 5.000049019607844e-2, 5.000035294117655e-2, 5.0000313725490186e-2, 5.0000058823529425e-2, -4.9999921568627426e-2, 4.9999803921568664e-2, 4.999994117647055e-2, -4.999964705882354e-2, -4.999994117647055e-2, -5.0000058823529425e-2, -4.999994117647055e-2, -4.9999980392156806e-2, -5.000021568627444e-2, 5.000015686274509e-2], [0.0, 0.0, 0.0, 0.0, -5.000035294117647e-2, -5.000007843137255e-2, -4.999994117647055e-2, -5.0000000000000044e-2, 5.000023529411757e-2, -4.9999529411764665e-2, -5.0000392156862694e-2, -4.999954901960779e-2, -4.999968627450979e-2, 5.0000313725490186e-2, 5.000033333333331e-2, 4.999990196078441e-2, 4.9999784313725426e-2, 4.9999666666666664e-2, 5.000009803921568e-2, 5.0000392156862805e-2, 5.0000117647058806e-2, -4.9999862745098045e-2, -5.0000000000000044e-2, -5.0000117647058806e-2, -4.999994117647055e-2, -5.000025490196075e-2, -5.000039215686275e-2, 0.0], [-5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2], [-5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0], [-5.0e-2, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0], [-5.0e-2, 0.0, -5.0e-2, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0, -5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, -5.0e-2, 0.0, 0.0]])
counterexample2 = np.array([[0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0], [0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0], [5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2], [5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0], [5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 3.5595e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 4.999986274509804e-2, 4.999986274509804e-2, 5.0e-2, 5.0e-2, 0.0, -4.99997254901961e-2, -2.3529411764705882e-2, 0.0, 5.0e-2, 5.0e-2, 5.000043137254902e-2, 0.0, -1.1764705882352941e-2, 5.00001568627451e-2, -7.84313725490196e-3, 5.0e-2], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.000029411764706e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0000117647058834e-2, 5.000027450980393e-2, -5.0000117647058806e-2, -5.000043137254895e-2, -4.9999862745098045e-2, 5.0e-2, 4.999972549019607e-2, 5.000043137254902e-2, 5.0e-2, -3.92156862745098e-3, 0.0, 0.0, 5.0e-2, 5.0e-2], [0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, -4.9999901960784326e-2, 4.999956862745103e-2, -5.0000000000000044e-2, 4.999964705882354e-2, 5.000021568627455e-2, -5.000009803921568e-2, -4.9999529411764665e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.00001568627451e-2, 0.0, 4.999972549019607e-2, -4.999982352941179e-2, -5.000023529411765e-2, 5.0e-2], [0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.000043137254902e-2, 5.0e-2, 5.0e-2, 5.0e-2, -4.7058823529411764e-2, -4.999978431372548e-2, -5.0000000000000044e-2, -5.000047058823531e-2, -4.9999862745098045e-2, -5.000039215686275e-2, -5.000019607843137e-2, 4.999982352941179e-2, -4.999974509803917e-2, 4.999982352941168e-2, 4.9999705882352946e-2, 5.0e-2, 5.0e-2, 0.0, -4.999990196078441e-2, -5.000037254901957e-2, -4.99997254901961e-2, 5.0e-2], [0.0, -7.84313725490196e-3, -1.1764705882352941e-2, -7.84313725490196e-3, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, -4.999988235294117e-2, -5.000035294117644e-2, -5.0000058823529425e-2, -4.99997254901961e-2, -4.999980392156861e-2, -5.000029411764706e-2, -4.999968627450979e-2, -5.000017647058819e-2, -5.000023529411768e-2, -4.9999529411764665e-2, -4.999988235294117e-2, -5.000049019607844e-2, 5.000009803921568e-2, -4.9999960784313735e-2, -5.0000117647058806e-2, -5.0000450980392186e-2, -5.000035294117655e-2, -5.000015686274506e-2, 5.0e-2], [5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 4.999999999999999e-2, 4.999988235294117e-2, -5.000007843137255e-2, -4.999982352941179e-2, -4.999998039215686e-2, -4.999980392156861e-2, -5.000021568627444e-2, -4.99997647058823e-2, -5.000001960784317e-2, -5.000013725490193e-2, -4.999982352941179e-2, -4.9999666666666664e-2, -4.999994117647055e-2, -5.000049019607844e-2, -4.9999803921568664e-2, -4.99996274509803e-2, -5.000035294117644e-2, -5.000029411764706e-2, -4.999990196078441e-2, -4.9999725490196045e-2, -5.000023529411768e-2, 4.999988235294117e-2], [5.0e-2, 5.0000019607843144e-2, -5.000009803921571e-2, -5.000041176470588e-2, 5.000015686274506e-2, 5.00003137254903e-2, 4.9999607843137284e-2, 5.0000333333333424e-2, 5.0000333333333424e-2, -5.000043137254906e-2, -4.999982352941179e-2, -5.0000196078431314e-2, -4.999956862745103e-2, -5.0000450980392186e-2, -4.999950980392154e-2, -4.999950980392154e-2, -5.000015686274506e-2, -4.999950980392154e-2, -5.000009803921568e-2, 4.99999019607843e-2, 4.999962745098041e-2, -4.999964705882354e-2, 4.999992156862754e-2, -5.000023529411768e-2, -5.000027450980393e-2, -4.9999725490196045e-2, -5.000035294117644e-2, -4.9999666666666664e-2], [5.000013725490196e-2, 4.9999666666666664e-2, -5.000043137254906e-2, -4.9999980392156806e-2, -5.000035294117644e-2, 5.000033333333331e-2, 5.000041176470593e-2, 4.999950980392154e-2, 4.999982352941179e-2, -4.9999784313725426e-2, -5.000021568627444e-2, -5.0000000000000044e-2, -4.9999862745098045e-2, -5.0000254901960806e-2, -4.9999803921568664e-2, -4.9999607843137284e-2, -4.999956862745103e-2, -5.000035294117644e-2, 5.000023529411757e-2, 5.000021568627444e-2, 4.999950980392165e-2, 4.999992156862754e-2, 4.99999019607843e-2, -5.0000254901960806e-2, -4.9999980392156806e-2, -4.9999725490196045e-2, -4.999982352941179e-2, 5.000029411764709e-2], [-5.000003921568624e-2, -5.000027450980393e-2, -5.000035294117655e-2, -5.000021568627444e-2, 5.000035294117644e-2, 5.000027450980393e-2, 5.000029411764706e-2, 5.000027450980393e-2, 4.9999666666666664e-2, -5.000007843137255e-2, -5.000037254901968e-2, -1.87149803921568e-2, -4.9999529411764665e-2, -4.9999921568627426e-2, -5.000047058823531e-2, -5.0000196078431425e-2, -4.999974509803917e-2, -4.999984313725492e-2, -5.0000450980392186e-2, 4.999988235294117e-2, 4.999970588235292e-2, 4.999999999999993e-2, 5.000029411764706e-2, -5.000001960784317e-2, -5.000013725490193e-2, -5.000001960784317e-2, 3.529411764705881e-2, 4.999976470588241e-2], [-4.9999607843137256e-2, 4.999982352941179e-2, -5.00000392156863e-2, -4.999970588235292e-2, 5.00000392156863e-2, 4.999958823529416e-2, 4.999958823529416e-2, 5.000015686274517e-2, 4.9999862745098045e-2, 5.000043137254895e-2, -4.999968627450979e-2, -4.999976470588241e-2, -5.0000196078431425e-2, -4.9999529411764665e-2, -4.9999725490196045e-2, -4.999988235294117e-2, -5.00000392156863e-2, -4.99997647058823e-2, -5.00000392156863e-2, 4.999970588235292e-2, 5.000011764705892e-2, 5.000011764705892e-2, 5.0000450980392075e-2, 4.9999725490196045e-2, -5.0000000000000044e-2, 5.000011764705892e-2, 5.000033333333331e-2, 5.000003921568627e-2], [0.0, -5.0000215686274496e-2, -4.999962745098041e-2, -4.999992156862754e-2, 5.000021568627455e-2, 5.0000392156862694e-2, 4.999970588235292e-2, 4.9999725490196045e-2, 5.0000313725490186e-2, -4.999982352941179e-2, -4.999970588235292e-2, 4.999956862745103e-2, -4.99997647058823e-2, -4.999964705882354e-2, -4.999994117647055e-2, 4.9999921568627426e-2, 4.9999921568627426e-2, 4.999996078431368e-2, -5.000047058823531e-2, -5.0000196078431314e-2, -5.000049019607844e-2, 5.000023529411768e-2, 4.999962745098041e-2, 5.00000392156863e-2, -4.999970588235303e-2, 5.0000000000000044e-2, -5.000049019607844e-2, 5.0000019607843144e-2], [5.0e-2, 5.0e-2, -4.999996078431371e-2, -5.000035294117655e-2, 5.000047058823531e-2, 4.99999019607843e-2, 4.999999999999993e-2, 4.999982352941179e-2, -4.999988235294117e-2, 4.9999921568627426e-2, 5.00000392156863e-2, 4.9999607843137284e-2, 5.000001960784317e-2, -4.999996078431368e-2, -5.000049019607844e-2, -5.000035294117655e-2, -5.0000313725490186e-2, -5.0000058823529425e-2, 4.9999921568627426e-2, -4.9999803921568664e-2, -4.999994117647055e-2, 4.999964705882354e-2, 4.999994117647055e-2, 5.0000058823529425e-2, 4.999994117647055e-2, 4.9999980392156806e-2, 5.000021568627444e-2, -5.000015686274509e-2], [0.0, 0.0, 0.0, 0.0, 5.000035294117647e-2, 5.000007843137255e-2, 4.999994117647055e-2, 5.0000000000000044e-2, -5.000023529411757e-2, 4.9999529411764665e-2, 5.0000392156862694e-2, 4.999954901960779e-2, 4.999968627450979e-2, -5.0000313725490186e-2, -5.000033333333331e-2, -4.999990196078441e-2, -4.9999784313725426e-2, -4.9999666666666664e-2, -5.000009803921568e-2, -5.0000392156862805e-2, -5.0000117647058806e-2, 4.9999862745098045e-2, 5.0000000000000044e-2, 5.0000117647058806e-2, 4.999994117647055e-2, 5.000025490196075e-2, 5.000039215686275e-2, 0.0], [5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0e-2, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0]])
counterexampleOld = np.array([[0.0, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2], [0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2], [0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2], [5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0], [5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2], [0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, -7.842862745098038e-3, -7.842862745098038e-3, 0.0, 0.0, 5.0e-2, 4.99997254901961e-2, 5.0000411764705883e-2, 5.0e-2, 0.0, 0.0, -3.921431372549019e-3, 0.0, 4.999970588235294e-2, -1.96081568627451e-2, 5.000013725490196e-2, 0.0], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, -1.176429411764706e-2, 0.0, 0.0, 0.0, -5.0000117647058834e-2, -5.000027450980393e-2, 5.0000117647058806e-2, 5.000043137254895e-2, 4.9999862745098045e-2, 0.0, -1.5686725490196077e-2, -3.921431372549019e-3, 0.0, 4.999956862745098e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 4.9999901960784326e-2, -4.999956862745103e-2, -5.0000000000000044e-2, -4.999964705882354e-2, -5.000021568627455e-2, 5.000009803921568e-2, -5.000047058823537e-2, 0.0, 0.0, 0.0, -1.96081568627451e-2, 5.0e-2, -1.5686725490196077e-2, 4.999982352941179e-2, 5.000023529411765e-2, 0.0], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, -3.921431372549019e-3, 0.0, 0.0, 0.0, 4.999982352941176e-2, 4.999978431372548e-2, 0.0, 5.000047058823531e-2, 4.9999862745098045e-2, 5.000039215686275e-2, -4.999980392156855e-2, -4.999982352941179e-2, 4.999974509803917e-2, -4.999982352941168e-2, -4.9999705882352946e-2, 0.0, 0.0, 5.0e-2, 4.999990196078441e-2, 5.000037254901957e-2, 4.99997254901961e-2, 0.0], [5.0e-2, 5.000013725490196e-2, 4.999970588235294e-2, 5.000013725490196e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 4.999988235294117e-2, 5.000035294117644e-2, 5.0000058823529425e-2, 4.99997254901961e-2, -5.0000196078431425e-2, 5.000029411764706e-2, 4.999968627450979e-2, 4.7059176470588215e-2, 5.000023529411768e-2, 4.9999529411764665e-2, 4.999988235294117e-2, 5.000049019607844e-2, -5.000009803921568e-2, 4.9999960784313735e-2, 5.0000117647058806e-2, 5.0000450980392186e-2, 5.000035294117655e-2, 5.000015686274506e-2, 5.0e-2], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, -4.999999999999999e-2, -4.999988235294117e-2, -4.9999921568627426e-2, 4.999982352941179e-2, 4.999998039215686e-2, 4.999980392156861e-2, 5.000021568627444e-2, 4.99997647058823e-2, 5.000001960784317e-2, 5.000013725490193e-2, 4.999982352941179e-2, 4.9999666666666664e-2, 4.999994117647055e-2, 5.000049019607844e-2, 4.9999803921568664e-2, 4.99996274509803e-2, 5.000035294117644e-2, 5.000029411764706e-2, 4.999990196078441e-2, 4.9999725490196045e-2, 5.000023529411768e-2, 5.000011764705882e-2], [0.0, 4.999998039215686e-2, 5.000009803921571e-2, 5.000041176470588e-2, -5.000015686274506e-2, -5.00003137254903e-2, -4.9999607843137284e-2, 4.9999666666666664e-2, 4.9999666666666664e-2, 5.000043137254906e-2, 4.999982352941179e-2, 5.0000196078431314e-2, 4.999956862745103e-2, 5.0000450980392186e-2, 4.999950980392154e-2, 4.999950980392154e-2, 5.000015686274506e-2, -5.000049019607844e-2, 5.000009803921568e-2, -4.99999019607843e-2, -4.999962745098041e-2, 4.999964705882354e-2, -4.999992156862754e-2, 5.000023529411768e-2, 5.000027450980393e-2, 4.9999725490196045e-2, 5.000035294117644e-2, 4.9999666666666664e-2], [4.9999862745098045e-2, 5.000033333333331e-2, 5.000043137254906e-2, 4.9999980392156806e-2, -4.999964705882354e-2, -5.000033333333331e-2, -5.000041176470593e-2, 1.3600490196078452e-2, -4.999982352941179e-2, 4.9999784313725426e-2, 5.000021568627444e-2, 5.0000000000000044e-2, -5.0000137254902044e-2, 5.0000254901960806e-2, 4.9999803921568664e-2, 4.9999607843137284e-2, 4.999956862745103e-2, 5.000035294117644e-2, 4.999976470588241e-2, -5.000021568627444e-2, -4.999950980392165e-2, -4.999992156862754e-2, -4.99999019607843e-2, -4.999974509803917e-2, 4.9999980392156806e-2, 4.9999725490196045e-2, 4.999982352941179e-2, -5.000029411764709e-2], [5.000003921568624e-2, 5.000027450980393e-2, 5.000035294117655e-2, 5.000021568627444e-2, -5.000035294117644e-2, -5.000027450980393e-2, -5.000029411764706e-2, -5.000027450980393e-2, 5.000033333333331e-2, 5.000007843137255e-2, 5.000037254901968e-2, -5.000001960784317e-2, 4.9999529411764665e-2, 4.9999921568627426e-2, 5.000047058823531e-2, 5.0000196078431425e-2, 4.999974509803917e-2, 4.999984313725492e-2, 5.0000450980392186e-2, 5.0000117647058806e-2, -4.999970588235292e-2, -4.999999999999993e-2, -5.000029411764706e-2, 5.000001960784317e-2, 5.000013725490193e-2, 5.000001960784317e-2, -5.000011764705892e-2, 5.000023529411762e-2], [4.9999607843137256e-2, -4.999982352941179e-2, 5.00000392156863e-2, 4.999970588235292e-2, -5.00000392156863e-2, -4.999958823529416e-2, -4.999958823529416e-2, -5.000015686274517e-2, -4.9999862745098045e-2, -5.000043137254895e-2, 4.999968627450979e-2, 4.999976470588241e-2, 5.0000196078431425e-2, 4.9999529411764665e-2, -5.000027450980393e-2, 4.999988235294117e-2, 5.00000392156863e-2, -5.000023529411768e-2, -4.999996078431368e-2, -4.999970588235292e-2, 4.999988235294117e-2, 4.999988235294117e-2, 4.99995490196079e-2, -4.9999725490196045e-2, 5.0000000000000044e-2, -5.000011764705892e-2, -5.000033333333331e-2, -5.000003921568627e-2], [5.0e-2, 5.0000215686274496e-2, 4.999962745098041e-2, 4.999992156862754e-2, -5.000021568627455e-2, -5.0000392156862694e-2, -4.999970588235292e-2, -4.9999725490196045e-2, -5.0000313725490186e-2, 4.999982352941179e-2, 4.999970588235292e-2, 5.000043137254895e-2, 4.99997647058823e-2, 4.999964705882354e-2, 4.999994117647055e-2, -4.9999921568627426e-2, -4.9999921568627426e-2, -4.999996078431368e-2, 5.000047058823531e-2, 5.0000196078431314e-2, 5.000049019607844e-2, -5.000023529411768e-2, -4.999962745098041e-2, -5.00000392156863e-2, 4.999970588235303e-2, -5.0000000000000044e-2, 5.000049019607844e-2, -2.7451019607843137e-2], [0.0, 0.0, 4.999996078431371e-2, 5.000035294117655e-2, -5.000047058823531e-2, -4.99999019607843e-2, -4.999999999999993e-2, -4.999982352941179e-2, 4.999988235294117e-2, -4.9999921568627426e-2, -5.00000392156863e-2, 5.0000392156862805e-2, 4.9999980392156806e-2, 4.999996078431368e-2, -4.999950980392154e-2, 5.000035294117655e-2, 5.0000313725490186e-2, 5.0000058823529425e-2, -4.9999921568627426e-2, 4.9999803921568664e-2, 4.999994117647055e-2, 5.000035294117655e-2, -4.999994117647055e-2, -5.0000058823529425e-2, -4.999994117647055e-2, -4.9999980392156806e-2, -5.000021568627444e-2, 5.000015686274509e-2], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, -5.000035294117647e-2, -5.000007843137255e-2, -4.999994117647055e-2, -5.0000000000000044e-2, 5.000023529411757e-2, -4.9999529411764665e-2, 4.9999607843137284e-2, -4.999954901960779e-2, 5.0000313725490186e-2, 5.0000313725490186e-2, -4.9999666666666664e-2, 4.999990196078441e-2, 4.9999784313725426e-2, 4.9999666666666664e-2, 5.000009803921568e-2, 5.0000392156862805e-2, 5.0000117647058806e-2, 5.000013725490193e-2, -5.0000000000000044e-2, -5.0000117647058806e-2, -4.999994117647055e-2, -5.000025490196075e-2, -5.000039215686275e-2, 5.0e-2], [5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2], [5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 5.0e-2], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2], [0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 5.0e-2], [0.0, 5.0e-2, 0.0, 5.0e-2, 0.0, 0.0, 0.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

counterexample = counterexample2

#Check it is a valid image
criterion = (counterexample <= 1) & (counterexample >= 0)
print("Valid image criterion satisfied on all pixels: ", criterion.all())

#original image and label (index 104 of the test set)

original_image = idx2numpy.convert_from_file('../idxdata/individuals/Image104.idx')[0]
original_label = idx2numpy.convert_from_file('../idxdata/individuals/Label104.idx')[0]

#check it is epsilon distance from the original image
print("Within epsilon distance: ", np.allclose(original_image, counterexample, rtol=0, atol=0.05001))

#Check a robustness counterexample causes a missclassification
#ort
ortSession = ort.InferenceSession("../onnxnetworks/fashion1l32n.onnx")

#original image classified label
image = np.array([np.array(row) for row in original_image])
imageForRuntime = image.astype(np.float32)
imageReshape = np.array([imageForRuntime])
ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
original_image_prediction = ortSession.run(None, ort_inputs)

#counterexample classified label
image = np.array([np.array(row) for row in counterexample])
imageForRuntime = image.astype(np.float32)
imageReshape = np.array([imageForRuntime])
ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
counterexample_prediction = ortSession.run(None, ort_inputs)

print("Original image predicted label: ", np.argmax(original_image_prediction))
print("Counterexample predicted label: ", np.argmax(counterexample_prediction))
print("True Label: ", original_label)

#assert(original label == classified label)
np.testing.assert_allclose(original_image, counterexample, rtol=0, atol=0.05001)
