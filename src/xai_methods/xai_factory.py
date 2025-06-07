from torch.nn import Module
from captum.attr import LayerGradCam

class XaiFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, explainer_class):

        pass

    @staticmethod
    def create(self, xai_methode:str, model: Module):
        if  xai_methode=="GrandCam":
            target_layer = model.layer4[-1] #todo was hat es sich mit grand cam und der layer auswahl aufsich
            return LayerGradCam(model,target_layer)