from models.change_detection.change_former.modules.equal_lr import EqualLR


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
