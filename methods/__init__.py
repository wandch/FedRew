from .FedRew import eICU_LocalUpdate_FedRew

def local_update(rule):
    LocalUpdate = {
                    'FedRew': eICU_LocalUpdate_FedRew,
    }
    return LocalUpdate[rule]