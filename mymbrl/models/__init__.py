import importlib

def get_item(name):
    dict = {
        "cartpole_model": "CartpoleModel"
        , "half_cheetah_model": "HalfCheetahModel"
        , "pusher_model": "PusherModel"
    }
    module = importlib.import_module("mymbrl.models."+name)
    module_class = getattr(module, dict[name])
    return module_class
    