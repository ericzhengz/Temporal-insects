def get_model(model_name, args):
    name = model_name.lower()
    if name == "mos_modified":
        from models.mos_modified import Learner
    return Learner(args)