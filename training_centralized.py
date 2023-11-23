from fastbook import *

p_path=Path(path)
fns = get_image_files(path)

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])
    
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
    
dls = bears.dataloaders(path)

learn = vision_learner(dls, squeezenet1_0, metrics=[error_rate, accuracy],
                       opt_func=Adam, lr=0.001, model_dir='models',
                       cut=None)


learn.fine_tune(epochs)
