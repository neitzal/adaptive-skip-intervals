from asi.callbacks.train_callback import TrainCallback
from utils.misc import ObjectSaver


class SaverCallback(TrainCallback):
    def __init__(self, object_saver: ObjectSaver):
        self.object_saver = object_saver

    def on_epoch_end(self, trainer, predictor, i_epoch):
        self.object_saver.save()

    def on_keyboard_interrupt(self, trainer):
        self.object_saver.save()

    def on_training_end(self, trainer):
        self.object_saver.save()
