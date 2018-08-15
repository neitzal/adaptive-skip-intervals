class TrainCallback:
    """
    Keras-inspired callback object.
    Override desired methods.
    """

    def on_epoch_end(self, trainer, predictor, i_epoch):
        pass

    def on_batch_end(self, trainer, predictor, i_batch):
        pass

    def on_validation_end(self, trainer, predictor,
                          train_metrics, valid_metrics, i_epoch):
        pass

    def on_keyboard_interrupt(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass