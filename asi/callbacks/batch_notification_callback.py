from asi.callbacks.train_callback import TrainCallback


class BatchNotificationCallback(TrainCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_batch_end(self, trainer, predictor, i_batch):
        if i_batch % 100 == 0:
            self.logger.info('Batch {}'.format(i_batch))
