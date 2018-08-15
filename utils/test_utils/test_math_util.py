import numpy as np
import pytest

from utils.math_util import softmax, log_loss, cross_entropy_loss_bc


class TestSoftmax():
    def test_logits_zero(self):
        logits = [[0., 0., 0., 0.],
                  [0., 0., 0., 0.]]
        ps = softmax(logits)
        assert ps.tolist() == [[0.25, 0.25, 0.25, 0.25],
                               [0.25, 0.25, 0.25, 0.25]]

    def test_nontrivial(self):
        logits = [1.5, 2., -0.5]
        ps = softmax(logits)
        assert ps.tolist() == pytest.approx([0.35918810578,
                                             0.59220107018,
                                             0.04861082403])

    def test_shape(self):
        logits = np.random.normal(0, 1, (7, 3, 9, 5))
        ps = softmax(logits)
        assert np.allclose(np.sum(ps, axis=-1),
                           np.ones(logits.shape[:-1]))


class TestLogLoss():
    def test_log_loss(self):
        labels = [1, 0, 2, 1]
        predictions = [[0.5, 0.5, 0.0],
                       [0.2, 0.5, 0.3],
                       [0.1, 0.2, 0.7],
                       [0.2, 0.6, 0.2]]

        expected_losses = [0.69314716056,
                           1.60943786243,
                           0.35667492965,
                           0.51082560709]

        losses = log_loss(labels, predictions)

        print('losses', losses)

        assert losses.tolist() == pytest.approx(expected_losses, rel=1e-5)


class TestCE:
    def test_cross_entropy_loss_bc(self):
        ground_truth = [
            [[0.5, 0.5]],
            [[1.0, 0.0]]
        ]

        prediction = [
            [[0.5, 0.5], [1.0, 0.5], [0.3, 0.9]],
            [[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]]
        ]

        expected_losses = [[[6.93145181e-01, 6.93145181e-01, ],
                            [6.90775478e+00, 6.93145181e-01, ],
                            [7.80321493e-01, 1.20396725e+00, ], ],

                           [[1.05359405e-01, 1.05359405e-01, ],
                            [-9.99999500e-07, -9.99999500e-07, ],
                            [1.38155106e+01, 1.38155106e+01, ], ], ]

        ce = cross_entropy_loss_bc(ground_truth, prediction)
        assert np.allclose(ce, expected_losses)
