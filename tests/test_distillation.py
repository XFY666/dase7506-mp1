"""Prevent sequence-length-dependent KD weighting or teacher gradients."""
import unittest
import torch
from train_experiment import token_distillation_kl


class DistillationTests(unittest.TestCase):
    def test_token_average_does_not_depend_on_batch_or_sequence_repetition(self):
        torch.manual_seed(7)
        student, teacher = torch.randn(2, 3, 11), torch.randn(2, 3, 11)
        for temperature in (1., 2.):
            expected = token_distillation_kl(student, teacher, temperature)
            repeated = token_distillation_kl(student.repeat(2, 4, 1), teacher.repeat(2, 4, 1), temperature)
            torch.testing.assert_close(expected, repeated)

    def test_only_student_receives_finite_nonzero_gradient(self):
        student = torch.randn(2, 3, 11, requires_grad=True)
        teacher = torch.randn(2, 3, 11, requires_grad=True)
        loss = token_distillation_kl(student, teacher, 2.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
