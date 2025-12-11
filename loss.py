from torch import Tensor

from torch.nn import Module, MSELoss


class DistillationLoss(Module):
    def __init__(self, temperature: float):
        """
        Args:
            temperature (float): The smoothing parameter for the teacher's logits.
        """

        super().__init__()

        assert temperature > 0, "temperature must be greater than 0."

        self.mse_loss_function = MSELoss()
        self.temperature = temperature

    def forward(self, y_student: Tensor, y_teacher: Tensor) -> Tensor:
        s = y_student / self.temperature
        t = y_teacher / self.temperature

        loss = self.mse_loss_function(s, t)

        return loss
