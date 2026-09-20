import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LRProbe(t.nn.Module):
    def __init__(
        self,
        d_in: int,
        scaler_mean: t.Tensor | None = None,
        scaler_scale: t.Tensor | None = None,
    ):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x: t.Tensor) -> t.Tensor:
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Returns predicted probabilities in range [0, 1]."""
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x: t.Tensor) -> t.Tensor:
        """Returns binary predictions {0.0, 1.0}."""
        return self(x).round()

    @property
    def direction(self) -> t.Tensor:
        """The 1D feature direction vector (d_model)."""
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        C: float = 0.1,
        device: str = "cpu",
    ) -> "LRProbe":
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(
            C=C, random_state=42, fit_intercept=True, max_iter=1000
        )
        lr_model.fit(X_scaled, y)

        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)

        probe = LRProbe(
            acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale
        ).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr_model.coef_[0], dtype=t.float32).to(
            device
        )

        return probe
