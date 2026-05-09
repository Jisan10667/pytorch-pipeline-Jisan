import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from going_moduler import engine
from going_moduler.model_builder import create_tinyvgg


def test_tinyvgg_forward_shape() -> None:
    model = create_tinyvgg(hidden_units=4, output_shape=3, image_size=16)
    output = model(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 3)


def test_train_and_test_step_return_metrics() -> None:
    torch.manual_seed(42)
    model = create_tinyvgg(hidden_units=4, output_shape=3, image_size=16)
    dataset = TensorDataset(torch.randn(6, 3, 16, 16), torch.tensor([0, 1, 2, 0, 1, 2]))
    dataloader = DataLoader(dataset, batch_size=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, train_acc = engine.train_step(model, dataloader, loss_fn, optimizer, device="cpu")
    test_loss, test_acc = engine.test_step(model, dataloader, loss_fn, device="cpu")

    assert train_loss > 0
    assert 0 <= train_acc <= 1
    assert test_loss > 0
    assert 0 <= test_acc <= 1
