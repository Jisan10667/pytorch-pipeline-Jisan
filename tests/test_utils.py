import torch
import tempfile
from pathlib import Path
from going_modular import utils, model_builder

def test_save_and_load_model():
    model = model_builder.create_tinyvgg(10, 3, 64)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = utils.save_model(model, tmpdir, "test_model.pth")
        assert save_path.exists()
        
        loaded_model = model_builder.create_tinyvgg(10, 3, 64)
        loaded_model = utils.load_model(loaded_model, save_path)
        assert isinstance(loaded_model, torch.nn.Module)

def test_save_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test.json"
        utils.save_json({"a": 1}, json_path)
        assert json_path.exists()
        assert json_path.read_text().strip() == '{\n  "a": 1\n}'

def test_plot_curves():
    results = {"train_loss": [1.0, 0.5], "test_loss": [1.2, 0.6], "train_acc": [0.5, 0.8], "test_acc": [0.4, 0.7]}
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "loss_curves.png"
        utils.plot_loss_curves(results, save_path)
        assert save_path.exists()

def test_plot_experiment():
    summaries = [{"run_name": "a", "best_test_acc": 0.8}, {"run_name": "b", "best_test_acc": 0.9}]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "comp.png"
        utils.plot_experiment_comparison(summaries, save_path)
        assert save_path.exists()
