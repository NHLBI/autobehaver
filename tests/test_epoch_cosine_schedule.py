import importlib.util
from pathlib import Path
import math
import torch

spec = importlib.util.spec_from_file_location('epoch_schedulers', Path(__file__).parents[1]/'autobehaver/schedulers.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_loader_sized_warmup_decay_and_no_restart():
    for batches in (3, 7):
        check_schedule(batches)


def check_schedule(batches):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1e-4)
    sched = module.EpochCosineScheduler(opt, batches, 1, 3, 1e-6)
    rates = [opt.param_groups[0]['lr']]
    for _ in range(5*batches):
        opt.step()
        sched.step()
        rates.append(opt.param_groups[0]['lr'])
    assert math.isclose(rates[0], 1e-9)
    assert math.isclose(rates[batches], 1e-4)
    assert rates[4*batches] == 1e-6
    assert all(a >= b for a,b in zip(rates[batches:],rates[batches+1:]))
    assert rates[-1] == 1e-6


def test_state_restores_next_lr():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1e-4)
    sched = module.EpochCosineScheduler(opt, 4, 1, 3, 1e-6)
    for _ in range(8): opt.step(); sched.step()
    q = torch.nn.Parameter(torch.zeros(1))
    opt2 = torch.optim.SGD([q], lr=1e-4)
    sched2 = module.EpochCosineScheduler(opt2, 4, 1, 3, 1e-6)
    opt2.load_state_dict(opt.state_dict())
    sched2.load_state_dict(sched.state_dict())
    opt.step(); sched.step(); opt2.step(); sched2.step()
    assert sched.get_last_lr() == sched2.get_last_lr()

if __name__ == "__main__":
    test_loader_sized_warmup_decay_and_no_restart()
    test_state_restores_next_lr()
    print("Epoch cosine schedule checks passed")
