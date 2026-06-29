import torch

from finflow.models import MeanFlowStudent, TransitionFM, warm_start_mean_flow_from_fm


def test_mean_flow_forward_shape_and_grad():
    torch.manual_seed(0)
    student = MeanFlowStudent(state_dim=1, condition_dim=4, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    x = torch.randn(5, 1, requires_grad=False)
    r = torch.rand(5)
    t = r + torch.rand(5) * (1.0 - r)
    cond = torch.randn(5, 4)
    out = student(x, r, t, cond)
    assert out.shape == (5, 1)
    out.sum().backward()
    assert any(p.grad is not None for p in student.parameters())


def test_warm_start_copies_some_parameters():
    teacher = TransitionFM(state_dim=1, condition_dim=4, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    student = MeanFlowStudent(state_dim=1, condition_dim=4, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    n = warm_start_mean_flow_from_fm(student, teacher)
    assert n > 0


def test_mean_flow_rejects_shape_mismatch():
    student = MeanFlowStudent(state_dim=1, condition_dim=3, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    try:
        student(torch.randn(4, 2), torch.rand(4), torch.rand(4), torch.randn(4, 3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong state_dim")
