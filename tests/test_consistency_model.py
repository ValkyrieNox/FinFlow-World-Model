import torch

from finflow.models import ConsistencyStudent, TransitionFM, warm_start_consistency_from_fm


def test_consistency_boundary_at_t_one_is_identity():
    torch.manual_seed(0)
    student = ConsistencyStudent(state_dim=1, condition_dim=3, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    x = torch.randn(7, 1)
    t = torch.ones(7)
    cond = torch.randn(7, 3)
    out = student(x, t, cond)
    # c_skip(1) = 1, c_out(1) = 0  =>  f(x, 1) = x
    assert torch.allclose(out, x, atol=1e-6)


def test_consistency_forward_grad_flow():
    student = ConsistencyStudent(state_dim=1, condition_dim=3, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    x = torch.randn(4, 1)
    t = torch.tensor([0.1, 0.3, 0.5, 0.9])
    cond = torch.randn(4, 3)
    out = student(x, t, cond)
    out.sum().backward()
    assert any(p.grad is not None for p in student.parameters())


def test_consistency_warm_start_copies_backbone():
    teacher = TransitionFM(state_dim=1, condition_dim=3, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    student = ConsistencyStudent(state_dim=1, condition_dim=3, hidden_dim=32, time_embedding_dim=16, num_blocks=2)
    n = warm_start_consistency_from_fm(student, teacher)
    assert n > 0
