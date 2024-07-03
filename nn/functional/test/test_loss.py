import pytest
import torch
import d2l.nn.functional as d2l
import torch.nn.functional as F


class Test_NLLLoss_1d:
    X = torch.randn(100, 10)
    y = torch.empty(100, dtype=torch.long).random_(0, 10)
    weight = torch.randn(10)

    def test_nll_loss_1d_without_weight(self):
        test = d2l.nll_loss(self.X, self.y)
        real = F.nll_loss(self.X, self.y)
        assert torch.isclose(test, real)

    def test_nll_loss_1d_with_weight(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight)
        real = F.nll_loss(self.X, self.y, weight=self.weight)
        assert torch.isclose(test, real)

    def test_nll_loss_1d_sum_reduction(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight, reduction='sum')
        real = F.nll_loss(self.X, self.y, weight=self.weight, reduction='sum')
        assert torch.isclose(test, real)

    def test_nll_loss_1d_none_reduction(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight, reduction='none')
        real = F.nll_loss(self.X, self.y, weight=self.weight, reduction='none')
        assert torch.allclose(test, real)


class Test_NLLLoss_2d:
    X = torch.randn(100, 10, 256, 256)
    y = torch.empty(100, 256, 256, dtype=torch.long).random_(0, 10)
    weight = torch.randn(10)

    def test_nll_loss_2d_without_weight(self):
        test = d2l.nll_loss(self.X, self.y)
        real = F.nll_loss(self.X, self.y)
        assert torch.isclose(test, real)

    def test_nll_loss_2d_with_weight(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight)
        real = F.nll_loss(self.X, self.y, weight=self.weight)
        assert torch.isclose(test, real)

    def test_nll_loss_2d_sum_reduction(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight, reduction='sum')
        real = F.nll_loss(self.X, self.y, weight=self.weight, reduction='sum')
        assert torch.isclose(test, real)

    def test_nll_loss_2d_none_reduction(self):
        test = d2l.nll_loss(self.X, self.y, weight=self.weight, reduction='none')
        real = F.nll_loss(self.X, self.y, weight=self.weight, reduction='none')
        assert torch.allclose(test, real)


class Test_CrossEntropyLoss_1d:
    X = torch.randn(100, 10, 256, 256)
    y = torch.empty(100, 256, 256, dtype=torch.long).random_(0, 10)
    weight = torch.randn(10)

    def test_cross_entropy_loss_1d_without_weight(self):
        test = d2l.cross_entropy(self.X, self.y)
        real = F.cross_entropy(self.X, self.y)
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_1d_with_weight(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight)
        real = F.cross_entropy(self.X, self.y, weight=self.weight)
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_1d_sum_reduction(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight, reduction='sum')
        real = F.cross_entropy(self.X, self.y, weight=self.weight, reduction='sum')
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_1d_none_reduction(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight, reduction='none')
        real = F.cross_entropy(self.X, self.y, weight=self.weight, reduction='none')
        assert torch.allclose(test, real)


class Test_CrossEntropyLoss_2d:
    X = torch.randn(100, 10, 256, 256)
    y = torch.empty(100, 256, 256, dtype=torch.long).random_(0, 10)
    weight = torch.randn(10)

    def test_cross_entropy_loss_2d_without_weight(self):
        test = d2l.cross_entropy(self.X, self.y)
        real = F.cross_entropy(self.X, self.y)
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_2d_with_weight(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight)
        real = F.cross_entropy(self.X, self.y, weight=self.weight)
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_2d_sum_reduction(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight, reduction='sum')
        real = F.cross_entropy(self.X, self.y, weight=self.weight, reduction='sum')
        assert torch.isclose(test, real)

    def test_cross_entropy_loss_2d_none_reduction(self):
        test = d2l.cross_entropy(self.X, self.y, weight=self.weight, reduction='none')
        real = F.cross_entropy(self.X, self.y, weight=self.weight, reduction='none')
        assert torch.allclose(test, real)


if __name__ == '__main__':
    pytest.main()
