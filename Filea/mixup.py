def mixup_data(x, y, alpha=1.0):
    """Apply Mix-Up regularization to a batch of input data and labels."""
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
