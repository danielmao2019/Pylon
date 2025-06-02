def test_compare_pytorch() -> None:
    dataset = dataset_config['class'](**dataset_config['args'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
    criterion = config['criterion']['class'](**config['criterion']['args'])
    metric = config['metric']['class'](**config['metric']['args'])
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-03)
    all_accuracies: List[float] = []
    for _ in range(config['epochs']):
        # train epoch
        model.train()
        for example in dataloader:
            outputs = model(example['inputs']['image'])
            loss = criterion(y_pred=outputs, y_true=example['labels']['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # val epoch
        model.eval()
        metric.reset_buffer()
        for example in dataloader:
            outputs = model(example['inputs']['image'])
            metric(y_pred=outputs, y_true=example['labels']['target'])
        all_accuracies.append(metric.summarize()['accuracy'])
    plt.figure()
    plt.plot(all_accuracies)
    plt.savefig(os.path.join(config['work_dir'], "acc_pytorch.png"))
