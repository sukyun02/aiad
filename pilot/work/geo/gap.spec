        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=4),
        )
#---
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 1 * 1 + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.n_classes),
        )
