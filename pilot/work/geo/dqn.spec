        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
#---
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 8 * 8 + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.n_classes),
        )
