        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1)),
            torch.nn.ReLU(),
        )
#---
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 4 * 18 + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.n_classes),
        )
