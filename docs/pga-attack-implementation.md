# How to implement the Projected Gradient Ascent (PGA) attack 

The set up for this attack is a bit more complex of the rest of the attacks. This is why PGA needs to perform a process called SGA (Stochastic Gradient Ascent) to be able to perform the attack. In order to achieve this, we need to declare a function on the client side that will be able to perform the SGA. This function will be called by the server to perform the attack.

## Adding the `reverse_train` function to the client
Setting up the `NumpyClient`, we need to add an additional function to it called `reverse_train`. This function will function similarly to the `fit_client` function, but instead of performing the training, it will perform the SGA. 

A simple `NumpyClient` using PyTorch called `PGAClient` could look like this:

```python
class PGAClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset

        # Instantiate model
        self.model = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self):
        ...
    
    def fit(self, parameters, config):
        ...

    def evaluate(self, parameters, config):
        ...

    def reverse_train(self, parameters, reverse_config):
		device = self.device
		model = self.model
		set_params(model, parameters)
		model.to(device)

		# Read from config
        batch = reverse_config["batch_size"]
        epochs = reverse_config["epochs"]
        lr = reverse_config["lr"]

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # The custom training loop that will perform the SGA
        # Note the negative sign to get the loss
        criterion = torch.nn.CrossEntropyLoss()
        net.train()
        for _ in range(epochs):
            for batch in trainloader:    
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optim.zero_grad()
                
                # Note the negative sign
                loss = - criterion(net(images), labels) 
                
                loss.backward()
                optim.step()
		return self.get_parameters(config={})
```

This extra function will allow the server to call the `reverse_train` function on the client to perform the SGA.

