::: {#cell-0 .cell execution_count=1}
``` {.python .cell-code}
import torch
import torchvision

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import torch.nn as nn

from torch import utils
from torch import optim
from torch import device
from torch import inference_mode

import tqdm

from timeit import default_timer as timer
from tqdm.auto import tqdm

from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import numpy
from torchvision.transforms.v2 import (
    ConvertImageDtype,
    Normalize,
    Resize,
    CenterCrop,
    ToTensor,
    ToImage,
    Compose
)

import medmnist
from medmnist import INFO, Evaluator
```
:::


::: {#cell-1 .cell execution_count=2}
``` {.python .cell-code}
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    print(f"{i+1}. GPU {i}: {torch.cuda.get_device_name(i)}")

device = 0  # "Select the index of the GPU you wish to use"
torch.cuda.set_device(device)
print(f"GPU selection: {torch.cuda.get_device_name(device)}")

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device1}")
```

::: {.cell-output .cell-output-stdout}
```
Number of GPUs available: 1
1. GPU 0: NVIDIA GeForce MX110
GPU selection: NVIDIA GeForce MX110
Using device: cuda:0
```
:::
:::


::: {#cell-2 .cell execution_count=3}
``` {.python .cell-code}
transformacion = Compose([
    ToTensor(), 
    Normalize(mean=[0.5], std=[0.5])
    ])
```

::: {.cell-output .cell-output-stderr}
```
/home/pablo/.local/lib/python3.10/site-packages/torchvision/transforms/v2/_deprecated.py:42: UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.Output is equivalent up to float precision.
  warnings.warn(
```
:::
:::


::: {#cell-3 .cell execution_count=4}
``` {.python .cell-code}
data_flag = "pathmnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])

# Load the training and testing datasets
train_data = DataClass(split="train", transform=transformacion, download=True)
val_data = DataClass(split="val", transform=transformacion, download=True)
test_data = DataClass(split="test", transform=transformacion, download=True)
```

::: {.cell-output .cell-output-stdout}
```
Downloading https://zenodo.org/records/10519652/files/pathmnist.npz?download=1 to /home/pablo/.medmnist/pathmnist.npz
```
:::

::: {.cell-output .cell-output-stderr}
```
100%|██████████| 205615438/205615438 [00:20<00:00, 10059790.46it/s]
```
:::

::: {.cell-output .cell-output-stdout}
```
Using downloaded and verified file: /home/pablo/.medmnist/pathmnist.npz
Using downloaded and verified file: /home/pablo/.medmnist/pathmnist.npz
```
:::
:::


::: {#cell-4 .cell execution_count=5}
``` {.python .cell-code}
# check data properties
img = train_data[0][0]
label = train_data[0][1]

print(f"Image:\n {img}")
print(f"Label:\n {label}")

print(f"Image shape: {img.shape}")
print(f"Label: {label}")
```

::: {.cell-output .cell-output-stdout}
```
Image:
 tensor([[[0.7255, 0.7176, 0.7255,  ..., 0.7255, 0.7176, 0.7333],
         [0.7098, 0.7255, 0.7176,  ..., 0.5451, 0.5059, 0.4902],
         [0.7255, 0.7255, 0.7176,  ..., 0.6314, 0.6235, 0.6392],
         ...,
         [0.7098, 0.7020, 0.7333,  ..., 0.7333, 0.7255, 0.7333],
         [0.6706, 0.7020, 0.7333,  ..., 0.7333, 0.7333, 0.7333],
         [0.6863, 0.7255, 0.7333,  ..., 0.7255, 0.7333, 0.7412]],

        [[0.6314, 0.6235, 0.6235,  ..., 0.6314, 0.6235, 0.6314],
         [0.6157, 0.6235, 0.6157,  ..., 0.3882, 0.3490, 0.3176],
         [0.6314, 0.6235, 0.6078,  ..., 0.4980, 0.5059, 0.5216],
         ...,
         [0.6078, 0.5765, 0.6314,  ..., 0.6314, 0.6314, 0.6392],
         [0.5059, 0.5686, 0.6314,  ..., 0.6314, 0.6392, 0.6314],
         [0.5294, 0.6235, 0.6314,  ..., 0.6314, 0.6314, 0.6392]],

        [[0.7804, 0.7804, 0.7804,  ..., 0.7804, 0.7804, 0.7804],
         [0.7725, 0.7725, 0.7725,  ..., 0.5843, 0.5451, 0.5294],
         [0.7725, 0.7725, 0.7647,  ..., 0.6706, 0.6706, 0.6941],
         ...,
         [0.7647, 0.7412, 0.7804,  ..., 0.7804, 0.7804, 0.7804],
         [0.7098, 0.7412, 0.7804,  ..., 0.7804, 0.7804, 0.7804],
         [0.7255, 0.7725, 0.7804,  ..., 0.7804, 0.7804, 0.7882]]])
Label:
 [0]
Image shape: torch.Size([3, 28, 28])
Label: [0]
```
:::
:::


::: {#cell-5 .cell execution_count=6}
``` {.python .cell-code}
# Number of image channels
n_channels = info["n_channels"]
print(f"number of channels: {n_channels}")

# Number of classes
n_classes = len(info["label"])
print(f"number of classes: {n_classes}")

# Get the class names from the dataset
class_names = info["label"]
print(f"class names: {class_names}")
```

::: {.cell-output .cell-output-stdout}
```
number of channels: 3
number of classes: 9
class names: {'0': 'adipose', '1': 'background', '2': 'debris', '3': 'lymphocytes', '4': 'mucus', '5': 'smooth muscle', '6': 'normal colon mucosa', '7': 'cancer-associated stroma', '8': 'colorectal adenocarcinoma epithelium'}
```
:::
:::


::: {#cell-6 .cell execution_count=7}
``` {.python .cell-code}
for i in range(3):
    img = train_data[i][0]
    label = train_data[i][1]
    plt.figure(figsize=(3, 3))
    plt.imshow(img.permute(1, 2, 0))
    plt.title(label)
    plt.axis(False)
```

::: {.cell-output .cell-output-stderr}
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.6313726..0.84313726].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.372549..0.85882354].
```
:::

::: {.cell-output .cell-output-display}
![](lab002_CNN_files/figure-html/cell-8-output-2.png){}
:::

::: {.cell-output .cell-output-display}
![](lab002_CNN_files/figure-html/cell-8-output-3.png){}
:::

::: {.cell-output .cell-output-display}
![](lab002_CNN_files/figure-html/cell-8-output-4.png){}
:::
:::


::: {#cell-7 .cell execution_count=8}
``` {.python .cell-code}
# change data into dataloader form
BATCH_SIZE = 128
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
```
:::


::: {#cell-8 .cell execution_count=9}
``` {.python .cell-code}
# check dataloader
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")
```

::: {.cell-output .cell-output-stdout}
```
Dataloaders: (<torch.utils.data.dataloader.DataLoader object at 0x7fd88dd53cd0>, <torch.utils.data.dataloader.DataLoader object at 0x7fd88dd53490>)
Length of train dataloader: 704 batches of 128
Length of test dataloader: 57 batches of 128
Length of val dataloader: 79 batches of 128
```
:::
:::


::: {#cell-9 .cell execution_count=10}
``` {.python .cell-code}
# define training loop functions
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):

    train_loss, train_acc = 0, 0
    model.to(device)

    for batch, (X, y) in enumerate(data_loader):
        # need to change target shape for this medmnist data
        y = y.squeeze().long()

        # Send data to selected device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc
```
:::


::: {#cell-10 .cell execution_count=11}
``` {.python .cell-code}
def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):

    test_loss, test_acc = 0, 0
    model.to(device)

    model.eval()  # eval mode for testing
    with torch.inference_mode():  # Inference context manager
        for X, y in data_loader:
            # need to change target shape for this medmnist data
            y = y.squeeze().long()

            # Send data to selected device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        return test_loss, test_acc
```
:::


::: {#cell-11 .cell execution_count=12}
``` {.python .cell-code}
def eval_func(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):

    eval_loss, eval_acc = 0, 0
    model.to(device)

    model.eval()
    y_preds = []
    y_targets = []
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader)):
            # need to change target shape for this medmnist data
            y = y.squeeze().long()

            # Send data to selected device
            X, y = X.to(device), y.to(device)

            # Forward pass
            eval_pred = model(X)

            # Find loss and accuracy
            eval_loss += loss_fn(eval_pred, y)
            eval_acc += accuracy_fn(y_true=y, y_pred=eval_pred.argmax(dim=1))

            # Add prediction and target labels to list
            eval_labels = torch.argmax(torch.softmax(eval_pred, dim=1), dim=1)
            y_preds.append(eval_labels)
            y_targets.append(y)

        # Scale loss and acc
        eval_loss /= len(data_loader)
        eval_acc /= len(data_loader)

        # Put predictions on CPU for evaluation
        y_preds = torch.cat(y_preds).cpu()
        y_targets = torch.cat(y_targets).cpu()

        return {
            "model_name": model.__class__.__name__,
            "loss": eval_loss.item(),
            "accuracy": eval_acc,
            "predictions": y_preds,
            "targets": y_targets,
        }
```
:::


::: {#cell-12 .cell execution_count=13}
``` {.python .cell-code}
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
```
:::


::: {#cell-13 .cell execution_count=14}
``` {.python .cell-code}
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```
:::


::: {#cell-14 .cell execution_count=15}
``` {.python .cell-code}
class cnn(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape, out_channels=hidden_units, kernel_size=3
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, out_channels=hidden_units, kernel_size=3
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, out_channels=hidden_units * 4, kernel_size=3
            ),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units * 4,
                out_channels=hidden_units * 4,
                kernel_size=3,
            ),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units * 4,
                out_channels=hidden_units * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_units * 4 * 4 * 4, hidden_units * 8),
            nn.ReLU(),
            nn.Linear(hidden_units * 8, hidden_units * 8),
            nn.ReLU(),
            nn.Linear(hidden_units * 8, n_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define Model
model = cnn(input_shape=n_channels, hidden_units=16, output_shape=n_classes).to(device)


# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# View Model
model
```

::: {.cell-output .cell-output-display execution_count=15}
```
cnn(
  (layer1): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (layer2): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer3): Sequential(
    (0): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (layer4): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (layer5): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=1024, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=9, bias=True)
  )
)
```
:::
:::


::: {#cell-15 .cell execution_count=17}
``` {.python .cell-code}
torch.manual_seed(42)

# Measure Time

train_time_start_model = timer()

iteration_loss_list = []
iteration_accuracy_list = []

# set parameters
epochs = 10
best_loss = 10

# call train and test function
for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(
        data_loader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device1,
    )

    test_loss, test_acc = test_step(
        data_loader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device1,
    )

    for iteration, (x, y) in enumerate(train_dataloader):
        iteration_loss_list.append(train_loss.item())
        iteration_accuracy_list.append(train_acc)

    print(
        f"Epoch: {epoch} | Training loss: {train_loss:.3f} | Training acc: {train_acc:.2f} | Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}"
    )

    # save best model instance

    if test_loss < best_loss:
        best_loss = test_loss
        print(f"Saving best model for epoch: {epoch}")
        torch.save(obj=model.state_dict(), f="./model.pth")


train_time_end_model = timer()
total_train_time_model = print_train_time(
    start=train_time_start_model, end=train_time_end_model, device=device1
)
```

::: {.cell-output .cell-output-display}

```{=html}
<script type="application/vnd.jupyter.widget-view+json">
{"model_id":"debbfb67bc1244a4b0b152f46eff793e","version_major":2,"version_minor":0,"quarto_mimetype":"application/vnd.jupyter.widget-view+json"}
</script>
```

:::

::: {.cell-output .cell-output-error}

::: {.ansi-escaped-output}

```{=html}
<pre><span class="ansi-red-fg">---------------------------------------------------------------------------</span>
<span class="ansi-red-fg">KeyboardInterrupt</span>                         Traceback (most recent call last)
Cell <span class="ansi-green-fg">In[17], line 33</span>
<span class="ansi-green-fg ansi-bold">     16</span> train_loss, train_acc <span style="color:rgb(98,98,98)">=</span> train_step(
<span class="ansi-green-fg ansi-bold">     17</span>     data_loader<span style="color:rgb(98,98,98)">=</span>train_dataloader,
<span class="ansi-green-fg ansi-bold">     18</span>     model<span style="color:rgb(98,98,98)">=</span>model,
<span class="ansi-green-fg">   (...)</span>
<span class="ansi-green-fg ansi-bold">     22</span>     device<span style="color:rgb(98,98,98)">=</span>device1,
<span class="ansi-green-fg ansi-bold">     23</span> )
<span class="ansi-green-fg ansi-bold">     25</span> test_loss, test_acc <span style="color:rgb(98,98,98)">=</span> test_step(
<span class="ansi-green-fg ansi-bold">     26</span>     data_loader<span style="color:rgb(98,98,98)">=</span>test_dataloader,
<span class="ansi-green-fg ansi-bold">     27</span>     model<span style="color:rgb(98,98,98)">=</span>model,
<span class="ansi-green-fg">   (...)</span>
<span class="ansi-green-fg ansi-bold">     30</span>     device<span style="color:rgb(98,98,98)">=</span>device1,
<span class="ansi-green-fg ansi-bold">     31</span> )
<span class="ansi-green-fg">---&gt; 33</span> <span style="font-weight:bold;color:rgb(0,135,0)">for</span> iteration, (x, y) <span style="font-weight:bold;color:rgb(175,0,255)">in</span> <span style="color:rgb(0,135,0)">enumerate</span>(train_dataloader):
<span class="ansi-green-fg ansi-bold">     34</span>     iteration_loss_list<span style="color:rgb(98,98,98)">.</span>append(train_loss<span style="color:rgb(98,98,98)">.</span>item())
<span class="ansi-green-fg ansi-bold">     35</span>     iteration_accuracy_list<span style="color:rgb(98,98,98)">.</span>append(train_acc)

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py:630</span>, in <span class="ansi-cyan-fg">_BaseDataLoaderIter.__next__</span><span class="ansi-blue-fg">(self)</span>
<span class="ansi-green-fg ansi-bold">    627</span> <span style="font-weight:bold;color:rgb(0,135,0)">if</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_sampler_iter <span style="font-weight:bold;color:rgb(175,0,255)">is</span> <span style="font-weight:bold;color:rgb(0,135,0)">None</span>:
<span class="ansi-green-fg ansi-bold">    628</span>     <span style="font-style:italic;color:rgb(95,135,135)"># TODO(https://github.com/pytorch/pytorch/issues/76750)</span>
<span class="ansi-green-fg ansi-bold">    629</span>     <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_reset()  <span style="font-style:italic;color:rgb(95,135,135)"># type: ignore[call-arg]</span>
<span class="ansi-green-fg">--&gt; 630</span> data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)" class="ansi-yellow-bg">self</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">_next_data</span><span class="ansi-yellow-bg">(</span><span class="ansi-yellow-bg">)</span>
<span class="ansi-green-fg ansi-bold">    631</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_num_yielded <span style="color:rgb(98,98,98)">+</span><span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(98,98,98)">1</span>
<span class="ansi-green-fg ansi-bold">    632</span> <span style="font-weight:bold;color:rgb(0,135,0)">if</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_dataset_kind <span style="color:rgb(98,98,98)">==</span> _DatasetKind<span style="color:rgb(98,98,98)">.</span>Iterable <span style="font-weight:bold;color:rgb(175,0,255)">and</span> \
<span class="ansi-green-fg ansi-bold">    633</span>         <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_IterableDataset_len_called <span style="font-weight:bold;color:rgb(175,0,255)">is</span> <span style="font-weight:bold;color:rgb(175,0,255)">not</span> <span style="font-weight:bold;color:rgb(0,135,0)">None</span> <span style="font-weight:bold;color:rgb(175,0,255)">and</span> \
<span class="ansi-green-fg ansi-bold">    634</span>         <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_num_yielded <span style="color:rgb(98,98,98)">&gt;</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_IterableDataset_len_called:

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py:673</span>, in <span class="ansi-cyan-fg">_SingleProcessDataLoaderIter._next_data</span><span class="ansi-blue-fg">(self)</span>
<span class="ansi-green-fg ansi-bold">    671</span> <span style="font-weight:bold;color:rgb(0,135,0)">def</span> <span style="color:rgb(0,0,255)">_next_data</span>(<span style="color:rgb(0,135,0)">self</span>):
<span class="ansi-green-fg ansi-bold">    672</span>     index <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_next_index()  <span style="font-style:italic;color:rgb(95,135,135)"># may raise StopIteration</span>
<span class="ansi-green-fg">--&gt; 673</span>     data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)" class="ansi-yellow-bg">self</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">_dataset_fetcher</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">fetch</span><span class="ansi-yellow-bg">(</span><span class="ansi-yellow-bg">index</span><span class="ansi-yellow-bg">)</span>  <span style="font-style:italic;color:rgb(95,135,135)"># may raise StopIteration</span>
<span class="ansi-green-fg ansi-bold">    674</span>     <span style="font-weight:bold;color:rgb(0,135,0)">if</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_pin_memory:
<span class="ansi-green-fg ansi-bold">    675</span>         data <span style="color:rgb(98,98,98)">=</span> _utils<span style="color:rgb(98,98,98)">.</span>pin_memory<span style="color:rgb(98,98,98)">.</span>pin_memory(data, <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>_pin_memory_device)

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py:52</span>, in <span class="ansi-cyan-fg">_MapDatasetFetcher.fetch</span><span class="ansi-blue-fg">(self, possibly_batched_index)</span>
<span class="ansi-green-fg ansi-bold">     50</span>         data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>dataset<span style="color:rgb(98,98,98)">.</span>__getitems__(possibly_batched_index)
<span class="ansi-green-fg ansi-bold">     51</span>     <span style="font-weight:bold;color:rgb(0,135,0)">else</span>:
<span class="ansi-green-fg">---&gt; 52</span>         data <span style="color:rgb(98,98,98)">=</span> [<span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>dataset[idx] <span style="font-weight:bold;color:rgb(0,135,0)">for</span> idx <span style="font-weight:bold;color:rgb(175,0,255)">in</span> possibly_batched_index]
<span class="ansi-green-fg ansi-bold">     53</span> <span style="font-weight:bold;color:rgb(0,135,0)">else</span>:
<span class="ansi-green-fg ansi-bold">     54</span>     data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>dataset[possibly_batched_index]

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py:52</span>, in <span class="ansi-cyan-fg">&lt;listcomp&gt;</span><span class="ansi-blue-fg">(.0)</span>
<span class="ansi-green-fg ansi-bold">     50</span>         data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>dataset<span style="color:rgb(98,98,98)">.</span>__getitems__(possibly_batched_index)
<span class="ansi-green-fg ansi-bold">     51</span>     <span style="font-weight:bold;color:rgb(0,135,0)">else</span>:
<span class="ansi-green-fg">---&gt; 52</span>         data <span style="color:rgb(98,98,98)">=</span> [<span style="color:rgb(0,135,0)" class="ansi-yellow-bg">self</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">dataset</span><span class="ansi-yellow-bg">[</span><span class="ansi-yellow-bg">idx</span><span class="ansi-yellow-bg">]</span> <span style="font-weight:bold;color:rgb(0,135,0)">for</span> idx <span style="font-weight:bold;color:rgb(175,0,255)">in</span> possibly_batched_index]
<span class="ansi-green-fg ansi-bold">     53</span> <span style="font-weight:bold;color:rgb(0,135,0)">else</span>:
<span class="ansi-green-fg ansi-bold">     54</span>     data <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>dataset[possibly_batched_index]

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/medmnist/dataset.py:138</span>, in <span class="ansi-cyan-fg">MedMNIST2D.__getitem__</span><span class="ansi-blue-fg">(self, index)</span>
<span class="ansi-green-fg ansi-bold">    132</span> <span style="font-style:italic;color:rgb(175,0,0)">"""</span>
<span class="ansi-green-fg ansi-bold">    133</span> <span style="font-style:italic;color:rgb(175,0,0)">return: (without transform/target_transofrm)</span>
<span class="ansi-green-fg ansi-bold">    134</span> <span style="font-style:italic;color:rgb(175,0,0)">    img: PIL.Image</span>
<span class="ansi-green-fg ansi-bold">    135</span> <span style="font-style:italic;color:rgb(175,0,0)">    target: np.array of `L` (L=1 for single-label)</span>
<span class="ansi-green-fg ansi-bold">    136</span> <span style="font-style:italic;color:rgb(175,0,0)">"""</span>
<span class="ansi-green-fg ansi-bold">    137</span> img, target <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>imgs[index], <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>labels[index]<span style="color:rgb(98,98,98)">.</span>astype(<span style="color:rgb(0,135,0)">int</span>)
<span class="ansi-green-fg">--&gt; 138</span> img <span style="color:rgb(98,98,98)">=</span> <span class="ansi-yellow-bg">Image</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">fromarray</span><span class="ansi-yellow-bg">(</span><span class="ansi-yellow-bg">img</span><span class="ansi-yellow-bg">)</span>
<span class="ansi-green-fg ansi-bold">    140</span> <span style="font-weight:bold;color:rgb(0,135,0)">if</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>as_rgb:
<span class="ansi-green-fg ansi-bold">    141</span>     img <span style="color:rgb(98,98,98)">=</span> img<span style="color:rgb(98,98,98)">.</span>convert(<span style="color:rgb(175,0,0)">"</span><span style="color:rgb(175,0,0)">RGB</span><span style="color:rgb(175,0,0)">"</span>)

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/PIL/Image.py:3304</span>, in <span class="ansi-cyan-fg">fromarray</span><span class="ansi-blue-fg">(obj, mode)</span>
<span class="ansi-green-fg ansi-bold">   3301</span>         msg <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(175,0,0)">"</span><span style="color:rgb(175,0,0)">'</span><span style="color:rgb(175,0,0)">strides</span><span style="color:rgb(175,0,0)">'</span><span style="color:rgb(175,0,0)"> requires either tobytes() or tostring()</span><span style="color:rgb(175,0,0)">"</span>
<span class="ansi-green-fg ansi-bold">   3302</span>         <span style="font-weight:bold;color:rgb(0,135,0)">raise</span> <span style="font-weight:bold;color:rgb(215,95,95)">ValueError</span>(msg)
<span class="ansi-green-fg">-&gt; 3304</span> <span style="font-weight:bold;color:rgb(0,135,0)">return</span> <span class="ansi-yellow-bg">frombuffer</span><span class="ansi-yellow-bg">(</span><span class="ansi-yellow-bg">mode</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">size</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">obj</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span style="color:rgb(175,0,0)" class="ansi-yellow-bg">"</span><span style="color:rgb(175,0,0)" class="ansi-yellow-bg">raw</span><span style="color:rgb(175,0,0)" class="ansi-yellow-bg">"</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">rawmode</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">0</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">1</span><span class="ansi-yellow-bg">)</span>

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/PIL/Image.py:3206</span>, in <span class="ansi-cyan-fg">frombuffer</span><span class="ansi-blue-fg">(mode, size, data, decoder_name, *args)</span>
<span class="ansi-green-fg ansi-bold">   3203</span>         im<span style="color:rgb(98,98,98)">.</span>readonly <span style="color:rgb(98,98,98)">=</span> <span style="color:rgb(98,98,98)">1</span>
<span class="ansi-green-fg ansi-bold">   3204</span>         <span style="font-weight:bold;color:rgb(0,135,0)">return</span> im
<span class="ansi-green-fg">-&gt; 3206</span> <span style="font-weight:bold;color:rgb(0,135,0)">return</span> <span class="ansi-yellow-bg">frombytes</span><span class="ansi-yellow-bg">(</span><span class="ansi-yellow-bg">mode</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">size</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">data</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">decoder_name</span><span class="ansi-yellow-bg">,</span><span class="ansi-yellow-bg"> </span><span class="ansi-yellow-bg">args</span><span class="ansi-yellow-bg">)</span>

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/PIL/Image.py:3138</span>, in <span class="ansi-cyan-fg">frombytes</span><span class="ansi-blue-fg">(mode, size, data, decoder_name, *args)</span>
<span class="ansi-green-fg ansi-bold">   3135</span> _check_size(size)
<span class="ansi-green-fg ansi-bold">   3137</span> im <span style="color:rgb(98,98,98)">=</span> new(mode, size)
<span class="ansi-green-fg">-&gt; 3138</span> <span style="font-weight:bold;color:rgb(0,135,0)">if</span> im<span style="color:rgb(98,98,98)">.</span>width <span style="color:rgb(98,98,98)">!=</span> <span style="color:rgb(98,98,98)">0</span> <span style="font-weight:bold;color:rgb(175,0,255)">and</span> <span class="ansi-yellow-bg">im</span><span style="color:rgb(98,98,98)" class="ansi-yellow-bg">.</span><span class="ansi-yellow-bg">height</span> <span style="color:rgb(98,98,98)">!=</span> <span style="color:rgb(98,98,98)">0</span>:
<span class="ansi-green-fg ansi-bold">   3139</span>     decoder_args: Any <span style="color:rgb(98,98,98)">=</span> args
<span class="ansi-green-fg ansi-bold">   3140</span>     <span style="font-weight:bold;color:rgb(0,135,0)">if</span> <span style="color:rgb(0,135,0)">len</span>(decoder_args) <span style="color:rgb(98,98,98)">==</span> <span style="color:rgb(98,98,98)">1</span> <span style="font-weight:bold;color:rgb(175,0,255)">and</span> <span style="color:rgb(0,135,0)">isinstance</span>(decoder_args[<span style="color:rgb(98,98,98)">0</span>], <span style="color:rgb(0,135,0)">tuple</span>):
<span class="ansi-green-fg ansi-bold">   3141</span>         <span style="font-style:italic;color:rgb(95,135,135)"># may pass tuple instead of argument list</span>

File <span class="ansi-green-fg">~/.local/lib/python3.10/site-packages/PIL/Image.py:559</span>, in <span class="ansi-cyan-fg">Image.height</span><span class="ansi-blue-fg">(self)</span>
<span class="ansi-green-fg ansi-bold">    555</span> <span style="color:rgb(175,0,255)">@property</span>
<span class="ansi-green-fg ansi-bold">    556</span> <span style="font-weight:bold;color:rgb(0,135,0)">def</span> <span style="color:rgb(0,0,255)">width</span>(<span style="color:rgb(0,135,0)">self</span>) <span style="color:rgb(98,98,98)">-</span><span style="color:rgb(98,98,98)">&gt;</span> <span style="color:rgb(0,135,0)">int</span>:
<span class="ansi-green-fg ansi-bold">    557</span>     <span style="font-weight:bold;color:rgb(0,135,0)">return</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>size[<span style="color:rgb(98,98,98)">0</span>]
<span class="ansi-green-fg">--&gt; 559</span> <span style="color:rgb(175,0,255)">@property</span>
<span class="ansi-green-fg ansi-bold">    560</span> <span style="font-weight:bold;color:rgb(0,135,0)">def</span> <span style="color:rgb(0,0,255)">height</span>(<span style="color:rgb(0,135,0)">self</span>) <span style="color:rgb(98,98,98)">-</span><span style="color:rgb(98,98,98)">&gt;</span> <span style="color:rgb(0,135,0)">int</span>:
<span class="ansi-green-fg ansi-bold">    561</span>     <span style="font-weight:bold;color:rgb(0,135,0)">return</span> <span style="color:rgb(0,135,0)">self</span><span style="color:rgb(98,98,98)">.</span>size[<span style="color:rgb(98,98,98)">1</span>]
<span class="ansi-green-fg ansi-bold">    563</span> <span style="color:rgb(175,0,255)">@property</span>
<span class="ansi-green-fg ansi-bold">    564</span> <span style="font-weight:bold;color:rgb(0,135,0)">def</span> <span style="color:rgb(0,0,255)">size</span>(<span style="color:rgb(0,135,0)">self</span>) <span style="color:rgb(98,98,98)">-</span><span style="color:rgb(98,98,98)">&gt;</span> <span style="color:rgb(0,135,0)">tuple</span>[<span style="color:rgb(0,135,0)">int</span>, <span style="color:rgb(0,135,0)">int</span>]:

<span class="ansi-red-fg">KeyboardInterrupt</span>: </pre>
```

:::

:::
:::


::: {#cell-16 .cell}
``` {.python .cell-code}
# Load model
loaded_model = cnn(input_shape=n_channels, hidden_units=16, output_shape=n_classes).to(
    device
)

loaded_model.load_state_dict(torch.load(f="./model.pth"))

# get results
model_results = eval_func(
    data_loader=val_dataloader,
    model=loaded_model,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)

model_results
```
:::


::: {#cell-17 .cell}
``` {.python .cell-code}
# Get Model predictions and true targets
y_targets = model_results["targets"]
y_preds = model_results["predictions"]

# Setup confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=y_targets)

# Plot the confusion matrix
fix, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10, 7)
)
```
:::


::: {#cell-18 .cell}
``` {.python .cell-code}
# Get Model predictions and true targets
y_targets = model_results["targets"]
y_preds = model_results["predictions"]

# Setup confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=y_targets)

# Plot the confusion matrix
fix, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10, 7)
)
```
:::


