import torch.optim as optim
from tqdm import tqdm
import warnings
import resource
from dataHandler import *
from model import Predictor
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * 0.9), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

memory_limit()
print(f"Cuda available: {torch.cuda.is_available()}")

# config ----------------------
runName = "testTransformerSimpleDeep"
dataName = "full"
batchSize = 32
epochs = 20
learningRate = 0.001
onGpu = True
# config ----------------------

tb = SummaryWriter(log_dir="./logs")
data = DataHandler("DeepSimple", dataName)
trainLoader = DataLoader(data, "train", batchSize)
evalLoader = DataLoader(data, "eval", batchSize)
print(len(trainLoader), len(evalLoader))
device = torch.device("cuda") if onGpu else torch.device("cpu")
model = Predictor(batchSize, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learningRate)

tri = 0
evi = 0
for i in range(epochs):
    t = tqdm(trainLoader)
    trErr = 0
    for iT, dT, oT in t:
        if(len(iT) == 0):
            continue
        iT = iT.to(device)
        dT = dT.to(device)
        oT = oT.to(device)

        pred = model(iT, dT, (oT != 0) + 0)

        err = torch.mean(torch.abs(oT - pred)[oT != 0])
        adjusted = pred * (dT / pred.sum(dim=1)).repeat(pred.shape[1], 1).permute(1, 0)
        mae = torch.mean(torch.abs(oT - adjusted)[oT != 0])

        err.backward()
        optimizer.step()
        trErr += err.item()
        tri += 1
        t.set_description(f"Training Loss {int(err.item())}, MAE {int(mae.item())}")
        tb.add_scalar(runName + "/loss", err.item(), tri)
        tb.add_scalar(runName + "/mae", mae.item(), tri)
    with torch.no_grad():
        evalErr = 0
        t = tqdm(evalLoader)
        for iT, dT, oT in t:
            pred = model(iT, dT, oT != 0)
            err = torch.mean(torch.abs(oT - pred)[oT != 0])
            adjusted = pred * (dT / pred.sum(dim=1)).repeat(pred.shape[1], 1).permute(1, 0)
            mae = torch.mean(torch.abs(oT - adjusted)[oT != 0])
            evalErr += err.item()
            evi += 1
            t.set_description(f"Evaluating MAE {int(evalErr / evi)}")
            tb.add_scalar(runName + "/evloss", err.item(), evi)
            tb.add_scalar(runName + "/evmae", mae.item(), evi)
        print(f"Epoch {i}, Tr Error {trErr / len(trainLoader)}, Ev Error {evalErr / len(evalLoader)}")
