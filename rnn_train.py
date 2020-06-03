import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from tqdm import tqdm
from rnn_helpers import *
from rnn_model import *
from rnn_generate import *
import pickle
# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str, default="neg/mset7tk.dll")
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=1000000)
argparser.add_argument('--print_every', type=int, default=1)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=10)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true', default=True)
argparser.add_argument('--resume', action='store_true', default=False)
args = argparser.parse_args()
# Change the path and tag_name to differeciate
tag_name = 'malrnn'
malware_path = ''
save_path = ''
MALCONV_MODEL_PATH = 'trained_model/malconv/malconv.checkpoint'
NONNEG_MODEL_PATH = 'trained_model/nonneg/nonneg.checkpoint'


def sys_sample(bytez):
    result = []
    for i in range(0, len(bytez), 10000):
        result.append(int(bytez[i]))
    return result

from MalConv import MalConv
import threading
import torch.nn.functional as F
outputs1 = None
outputs2 = None
def check(_inp, model1, model2):
    with torch.no_grad():
        _inp = torch.from_numpy(np.frombuffer(_inp,dtype=np.uint8)[np.newaxis,:] )
        a = model1(_inp)
        b = model2(_inp)
        global outputs1
        global outputs2
        outputs1 = F.softmax( a, dim=-1)
        outputs2 = F.softmax( b, dim=-1)

malconv = MalConv(channels=256, window_size=512, embd_size=8).train()
non_malconv = MalConv(channels=256, window_size=512, embd_size=8).train()
if args.cuda:
    weights = torch.load(os.path.join(MALCONV_MODEL_PATH),map_location='cuda') 
    non_weights = torch.load(os.path.join(NONNEG_MODEL_PATH),map_location='cuda') 
else:
    weights = torch.load(os.path.join(MALCONV_MODEL_PATH),map_location='cpu') 
    non_weights = torch.load(os.path.join(NONNEG_MODEL_PATH),map_location='cpu') 
malconv.load_state_dict( weights['model_state_dict'])
non_malconv.load_state_dict( non_weights['model_state_dict'])

loss_record = []
duration_record = []

if args.resume:
    with open('visualization_and_model/domain_specific/loss_'+tag_name+'.txt', "rb") as filehandle:
        loss_record = pickle.load(filehandle)
    with open('visualization_and_model/domain_specific/loss_'+tag_name+'_pointer_'+'.txt', "rb") as filehandle:
        duration_record = pickle.load(filehandle)
index = len([i for i in duration_record if (i !=0)])
model_path = 'models/'
if args.cuda:
    print("Using CUDA")

if not os.path.exists(save_path):
    os.makedirs(save_path)
target_list = os.listdir(malware_path)
saved_list = os.listdir(save_path)
cur_num_remaining = len(target_list)

cur_file = ''
# To resum the progress: check existing files and skip
for i in range(0,index):
    print(cur_file)
    cur_file = target_list.pop()
if(index == 0):
    cur_file = target_list.pop()
file2, file_len2, bytez2= read_file(malware_path+cur_file)
pool_sample = sys_sample(file2)
counter = 0
file, file_len, bytez = read_file(args.filename)
print(index) 
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    global cur_file
    global file2
    global file_len2
    global cur_num_remaining
    global bytez2
    global counter
    global pool_sample
    counter += 1
    hidden = decoder.init_hidden(args.batch_size)
    flag = 0
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0
    predicted, output_list = generate_gan(decoder, prime_str=pool_sample, predict_len=35, cuda=args.cuda)
    generated_bytes = bytez2+bytearray(predicted[0])
    fresult =  []
    thread = threading.Thread(target=check, args=(generated_bytes,malconv,non_malconv,))
    thread.start()
    thread.join()
    fresult.append(outputs1.detach().numpy()[0,1])
    fresult.append(outputs2.detach().numpy()[0,1])
    if((fresult[0]<=0.5) and(fresult[1]<=0.35)):
        print("get one!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        with open(save_path+cur_file, 'wb') as f:
            f.write(bytearray(generated_bytes))
            print("saving: ",cur_file)
        try:
            cur_file = target_list.pop()
            file2, file_len2, bytez2 = read_file(malware_path+cur_file)
            pool_sample = sys_sample(file2)
        except:
            pass        
        cur_num_remaining -= 1
        counter = 0
        flag = 1
    if(counter>=50):
        try:
            cur_file = target_list.pop()
            file2, file_len2, bytez2 = read_file(malware_path+cur_file)
            pool_sample = sys_sample(file2)
        except:
            pass        
        cur_num_remaining -= 1 
        counter = 0
        flag = 2       
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> model score:',fresult)
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])
    loss.backward()
    decoder_optimizer.step()
    return loss.data / args.chunk_len, flag

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, 'visualization_and_model/'+tag_name+'.pt')
    print('Saved as %s' % save_filename)

# Initialize models_orig and start training
decoder = CharRNN(
    256,
    args.hidden_size,
    256,
    model=args.model,
    n_layers=args.n_layers,
)
if args.resume:
    decoder = torch.load('visualization_and_model/'+tag_name+'.pt')
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
def start_train():
    if args.cuda:
        decoder.cuda()
    else:
        decoder.cpu()
    start = time.time()
    loss_avg = 0
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        if(cur_num_remaining<=0):
            break
        try:
            loss, flag  = train(*random_training_set(args.chunk_len, args.batch_size))
        except:
            loss = 0
            flag = 3
        loss_record.append(loss)
        duration_record.append(flag)
        loss_avg += loss
        with open('visualization_and_model/loss_'+tag_name+'.txt', "wb") as fp:   #Pickling
            pickle.dump(loss_record, fp)
        with open('visualization_and_model/loss_'+tag_name+'_pointer_'+'.txt', "wb") as fp:   #Pickling
            pickle.dump(duration_record, fp)
        if epoch % args.print_every == 0:
            print('[total time:%s (epoch #:%d progress:%d%%) loss:%.4f]\n' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            save()
    print("Saving...")
    save()
start_train()    
