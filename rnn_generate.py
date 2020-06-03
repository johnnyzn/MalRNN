import torch
import os
import argparse
import numpy as np
from rnn_helpers import *
from rnn_model import *
import binascii
# Run as standalone script
def generate_gan( decoder, prime_str=[48, 48, 48], predict_len=1000, temperature=0.8, cuda=True):
    inp = torch.LongTensor(1, len(prime_str))
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
    predicted = prime_str
    prime_input = prime_input.type(torch.LongTensor)
    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
    output_list = []

    inp = prime_input[:,-1]
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_list.append(output)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = top_i
        predicted = np.append(predicted, predicted_char.cpu())
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return [predicted.tolist()], output_list

if __name__ == '__main__':
    file, file_len = read_file("neg/zerowalled.dll")
# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-filename', type=str, default='zerowalled.pt')
    argparser.add_argument('-p', '--prime_str', type=str, default=file[-100:-1])
    argparser.add_argument('-l', '--predict_len', type=int, default=1000)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true', default=True)
    args = argparser.parse_args()
    decoder = CharRNN(
        256,
        100,
        256,
        'gru',
        n_layers=1,
    )
    decoder = torch.load(args.filename)
    decoder.cuda()
    del args.filename
    out_int, output_listut = generate_gan(decoder, **vars(args))
    newarray = [0,0,0]
    newarray.extend(out_int)
    print(newarray)
