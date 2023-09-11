import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as transform
from torch.utils.data import Dataset
from datasets import load_dataset

# setup
import sys

english_file = "/Users/jayanthsadhasivan/Desktop/Personal_Projects/TransformerCode/parallel-n/IITB.en-hi.en"
hindi_file = "/Users/jayanthsadhasivan/Desktop/Personal_Projects/TransformerCode/parallel-n/IITB.en-hi.hi"

class TextDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

class SentenceTokenization():

    def __init__(self, english_file, hindi_file):
        self.english_tokens = set(["<START>", "<PADDING>" 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                          '.', ',', '!', '?', '-', ' ', ' \' ', "<END>"])
        self.hindi_tokens = set(["<START>", "<PADDING>", 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ा', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः',
                             'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
                             'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
                             'ष', 'स', 'ह', '.', ',', '!', '?', '-', ' ', '\' ', "<END>"])

        self.number_of_sentences = 100000

        self.english_sentences = []
        self.hindi_sentences = []

        temp_english = []
        temp_hindi = []
        with open(english_file, "r") as english_sentences, open(hindi_file, "r") as hindi_sentences:
            sentence_id = 0
            for english_sentence, hindi_sentence in zip(english_sentences, hindi_sentences):
                if sentence_id == self.number_of_sentences:
                    break
                temp_english.append(english_sentence)
                temp_hindi.append(hindi_sentence)

        english_char_limit = np.percentile([len(sentence) for sentence in temp_english], 97)
        hindi_char_limit = np.percentile([len(sentence) for sentence in temp_hindi], 97)

        for english_sentence, hindi_sentence in zip(temp_english, temp_hindi):
            if self.character_valid(english_sentence, self.english_tokens)\
                    and self.character_valid(hindi_sentence, self.hindi_tokens):
                if self.length_valid(english_sentence, english_char_limit)\
                        and self.length_valid(hindi_sentence, hindi_char_limit):
                    self.english_sentences.append(english_sentence)
                    self.hindi_sentences.append(hindi_sentence)

        english_to_int = dict([])
        int_to_english = dict()
        hindi_to_int = dict()
        int_to_hindi = dict()

    def character_valid(self, sentence, tokens):
        for char in sentence:
            if char not in tokens:
                return False
        return True

    def length_valid(self, sentence, char_limit):
        if len(sentence) > char_limit:
            return False
        return True

class PositionalEmbedding():

    def __init__(self, embedding_size, sequence_length):
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length

    def add_position(self):

        even_vals = torch.arange(0, self.embedding_size, 2)
        odd_vals = torch.arange(1, self.embedding_size, 2)
        evens = torch.tensor([])
        odds = torch.tensor([])


        even_curr = torch.sin( i / torch.pow(10000, even_vals / self.embedding_size))
        odd_curr = torch.cos( (i - 1) / torch.pow(10000, odd_vals / self.embedding_size))
        evens = torch.cat([evens, even_curr])
        odds = torch.cat([odds, odd_curr])


class MultiHeadAttention(nn.Module):

    def __init__(self, batch_size, max_sentence_length, embedding_size, num_heads):     # input sentences have been tokenized and positionally embedded
        super().__init__()
        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.qkv_layer = nn.Linear(embedding_size, 3 * embedding_size, dtype = torch.float64)

    def forward(self, sentences: torch.tensor, mask = None):
        # sentences  =  sentences.type(torch.float64)   # 30 x 200 x 512
        x = self.qkv_layer(sentences)   # 30 x 200 x 1536
        x = x.reshape(self.batch_size, self.max_sentence_length, self.num_heads, self.embedding_size)     # 30 x 200 x 8 x 192
        x = x.permute(0, 2, 1, 3)
        q, k, v = x.chunk(3, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.embedding_size)
        if mask:
            attention = attention + mask
        attention = transform.softmax(attention)
        attention = torch.matmul(attention, v)
        output = attention[0:self.batch_size]
        for i in range(1, self.num_heads):
            output = torch.cat((output, attention[self.batch_size * i, self.batch_size * (i + 1)]), dim=-1)
        return output

class LayerNormalization(nn.Module):

    def __init__(self, sentences, old_sentences):
        super().__init__()
        self.tiny = 1e-5
        self.attention = sentences
        self.old_embedding = old_sentences

    def forward(self):
        x = self.attention + self.old_embedding
        mean = torch.mean(x, dtype=torch.float64, dim=2, keepdim=True)
        std = torch.sqrt(torch.mean((x - mean)**2, dtype=torch.float64, dim=2, keepdim=True))
        return (x - mean)/(std + self.tiny)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sentences):
        batch_size, max_sentence_length, embedding_size = sentences.size()
        layer = nn.Linear(embedding_size, embedding_size)
        return layer(sentences)


class Encoder(nn.Module):

    def __init__(self, batch_number, max_sentence, embedding_size, iterations):
        super().__init__()
        self.batch_number = batch_number
        self.max_sentences = max_sentence
        self.embedding_size = embedding_size
        self.iterations = iterations

    def forward(self, tokenized_pos_sentences):
        a = MultiHeadAttention(tokenized_pos_sentences)
        a = MultiHeadAttention

b_size = 30
m_s_l = 200
e_size = 512
iterations = 10

