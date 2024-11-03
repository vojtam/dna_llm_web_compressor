import torch
import numpy as np
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from get_seq import get_sequence_ensembl

from typing import List, Dict
from tqdm import tqdm

from arithmetic_encoder.coder import Coder
from arithmetic_encoder.LanguageModelAE import LanguageModel

tokenizer = AutoTokenizer.from_pretrained("vojtam/dnagpt_smaller_dict")
model = AutoModelForCausalLM.from_pretrained("vojtam/dnagpt_smaller_dict")
# Ensure the model is on the same device
model.eval()

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
TOKENIZER_VOCAB_SIZE = len(tokenizer.vocab.items())

def scale_cdf(cdf, scale_factor=50000):
    scaled = np.round(cdf * scale_factor).astype(int)
    if scaled[0] == 0:
        scaled[0] = 1
    for token, scaled_prob in enumerate(scaled):
        if token > 0 and scaled[token - 1] >= scaled_prob:
            scaled[token] = scaled[token - 1] + 1
    return scaled

def precompute_cdfs_batch(token_ids, vocab_size=TOKENIZER_VOCAB_SIZE, device='cuda', scale_factor=50000, batch_size=32, max_context_len=50):
    contexts = [token_ids[i - max_context_len : i] if i > max_context_len else token_ids[:i] for i in range(1, len(token_ids))]
    start_cdf = scale_cdf(np.array([[1/vocab_size for _ in range(vocab_size,)]]).cumsum())
    
    results = []

    for i in tqdm(range(0, len(token_ids), batch_size)):
        batch_contexts = contexts[i:i+batch_size]

        if not batch_contexts:
            continue

        # Pad sequences in the batch
        padded_contexts = [context + [tokenizer.pad_token_id] * (max_context_len - len(context)) for context in batch_contexts]
        attention_mask = [[0 if token == tokenizer.pad_token_id else 1 for token in context] for context in padded_contexts]

        batch_cdfs = next_token_cdf_batch(padded_contexts, attention_mask)
        results.append(batch_cdfs)

    stacked_cdfs = np.concatenate(results)
    scaled_cdfs = np.apply_along_axis(scale_cdf, 1, stacked_cdfs)
    stacked_cdfs = scaled_cdfs.reshape(-1, scaled_cdfs.shape[-1])
    cdfs_result = np.insert(stacked_cdfs, 0, start_cdf, axis=0)
    return cdfs_result

@torch.no_grad()
def next_token_cdf_batch(ids_batch, attention_masks=None, device='cpu'):

    batch = torch.tensor(ids_batch).to(device)
    masks = torch.tensor(attention_masks).to(device) if attention_masks is not None else None

    outputs = model(batch, attention_mask=masks)

    if masks is not None:
        last_non_pad_indices = masks.sum(dim=1) - 1
    else:
        last_non_pad_indices = torch.tensor([len(seq) - 1 for seq in ids_batch]).to(device)

    # Select logits for the last non-padded token of each sequence
    batch_size, seq_len, vocab_size = outputs.logits.shape

    logits = outputs.logits[torch.arange(batch_size), last_non_pad_indices]
    probs = torch.softmax(logits, dim=-1)
    probs_array = probs.cpu().numpy()
    cdf = probs_array.cumsum(axis=1)

    return cdf

@torch.no_grad()
def next_token_cdf(input_ids: List[int], device='cpu') -> Dict[int, float]:
    """This function predicts probability distribution of the next token based on a list of previous tokens.

    Parameters
    ----------
    input_ids : List[int]
        List of token ids
    device : str, optional
        device to send the computations to (one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia), by default 'cuda'

    Returns
    -------
    Dict[int, float]
        dictionary specifying the probability distribution over token ids

    Examples
    --------
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> tokenizer = AutoTokenizer.from_pretrained("dnagpt/human_gpt2-v1")
    >>> model = GPT2LMHeadModel.from_pretrained("dnagpt/human_gpt2-v1")
    >>> dna = "AACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    >>> input_ids = tokenizer.encode(dna, return_tensors='pt').tolist()[0]
    >>> cdf = next_token_probs(input_ids)
    """
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    outputs = model(input_tensor)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    probs_array = probs.cpu().numpy()[0]
    cdf = probs_array.cumsum()
    
    return cdf

def precompute_cdfs(token_ids, vocab_size=TOKENIZER_VOCAB_SIZE, device='cpu', scale_factor=50000, max_context_len=50):
    contexts = [token_ids[max(0, i - max_context_len) : i] for i in range(1, len(token_ids))]
    start_cdf = scale_cdf(np.array([[1/vocab_size for _ in range(vocab_size)]]).cumsum())

    token_cdfs = [next_token_cdf(context, device) for context in tqdm(contexts, desc = 'token prediction: ')]
    scaled_cdfs = [scale_cdf(cdf, scale_factor) for cdf in tqdm(token_cdfs, desc="scaling: ")]
    # Stack the CDFs into a 2D numpy array
    cdfs_result = np.vstack([start_cdf] + scaled_cdfs)
    
    assert len(token_ids) == len(cdfs_result)
    return cdfs_result


def compression_percentage(compressed: List[int], uncompresed: str | List[int], uncompressed_vocab_size = 4) -> float:
    """Computes compression percentage (how much space we saved by compression)

    Parameters
    ----------
    compressed : List[int]
        a sequence of 0 and 1 (1bit)
    uncompresed : str | List[int]
        an uncompressed sequence. Can either be raw DNA string of A, C, G, T (2bits) or
        list of token ids 
    uncompressed_vocab_size : int, optional
        the number of distinct values in uncompressed sequence

    Returns
    -------
    float
        percentage indicating how much compression we achieved
    """
    percentage = (1 - (len(compressed) * np.log2(2)) / (len(uncompresed) * np.log2(uncompressed_vocab_size))) * 100
    return percentage

def encode(dna_seq: str, cdfs = None, input_ids = None):
    if cdfs is None or input_ids is None:
        print("NO CDFS")
        input_ids = tokenizer.encode(dna_seq, return_tensors='pt').tolist()[0]
        cdfs = precompute_cdfs_batch(input_ids)

    languageModel = LanguageModel(cdfs, list(range(TOKENIZER_VOCAB_SIZE)))
    coder = Coder(languageModel, state_bits = 64,  reset_model_after_finish=True)
    compressed = coder.encode(input_ids)
    compression_percent = compression_percentage(compressed, dna_seq)
    return compression_percent


def parse_locus_string(locus_str: str):
    chr_range = locus_str.split(':')
    chr = re.sub('chr', '', chr_range[0])
    loc_range = chr_range[1].split('-')
    start = int(loc_range[0][:loc_range[0].index('.')])
    end = int(loc_range[1][:loc_range[1].index('.')])
    return (chr, start, end)

def seq_complexity(dna_seq: str) -> float:
    max_unique_per_kmer = {
        k : 4 ** k if len(dna_seq) >= 4 ** k else len(dna_seq) - (k - 1) for k in range(1, 8)
    }
    kmer_counts = { k : set() for k in range(1, 8) }
    for k in range(1, 8):
        for i in range(len(dna_seq)):
            kmer = dna_seq[i : i + k]
            if len(kmer) == k:
                kmer_counts[k].add(kmer)
    result = torch.tensor([len(kmer_counts[k]) / max_unique_per_kmer[k] for k in range(1, 8)]).prod().item()
    return result

def LZcomplexity(sequence):
    complexity = 1
    prefix_length = 1
    length_component = 1
    max_length_component = 1
    pointer = 0

    while prefix_length + length_component <= len(sequence):
        if sequence[pointer + length_component - 1] == sequence[prefix_length + length_component - 1]:
            length_component += 1
        else:
            max_length_component = max(length_component, max_length_component)
            pointer += 1

            if pointer == prefix_length:
                complexity += 1
                prefix_length += max_length_component

                pointer = 0
                max_length_component = 1
            length_component = 1
    if length_component != 1:
        complexity += 1
    return complexity

def run_encoding(locus_str: str, cdfs = None, input_ids = None):
    chr, start, end = parse_locus_string(locus_str)
    #print(chr, start, end)
    dna_seq = get_sequence_ensembl(chr, start, end)
    #print(dna_seq)
    complexity = seq_complexity(dna_seq)
    lz_complexity = LZcomplexity(dna_seq)
    compression_percent = encode(dna_seq, cdfs, input_ids)
    return (compression_percent, complexity, lz_complexity)



#percentage = encode('CTCAGGAGAACTCACTCAGAGAACAGCAAGCGGGAAATCTGCCCCCACAATCCAACCACCTCCCACCAAGTCCCTCCACCAATACTAGAGATTACAATTTGAGATGAGATTTGGGTGGCGACTTAGAGCCAAACAATATCAAGGACTTTTTTGATTCTATATGAATTTTAGATATACGTGATCATGTTATGTCATCTGCAAACAGGGACAGTTTTACTTCTTTTCTAATTTGGACATCTTTTATTTCTTTTTGCTGCCTAATTGCTCTGGCTAGGACTTCCATTACTGTATGGAATAGAAGTGGTGAGAATGGGCATCTTTGTCTTGCTCCTGAACTTAGAAAAACGCTATGACAGGGATGTAAGTATGAGTATGATGTTAGCAGTGGACTTGTCATGTACGCCTTTTATTATGTTGAGATCCATTCCTTCAATATTTATGTTGTTGAGAGTTTTTATAATGCAATAATGTTAAATTTTGTCAAATGCTTTTTCTAGTCATACAATTATTATCCTTCATCCTGTTAATGTGGCATATCACATTTACTGATTTGTGTTTATTGAACCATTCTTACATTCTAGAGGTAAATCCCACTTGGTTATGATATATGATTCTCATAAGAGCTGTTGAATTCAGTTTACTGTTATTTTGTCGAAGATGTTTGCATCTATTTTCAGCAAGTAATTTTCTTTTCTTGCAGTGTCCTTCTCTGGATTTGATATCAGAATAACCTTTGCCTCATGAAATTAGCTTGTGGGTTCCTCATCTTCATTTTTTTTGGAAATTTTGAAAAGAACTGGCTTTAATTCTTCCCAAATATTTGGCAAAAATCACACATGAAGCCATTGGTTCTGCGTTTTTCCTTGTTGAGAAGTTTTTGCATGATGATTCACTACCTTTACTCATTATTTGTCTATTTGTATTTTCTGTTTTTTTCATGATTCAGTCTTAGTAGGTTGTATTCTTCTAGGAATTAATTCCTTTCTTTGGAGTTATTCAA')
#print(percentage)

#run_encoding('chr8:127737588.55987717-127737628.55987717')