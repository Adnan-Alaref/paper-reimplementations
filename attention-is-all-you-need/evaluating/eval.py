from __future__ import annotations

from src.vocabulary import Vocabulary
from src.imports import *

# ── Add Padding ────────────────────────────────────────────────────────
def pad_sentences(encoded_sentences: List[List[int]], pad_idx: int)-> torch.tensor:
    batch_size = len(encoded_sentences)
    
    max_seq_len = max(map(len, encoded_sentences))
    padded_tensor = torch.full(
        size = (batch_size, max_seq_len),
        dtype = torch.long,
        fill_value = pad_idx
    )

    for i, sentence in enumerate(encoded_sentences):
        padded_tensor[i, : len(sentence)]  = torch.tensor(
            sentence,
            dtype = torch.long
        )

    return padded_tensor # [B, T]


# ── Evaluate Model ────────────────────────────────────────────────────────
def eval_model(
    translator: nn.Module, 
    max_len : int,
    src_path: str, 
    tgt_path: str, 
    pad_idx:  int,
    eng_sentences: List[str],
    device:torch.device)-> None:
    
    # ── Load source and target vocabularies ──
    src_vocab = Vocabulary.load(src_path)
    tgt_vocab = Vocabulary.load(tgt_path)
    
    # ── Evaluation Mode ──────
    translator.eval()

    # ── Encode and pad English sentences ──
    encoded_eng_sentences = src_vocab.encode(eng_sentences)
    padded_input = pad_sentences(encoded_eng_sentences, pad_idx)
    encoder_input = padded_input.to(device) 

    # ── Initial decoder input: <BOS> ──
    batch_size = len(eng_sentences)
    decoder_input = torch.full(
        size = (batch_size, 1),
        dtype=torch.long,
        device = device,
        fill_value = tgt_vocab.BOS_IDX,
    )
    
    print(f"Encoder Input: {encoder_input}")
    print(f"Decoder Input: {decoder_input}")
    
    with torch.inference_mode():
        for idx in range(len(encoder_input)):
            words:List[str] = []
            
            # Current sentence
            one_encoder_input = encoder_input[idx][None,:]
            one_decoder_input = decoder_input[idx][None,:]
            
            for _ in range(max_len):                
                # ── Predict next token ──
                logits = translator(one_encoder_input, one_decoder_input) #[B, T, V]
                """
                where:
                    B = 1       batch size
                    T = current decoder sequence length
                    V = 150     target vocabulary size
                """
                # Last decoder position
                predicted_token_id = torch.argmax(  
                    logits[:, -1, :], # -1     → take the LAST decoder position → [B, V]
                    dim=-1
                )
        
                # ── Stop at <EOS> ──
                if predicted_token_id.item() == tgt_vocab.EOS_IDX:
                    print(f"\nEnglish: {eng_sentences[idx]}")
                    print(f"Arabic:  {' '.join(words)}")
                    print("-" * 120)     
                    break
        
                # ── Convert token ID to word ──
                word = tgt_vocab.idx_to_word[predicted_token_id.item()]
                words.append(word)
        
                # ── Append predicted token ──
                # Take the token I just predicted and append it to the decoder's input sequence and fix shapes.
                one_decoder_input = torch.cat(
                    [one_decoder_input, predicted_token_id[:, None]],
                    dim=1
                )
