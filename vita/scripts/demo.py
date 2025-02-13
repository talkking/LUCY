import os
import torch
import json
import transformers
import einops
import random
import numpy as np
import soundfile as sf
import vita.util.data_util as data_util
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from vita.model.language_model.vita_qwen2 import VITAQwen2ForCausalLM
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from snac import SNAC
from time import time
from itertools import groupby
from vita.util.sampling import sample
from vita.scripts.train_v2 import ModelArguments
from vita.util import conversation as conversation_lib, move_to_cuda, move_to_cpu
from vita.constants import IGNORE_INDEX, AUDIO_PH, PAD_TOKEN, EMOTION_SP

audio_num_codebook = 7
text_vocab_size = 152064
text_vocab_size_padded = text_vocab_size + 64
qwen2_1p5b_text_vocab_size = 151936
qwne2_1p5b_text_vocab_size_padded = qwen2_1p5b_text_vocab_size + 64
audio_vocab_size_padded = 4160
EOA    = 4096
PAD_A  = 4097
BOA    = 4098
ANS_A  = 4099
F10    = 4103
M29    = 4104
PAD_T  = 151937 - qwen2_1p5b_text_vocab_size + text_vocab_size
EOT    = 151936 - qwen2_1p5b_text_vocab_size + text_vocab_size
IM_END = 151645
FC_TOKEN = 27
NEUTRAL_TOKEN = 151648
JOY_TOKEN     = 151649
SADNESS_TOKEN = 151650
FEAR_TOKEN    = 151651
ANGER_TOKEN   = 151652
SUPRISE_TOKEN = 151653
DISGUST_TOKEN = 151654
SORRY_TOKEN   = 151655

NEUTRAL = "<|Neutral|>"
JOY     = "<|Joy|>"
SADNESS = "<|Sadness|>"
FEAR    = "<|Fear|>"
ANGER   = "<|Anger|>"
SUPRISE = "<|Surprise|>"
DISGUST = "<|Disgust|>"
SORRY   = "<|Sorry|>"



TIRQ         = "<|tirq|>"         # 151656 text interrupt
AIRQ_DENIAL  = "<|airq_denial|>"  # 151657 audio interrupt: denial and discontent
AIRQ_INQUIRY = "<|airq_inquiry|>" # 151658 audio interrupt: further inquiry
AIRQ_CHANGE  = "<|airq_change|>"  # 151659 audio interrupt: change topic
ANEG_AFFIRM  = "<|airq_affirm|>"  # 151660 audio negative interrupt: affirmative acknowledgement
ANEG_NOISE   = "<|airq_noise|>"   # 151661 audio negative interrupt:background noise
FC_START     = "<function="       # 151683
FC_END       = "</function>"      # 141684

STATE_TOKENS = [
    TIRQ,
    AIRQ_DENIAL,
    AIRQ_INQUIRY,
    AIRQ_CHANGE,
    ANEG_AFFIRM,
    ANEG_NOISE,
]

audio_encoder_type="whale"

@dataclass
class InferenceArguments:
    max_code_length: Optional[int] = field(default=None)
    snac_sr: Optional[int] = field(default=24000)
    snac_model: Optional[str] = field(default="hubertsiuzdak/snac_24khz")
    output_path: Optional[str] = field(default=None)
    save_audio: Optional[bool] = field(default=True)
    output_text_only: Optional[bool] = field(default=False)

def next_token(
    model, 
    audios=None,
    attention_mask=None,
    input_ids=None,
    audio_lengths=None,
    audio_attention_mask=None,
    past_key_values=None,
    audio_feature_lengths=None,
    state_start=None,
    state_end=None,
    max_input_length=1500,
    **kwargs,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        audios=audios,
        audio_lengths=audio_lengths, 
        audio_feature_lengths=audio_feature_lengths,
        audio_attention_mask=audio_attention_mask,
        past_key_values=past_key_values,
        use_cache=True, infer=True, max_input_length=max_input_length
    )
    batch_size = input_ids.shape[0]
    assert batch_size == 1 or batch_size == 2, batch_size
    # if batch size is 2, use first item to predict audio codec and use second item to predict text
    assert state_start is not None
    assert state_end is not None
    logits_t = outputs.logits[-1:,:,:text_vocab_size_padded] # last item in batch
    # logits_t = outputs.logits[:1,:,:text_vocab_size_padded] # first item in batch

    next_t = sample(logits_t, top_k=2).to(input_ids[0]).repeat(batch_size).unsqueeze(-1) # B x 1

    next_a, next_ua = [], [] # layer shifted/unshifted audio tokens

    for i in range(audio_num_codebook):
        start = text_vocab_size_padded + i * audio_vocab_size_padded
        end = text_vocab_size_padded + (i+1) * audio_vocab_size_padded
        logits_a_i = outputs.logits[:1, :,start:end]
        ua_i = input_ids.new_zeros(batch_size,1).fill_(PAD_A)
        ua_i[:1, :] = sample(logits_a_i, top_k=5) # B x 1 # first item in batch
        a_i = codec_layer_shift(ua_i, i) # B x 1
        next_a.append(a_i)
        next_ua.append(ua_i)
    
    next_a = torch.cat(next_a, dim=-1) # B x 7
    next_ua = torch.cat(next_ua, dim=-1) # B x 7
    logits_s = outputs.logits[-1:,:,state_start:state_end]
    state_id = sample(logits_s, top_k=1)
    state = STATE_TOKENS[state_id]
    past_key_values = outputs.past_key_values
    return next_t, next_a, next_ua, past_key_values, state

def decode_audio(snac, audio_codes_padded):
    T, N = audio_codes_padded.shape # length of auido codes and number of codebooks
    audio_codes = torch.zeros((T-N-1, N)).to(audio_codes_padded) # 1 for EOA
    for i in range(N):
        audio_codes[:,i] = audio_codes_padded[i+1:-(N-i), i]
    # print(audio_codes)
    (
        code_12hz, code_24hz, code_48hz
    ) = (
        audio_codes[:,0:1], 
        audio_codes[:,1:3],
        audio_codes[:,3:]
    )
    codes = [
        code_12hz.reshape(1, -1), 
        code_24hz.reshape(1, -1), 
        code_48hz.reshape(1, -1)
    ]
    audio = snac.decode(codes).view(-1)
    return audio


def load_wav(wavpath, sample_rate=16_000):
    wavpaths = [wavpath] if type(wavpath) is not list else wavpath
    
    wavs = []
    for i, wdata in enumerate(wavpaths):
        if type(wdata) is dict:
            wpath, start, end, audio_length = \
                wdata["wavpath"], wdata["start"], wdata["end"], wdata["audio_length"]
        else:
            wpath, start, end, audio_length = wdata, 0, None, None
        wav, sr = sf.read(wpath, start=start, stop=end)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert sr == sample_rate, f"Audio sampling rate {sr} != {sample_rate}"
        assert audio_length is None or len(wav) == audio_length, \
            f"Audio length {len(wav)} != {audio_length} of {wpath} with start {start} and end {end}"
        assert end is None or (end - start == audio_length), \
            f"Audio length {audio_length} != end {end} - start {start}"
        if i > 0:
            interval = random.uniform(0.35, 0.75)
            si_leng = int(interval * sample_rate)
            silence = np.zeros(si_leng)
            wavs.append(silence)
        wavs.append(wav)
    wav_cat = np.concatenate(wavs)
    wav_cat = torch.from_numpy(wav_cat).float().unsqueeze(0)
    return wav_cat, sr

def load_wav_feat(wavpaths, audio_processor, sample_rate=16_000, audio_feature_rate=50):
    wav, sr = load_wav(wavpaths)
    assert sr == sample_rate, f"{sr} != {sample_rate}"
    if audio_encoder_type == "whisper":
        wav = wav[0]
        audio_length = len(wav)
        audio = audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features
        audio_length = int(audio_length / sample_rate * audio_feature_rate) + 1
    elif audio_encoder_type == "whale":
        audio, audio_length = audio_processor.process(waveform=wav, sample_rate=sr)
    return audio, audio_length

def codec_layer_shift(input_id, layer):
    return input_id + text_vocab_size_padded + layer * audio_vocab_size_padded

def prepare_inputs_whisper(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
    ):
    shifted_PAD_A = torch.LongTensor([codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)])
    shifted_BOA   = torch.LongTensor([codec_layer_shift(BOA, i)   for i in range(audio_num_codebook)])
    shifted_EOA   = torch.LongTensor([codec_layer_shift(EOA, i)   for i in range(audio_num_codebook)])
    shifted_ANS_A = torch.LongTensor([codec_layer_shift(ANS_A, i) for i in range(audio_num_codebook)])
    shifted_F10   = torch.LongTensor([codec_layer_shift(F10, i)   for i in range(audio_num_codebook)])
    shifted_M29   = torch.LongTensor([codec_layer_shift(M29, i)   for i in range(audio_num_codebook)])
    AUDIO_PH_idx  = tokenizer.convert_tokens_to_ids(AUDIO_PH)
    PAD_TOKEN_idx = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    conv = conversation_lib.conv_qwen2.copy()
    conv.messages = []

    audios, audio_lengths = torch.zeros([0,80,3000]), torch.zeros([0]).long()
    if past_input_dict is not None:
        audio_lengths = past_input_dict["audio_lengths"]
        num_audio = len(audio_lengths) // 2
        audio_lengths = audio_lengths[:num_audio]
        audios = past_input_dict["audios"][:num_audio]
    has_audio_input = "wavpath" in source

    if has_audio_input and use_audio_input:
        audio, audio_length = load_wav_feat(source["wavpath"], audio_processor)
        message = AUDIO_PH * (audio_length + 2)
        audios = torch.cat([audios, audio])
        audio_lengths = torch.cat([
            audio_lengths, torch.LongTensor([audio_length])
        ])
        state = AIRQ_INQUIRY

    else:
        message = source["content"]
        state = TIRQ
    role = source["role"]
    speaker = source.get("speaker", "ANS_A")
    speaker = "M29"
    if add_system_prompt:  
        if system_prompt:
            conv.system = system_prompt
        conv.append_message(role, message)
        prompt = conv.get_prompt()
        print(prompt)
    else:
        prompt = f"<|im_start|>{role}\n{message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if past_input_dict is not None:
        input_ids = torch.cat([
            past_input_dict["input_ids"][0,:,-1], generated, input_ids]
        )

    input_codec = input_ids.new_zeros([len(input_ids), audio_num_codebook]).fill_(IGNORE_INDEX) # T x 7
    input_codec[:,:] = shifted_PAD_A[None,:]

    i_chunk, start, end = 0, 0, 0
    audio_attention_mask = input_ids == AUDIO_PH_idx
    for is_placeholder, chunk in groupby(audio_attention_mask.clone()):
        chunk_length = len(list(chunk))
        assert chunk_length > 2 # chunk has at least 1 BOA, 1 EOA, and 1 audio token
        end += chunk_length
        if is_placeholder:
            assert chunk_length == audio_lengths[i_chunk] + 2
            input_codec[start] = shifted_BOA
            input_codec[end-1] = shifted_EOA
            audio_attention_mask[[start,end-1]] = False
            i_chunk += 1
        start = end
    input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
    batched_input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1) # 2 x T x 8
    speaker_token = eval(f"shifted_{speaker}")
    batched_input_ids[0, -1, :-1] = speaker_token # the last position of the first item in the batch is ANS_A
    batched_audio_attention_mask = audio_attention_mask.unsqueeze(0).expand(2, -1) # 2 x T
    audio_lengths = audio_lengths.repeat(2) 
    attention_mask = batched_input_ids[...,-1].ne(PAD_TOKEN_idx)
    assert attention_mask.all()
    
    audios = torch.cat([audios, audios]) 
    state_start = tokenizer.convert_tokens_to_ids(TIRQ)
    state_end = state_start + len(STATE_TOKENS) 
    input_dict = {
        "input_ids": batched_input_ids,
        "labels": None,
        "attention_mask": attention_mask, 
        "audios": audios,
        "audio_lengths": audio_lengths,
        "audio_attention_mask": batched_audio_attention_mask,
        "state_start": state_start,
        "state_end": state_end,
        "default_state": state,
        "max_input_length": 1e10,
        "infer": True
    }
    return input_dict


def prepare_inputs_whale(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
    ):
    shifted_PAD_A = torch.LongTensor([codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)])
    shifted_BOA   = torch.LongTensor([codec_layer_shift(BOA, i)   for i in range(audio_num_codebook)])
    shifted_EOA   = torch.LongTensor([codec_layer_shift(EOA, i)   for i in range(audio_num_codebook)])
    shifted_ANS_A = torch.LongTensor([codec_layer_shift(ANS_A, i) for i in range(audio_num_codebook)])
    shifted_F10   = torch.LongTensor([codec_layer_shift(F10, i)   for i in range(audio_num_codebook)])
    shifted_M29   = torch.LongTensor([codec_layer_shift(M29, i)   for i in range(audio_num_codebook)])
    AUDIO_PH_idx  = tokenizer.convert_tokens_to_ids(AUDIO_PH)
    PAD_TOKEN_idx = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    conv = conversation_lib.conv_qwen2.copy()
    conv.messages = []

    H = 80
    audios, audio_lengths, audio_feature_lengths = torch.zeros(0, H), torch.zeros([0]).long(), torch.zeros([0]).long()
    if past_input_dict is not None:
        audio_lengths = past_input_dict["audio_lengths"]
        num_audio = len(audio_lengths) // 2
        audio_lengths = audio_lengths[:num_audio]
        audios = past_input_dict["audios"][:num_audio]
        audio_feature_lengths = past_input_dict["audio_feature_lengths"]
        audio_feature_lengths = audio_feature_lengths[:num_audio]
    has_audio_input = "wavpath" in source

    if has_audio_input and use_audio_input:
        audio, audio_length = load_wav_feat(source["wavpath"], audio_processor)
        audio_feature_length = len(audio)
        message = AUDIO_PH * (audio_length + 2)
        audio_lengths = torch.cat([
            audio_lengths, torch.LongTensor([audio_length])
        ])

        audio_feature_lengths = torch.cat([
            audio_feature_lengths, 
            torch.LongTensor([audio_feature_length])
        ])
        B = len(audio_feature_lengths) # 1 for new sample
        T = audio_feature_lengths.max()
        new_audios = torch.zeros(B, T, H)
        for i, (a, al) in enumerate(zip(audios, audio_feature_lengths[:B-1])):
            new_audios[i, :al] = a[:al]
        new_audios[-1, :audio_feature_length] = audio

        audios = new_audios
        state = AIRQ_INQUIRY
    else:
        message = source["content"]
        state = TIRQ
    role = source["role"]
    speaker = source.get("speaker", "ANS_A")
    speaker = "M29"
    speaker = "F10"
    if add_system_prompt:  
        if system_prompt:
            conv.system = system_prompt
        conv.append_message(role, message)
        prompt = conv.get_prompt()
        print(prompt)
    else:
        prompt = f"<|im_start|>{role}\n{message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if past_input_dict is not None:
        input_ids = torch.cat([
            past_input_dict["input_ids"][0,:,-1], generated, input_ids]
        )

    input_codec = input_ids.new_zeros([len(input_ids), audio_num_codebook]).fill_(IGNORE_INDEX) # T x 7
    input_codec[:,:] = shifted_PAD_A[None,:]

    i_chunk, start, end = 0, 0, 0
    audio_attention_mask = input_ids == AUDIO_PH_idx
    for is_placeholder, chunk in groupby(audio_attention_mask.clone()):
        chunk_length = len(list(chunk))
        assert chunk_length > 2 # chunk has at least 1 BOA, 1 EOA, and 1 audio token
        end += chunk_length
        if is_placeholder:
            assert chunk_length == audio_lengths[i_chunk] + 2
            input_codec[start] = shifted_BOA
            input_codec[end-1] = shifted_EOA
            audio_attention_mask[[start,end-1]] = False
            i_chunk += 1
        start = end
    input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
    batched_input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1) # 2 x T x 8
    speaker_token = eval(f"shifted_{speaker}")
    batched_input_ids[0, -1, :-1] = speaker_token # the last position of the first item in the batch is ANS_A
    batched_audio_attention_mask = audio_attention_mask.unsqueeze(0).expand(2, -1) # 2 x T
    audio_lengths = audio_lengths.repeat(2) 
    audio_feature_lengths = audio_feature_lengths.repeat(2)
    attention_mask = batched_input_ids[...,-1].ne(PAD_TOKEN_idx)
    assert attention_mask.all()
    
    audios = torch.cat([audios, audios]) 
    state_start = tokenizer.convert_tokens_to_ids(TIRQ)
    state_end = state_start + len(STATE_TOKENS) 
    input_dict = {
        "input_ids": batched_input_ids,
        "labels": None,
        "attention_mask": attention_mask, 
        "audios": audios,
        "audio_lengths": audio_lengths,
        "audio_feature_lengths": audio_feature_lengths,
        "audio_attention_mask": batched_audio_attention_mask,
        "state_start": state_start,
        "state_end": state_end,
        "default_state": state,
        "max_input_length": 1e10,
        "infer": True
    }
    return input_dict

def prepare_inputs(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
):
    if audio_encoder_type == "whale":
        input_dict = prepare_inputs_whale(
            source, use_audio_input, tokenizer, audio_processor, add_system_prompt, system_prompt, past_input_dict, generated
        )
    elif audio_encoder_type == "whisper":
        input_dict = prepare_inputs_whisper(
            source, use_audio_input, tokenizer, audio_processor, add_system_prompt, system_prompt, past_input_dict, generated
        )
    return input_dict

def get_past_kv(past_kv, index):
    B = 2
    ix = (B + index) % B
    past_kv_i = tuple([tuple([x[ix:ix+1] for x in l]) for l in past_kv])
    return past_kv_i

def repeat_past_kv(past_kv, n):
    past_kv_n = tuple([tuple([x.repeat(n,1,1,1) for x in l]) for l in past_kv])
    return past_kv_n
    
def is_emotion(token):
    return NEUTRAL_TOKEN <= token[-1] <= SORRY_TOKEN

def batch_parallel_decode(model, tokenizer, input_dict, infer_args, device, emotion=None):
    audio_pads_shifted = torch.LongTensor([
        codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)
    ]).to(device)
    text_pad = torch.LongTensor([PAD_T]).to(device)
    text_ends = False
    audio_ends = False
    content_ends = False
    audio_num_layer_ends = -1
    audio_tokens, text_tokens = [], []
    modality = "both"
    state_start, state_end = input_dict["state_start"], input_dict["state_end"]
    state_predicted = None
    for t in tqdm(range(infer_args.max_code_length)):
        if not infer_args.save_audio and text_ends:
            break
        if audio_num_layer_ends == audio_num_codebook and text_ends:
            break
        next_t, next_a, next_ua, past_kv, state = next_token(
            model, **input_dict
        )
        if t == 0:
            print("state:", state)
            state_predicted = state
        if t == 0 and state in [ANEG_AFFIRM, ANEG_NOISE]:
            print("negative state:", state)
            break
        if t == 0 and emotion is not None:
            next_t[:] = emotion
        _t = max(t-1, 0) if is_emotion(next_t) else t
        # past_kv (num_layer x (2 x [B, 2, T, 128]) )
        if _t < audio_num_codebook:
            num_pad = audio_num_codebook - _t
            next_a[0,-num_pad:] = audio_pads_shifted[-num_pad:]
            next_ua[0,-num_pad:] = PAD_A
        if modality in ["both", "audio"] and (text_ends or content_ends):
            next_t[0] = text_pad
        if modality in ["both", "audio"] and audio_ends:
            next_a[0,:audio_num_layer_ends] = audio_pads_shifted[:audio_num_layer_ends]
            next_ua[0,:audio_num_layer_ends] = PAD_A
            audio_num_layer_ends += 1
        if modality in ["text"]:
            next_a[0,:] = audio_pads_shifted[:]
            next_ua[0,:] = PAD_A


        audio_num_layer_ends = min(audio_num_layer_ends, audio_num_codebook)
            
        if modality in ["both", "audio"] and audio_num_layer_ends <= audio_num_codebook:
            audio_tokens.append(next_ua[0])

        function_triggered = (torch.stack(text_tokens) == FC_TOKEN).sum() if len(text_tokens) > 0 else 0
        
        if next_t[-1] == FC_TOKEN and not content_ends:
            next_t[0] = IM_END
        if len(text_tokens) > 0 and text_tokens[-1] == FC_TOKEN and function_triggered == 1:
            next_t[0] = EOT
        if modality == "both" and len(text_tokens) > 0 and text_tokens[-1] == IM_END:
            next_t[:] = EOT
            # save past_key_values of second item and retain only first item in the batch
            next_t = next_t[:1]
            next_a = next_a[:1]
            past_kv = get_past_kv(past_kv, 0)
            modality = "audio"
        if modality == "both" and audio_tokens[-1][-1] == EOA and audio_num_layer_ends == audio_num_codebook:
            next_t = next_t[-1:]
            next_a = next_a[-1:]
            past_kv = get_past_kv(past_kv, -1)
            modality = "text"

        if modality in ["text"] and text_tokens[-1] == IM_END:
            next_t[:] = EOT


        text_tokens.append(next_t[-1])
        if next_t[-1] == FC_TOKEN:
            content_ends = True
        if next_t[-1] == EOT:
            text_ends = True
        if next_ua[0,0] == EOA:
            audio_ends = True
            audio_num_layer_ends = 1
        next_input_ids = torch.cat([next_a, next_t], dim=-1)
        batch_size = next_input_ids.shape[0]
        if infer_args.output_text_only:
            next_input_ids = torch.cat([audio_pads_shifted.unsqueeze(0).repeat(batch_size, -1), next_t], dim=-1)
        next_input_ids = next_input_ids.view(batch_size,1,audio_num_codebook+1)
        input_dict = {
            "input_ids": next_input_ids,
            "past_key_values": past_kv,
            "state_start": state_start,
            "state_end": state_end,
            "max_input_length": 1e10,
            "infer": True,
            
        }
        if not text_ends:
            current_text = tokenizer.decode(torch.cat(text_tokens)) 
            print(current_text)
    text_tokens = torch.cat(text_tokens) if len(text_tokens) > 0 else None
    audio_tokens = torch.stack(audio_tokens) if len(audio_tokens) > 0 else None
    return text_tokens, audio_tokens, state_predicted
    
def get_audio_encoder_type(audio_encoder):
    if "whisper" in audio_encoder.lower():
        audio_encoder_type = "whisper" 
    elif "audio-encoder-qwen2-7b-instruct" in audio_encoder.lower():
        audio_encoder_type = "whale"
    elif "audio-encoder-qwen2.5-7b" in audio_encoder.lower():
        audio_encoder_type = "whale"
    else:
        raise ValueError(f"Unknown encoder type {model_args.audio_encoder}")
    return audio_encoder_type

@torch.inference_mode()
def demo(conversations, use_audio_input=True):
    
    print("use_audio_input", use_audio_input)
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    data_util.sync_data_args(model_args, data_args)
    global audio_encoder_type
    audio_encoder_type = get_audio_encoder_type(model_args.audio_encoder)
    print(model_args)
    print(data_args)
    print(infer_args)
    device = torch.cuda.current_device()
    model = VITAQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    audio_processor = model.get_audio_encoder().audio_processor
    snac = SNAC.from_pretrained(infer_args.snac_model, cache_dir=model_args.cache_dir).eval().to(device)

    global \
        NEUTRAL_TOKEN, \
        JOY_TOKEN, \
        SADNESS_TOKEN, \
        FEAR_TOKEN, \
        ANGER_TOKEN, \
        SUPRISE_TOKEN, \
        DISGUST_TOKEN, \
        SORRY_TOKEN, \
        FC_ST_TOKEN, FC_END_TOKEN, \
        FC_TOKEN
    NEUTRAL_TOKEN = tokenizer.convert_tokens_to_ids(NEUTRAL)
    JOY_TOKEN     = tokenizer.convert_tokens_to_ids(JOY)
    SADNESS_TOKEN = tokenizer.convert_tokens_to_ids(SADNESS)
    FEAR_TOKEN    = tokenizer.convert_tokens_to_ids(FEAR)
    ANGER_TOKEN   = tokenizer.convert_tokens_to_ids(ANGER)
    SUPRISE_TOKEN = tokenizer.convert_tokens_to_ids(SUPRISE)
    DISGUST_TOKEN = tokenizer.convert_tokens_to_ids(DISGUST)
    SORRY_TOKEN   = tokenizer.convert_tokens_to_ids(SORRY)
    FC_ST_TOKEN   = tokenizer.convert_tokens_to_ids(FC_START)
    FC_END_TOKEN  = tokenizer.convert_tokens_to_ids(FC_END)
    FC_TOKEN      = FC_ST_TOKEN
    emotion = FEAR_TOKEN
    emotion = None
    total, correct = 0, 0
    for k, conversation in enumerate(conversations):
        texts = ""
        past_input_dict, generated = None, None
        system_prompt = None
        if conversation[0]["role"] == "system":
            system_prompt = conversation[0]["content"]
            system_prompt = None
            conversation = conversation[1:]
        for i, source in enumerate(conversation):
            if source["role"] not in ["user", "observation"]:
                texts += f"ground_truth:\n{source['content']}\n\n\n"
                continue
            
            add_system_prompt = i == 0
            t0 = time()
            input_dict = prepare_inputs(source, use_audio_input, tokenizer, audio_processor, add_system_prompt, system_prompt, past_input_dict, generated)
            state = conversation[i+1].get("state", input_dict["default_state"])
            input_dict = move_to_cuda(input_dict, device)
            text_tokens, audio_tokens, state_predicted = batch_parallel_decode(model, tokenizer, input_dict, infer_args, device, emotion=emotion)
            
            texts += f"user/obs:\n{source['content']}\nstate:\n{state}\nstate_predicted\n{state_predicted}\n\n\n"

            if infer_args.save_audio and audio_tokens is not None:
                wav = decode_audio(snac, audio_tokens).cpu().numpy().reshape(-1)
                sf.write(f'{infer_args.output_path}/{k}-{i}.wav', wav, infer_args.snac_sr)
                t1 = time()
                gen_time = t1 - t0
                wav_dur = len(wav) / infer_args.snac_sr
                print(f"Used {gen_time:.4f}s to generate {wav_dur:.4f}s audio with RTF: {gen_time/wav_dur}")

            text = tokenizer.decode(text_tokens) if text_tokens is not None else "..."
            texts += f"model_output:\n{text.strip()}\n\n\n"

            total += 1
            correct += state == state_predicted
            if state_predicted in [ANEG_NOISE, ANEG_AFFIRM]:
                break

            past_input_dict = move_to_cpu(input_dict)
            generated = text_tokens[(text_tokens!=PAD_T)&(text_tokens!=EOT)].cpu()

        print(f"accuracy: {correct}/{total}={correct/total}")
        with open(f"{infer_args.output_path}/hyp_{k}.txt", "w", encoding='utf-8') as f:
            f.write(texts)
            
def load_conversations(data_json, num=20, with_negative=False):
    with open(data_json) as f:
        convs = json.load(f)
    convs_subset = []
    import random
    random.seed(42)
    random.shuffle(convs)
    for i, _conv in enumerate(convs[:num]):
        conv = _conv if "conversations" not in _conv else _conv["conversations"]
        wavpath = negative_paths[i]
        convs_subset.append(conv)
    return convs_subset

if __name__ == "__main__":
    json_data = "manifest/AudioQA-1M/eval.json"
    conversations = load_conversations(json_data, num=7, with_negative=False)
    demo(conversations)
