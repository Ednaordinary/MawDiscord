import math
import os
import random
import time
from inspect import isfunction
from typing import Optional, Callable, List, Type, Any, TypeVar, Union, Tuple, Dict

import librosa
import nltk
import numpy as np
import phonemizer
import torch
import torchaudio
import yaml
from nltk import word_tokenize

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Speech.text_utils import TextCleaner

from speech_models import *
from Utils.PLBERT.util import load_plbert

T = TypeVar("T")


# Almost all of this comes from https://github.com/yl4579/StyleTTS2




def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d

class SpeakerRunner:
    def __init__(self):
        device = "cuda"
        self.device = "cuda"
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        np.random.seed(0)
        nltk.download('punkt')
        self.text_cleaner = TextCleaner()
        to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4

        def preprocess(wave):
            wave_tensor = torch.from_numpy(wave).float()
            mel_tensor = to_mel(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
            return mel_tensor

        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,
                                                                  with_stress=True,
                                                                  words_mismatch='ignore')
        config = yaml.safe_load(open("Models/LJSpeech/config.yml"))
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)
        model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(device) for key in model]
        params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
        params = params_whole['net']
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [model[key].eval() for key in model]

        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )
        self.model = model

    def inference(self, text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                                  embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                                  embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()

    def LFinference(self, text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                             embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                             embedding_scale=embedding_scale).squeeze(0)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy(), s_pred