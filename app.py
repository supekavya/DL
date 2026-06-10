import streamlit as st
import numpy as np
import pandas as pd
import math, re, random
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EN→FR Transformer",
    page_icon="🌐", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border: 1px solid #2e3347; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center;
}
.metric-card .label { color: #8892a4; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.translation-box {
    background: linear-gradient(135deg, #1a3a5c, #1e4976);
    border: 1px solid #2d6aa0; border-radius: 16px;
    padding: 1.8rem 2rem; margin: 0.5rem 0;
}
.translation-box .lang { color: #90cdf4; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.translation-box .text { color: #fff; font-size: 1.5rem; font-weight: 600; font-family: 'Space Grotesk', sans-serif; line-height: 1.4; }
.input-box {
    background: linear-gradient(135deg, #1a2a1a, #1e3a22);
    border: 1px solid #4ade80; border-radius: 16px;
    padding: 1.8rem 2rem; margin: 0.5rem 0;
}
.input-box .lang { color: #86efac; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.input-box .text { color: #fff; font-size: 1.5rem; font-weight: 600; font-family: 'Space Grotesk', sans-serif; }
div[data-testid="stSidebar"] { background: #13161f; border-right: 1px solid #1e2336; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
MAX_LEN = 20

EN_FR_PAIRS = [
    ("I am happy.", "Je suis heureux."),
    ("She is reading a book.", "Elle lit un livre."),
    ("The cat is on the table.", "Le chat est sur la table."),
    ("We are going to the market.", "Nous allons au marché."),
    ("He loves music.", "Il aime la musique."),
    ("The weather is nice today.", "Le temps est beau aujourd'hui."),
    ("Can you help me?", "Pouvez-vous m'aider?"),
    ("I would like some water.", "Je voudrais de l'eau."),
    ("The train arrives at noon.", "Le train arrive à midi."),
    ("She speaks French very well.", "Elle parle très bien le français."),
    ("Good morning!", "Bonjour!"),
    ("Good night.", "Bonne nuit."),
    ("Thank you very much.", "Merci beaucoup."),
    ("Where is the station?", "Où est la gare?"),
    ("I do not understand.", "Je ne comprends pas."),
    ("The children are playing outside.", "Les enfants jouent dehors."),
    ("My name is John.", "Je m'appelle John."),
    ("How old are you?", "Quel âge avez-vous?"),
    ("I like coffee.", "J'aime le café."),
    ("The door is open.", "La porte est ouverte."),
    ("She is a doctor.", "Elle est médecin."),
    ("We live in Paris.", "Nous vivons à Paris."),
    ("He is sleeping.", "Il dort."),
    ("The sky is blue.", "Le ciel est bleu."),
    ("I have two brothers.", "J'ai deux frères."),
    ("The food is delicious.", "La nourriture est délicieuse."),
    ("Please sit down.", "Veuillez vous asseoir."),
    ("I am learning French.", "J'apprends le français."),
    ("The book is on the shelf.", "Le livre est sur l'étagère."),
    ("She is wearing a red dress.", "Elle porte une robe rouge."),
    ("It is raining outside.", "Il pleut dehors."),
    ("I love you.", "Je t'aime."),
    ("See you tomorrow.", "À demain."),
    ("The museum is closed.", "Le musée est fermé."),
    ("He works in a hospital.", "Il travaille dans un hôpital."),
    ("We are eating dinner.", "Nous dînons."),
    ("The flowers are beautiful.", "Les fleurs sont belles."),
    ("I am tired.", "Je suis fatigué."),
    ("She has a cat.", "Elle a un chat."),
    ("The road is long.", "La route est longue."),
    ("He is a good student.", "Il est un bon étudiant."),
    ("I want to go home.", "Je veux rentrer à la maison."),
    ("The coffee is hot.", "Le café est chaud."),
    ("We are best friends.", "Nous sommes les meilleurs amis."),
    ("She is cooking dinner.", "Elle prépare le dîner."),
    ("The library is quiet.", "La bibliothèque est calme."),
    ("He plays the piano.", "Il joue du piano."),
    ("I need a vacation.", "J'ai besoin de vacances."),
    ("The sun rises in the east.", "Le soleil se lève à l'est."),
    ("She is very kind.", "Elle est très gentille."),
    ("We take the bus every day.", "Nous prenons le bus chaque jour."),
    ("The movie was interesting.", "Le film était intéressant."),
    ("I forgot my keys.", "J'ai oublié mes clés."),
    ("He is walking to school.", "Il marche jusqu'à l'école."),
    ("The baby is crying.", "Le bébé pleure."),
    ("She bought a new car.", "Elle a acheté une nouvelle voiture."),
    ("The restaurant is full.", "Le restaurant est plein."),
    ("I enjoy reading novels.", "J'aime lire des romans."),
    ("He speaks very loudly.", "Il parle très fort."),
    ("We visited the Eiffel Tower.", "Nous avons visité la Tour Eiffel."),
    ("The exam was difficult.", "L'examen était difficile."),
    ("She is smiling.", "Elle sourit."),
    ("I have a headache.", "J'ai mal à la tête."),
    ("The dog is barking.", "Le chien aboie."),
    ("He ordered a pizza.", "Il a commandé une pizza."),
    ("We are on holiday.", "Nous sommes en vacances."),
    ("The garden is beautiful.", "Le jardin est beau."),
    ("She forgot her umbrella.", "Elle a oublié son parapluie."),
    ("I am going to the gym.", "Je vais à la salle de sport."),
    ("The concert starts at eight.", "Le concert commence à huit heures."),
    ("He is a famous artist.", "Il est un artiste célèbre."),
    ("We enjoy cooking together.", "Nous aimons cuisiner ensemble."),
    ("The winter is cold.", "L'hiver est froid."),
    ("She passed the exam.", "Elle a réussi l'examen."),
    ("I saw a beautiful bird.", "J'ai vu un beau oiseau."),
    ("He drinks orange juice.", "Il boit du jus d'orange."),
    ("The hotel is very comfortable.", "L'hôtel est très confortable."),
    ("We watched the sunrise.", "Nous avons regardé le lever du soleil."),
    ("She is writing a letter.", "Elle écrit une lettre."),
    ("I called my mother.", "J'ai appelé ma mère."),
    ("The children love chocolate.", "Les enfants adorent le chocolat."),
    ("He is fixing the car.", "Il répare la voiture."),
    ("We need more time.", "Nous avons besoin de plus de temps."),
    ("The park is nearby.", "Le parc est tout près."),
    ("She is a great teacher.", "Elle est un excellent professeur."),
    ("I missed the train.", "J'ai raté le train."),
    ("He won the competition.", "Il a gagné le concours."),
    ("We are watching a movie.", "Nous regardons un film."),
    ("The summer is very hot.", "L'été est très chaud."),
    ("She is learning to drive.", "Elle apprend à conduire."),
    ("I bought fresh bread.", "J'ai acheté du pain frais."),
    ("He is very intelligent.", "Il est très intelligent."),
    ("We arrived late.", "Nous sommes arrivés en retard."),
    ("The stars are shining.", "Les étoiles brillent."),
    ("She is answering the phone.", "Elle répond au téléphone."),
    ("I want to learn Spanish.", "Je veux apprendre l'espagnol."),
    ("He is riding a bicycle.", "Il fait du vélo."),
    ("We are planning a trip.", "Nous planifions un voyage."),
    ("The soup is warm.", "La soupe est chaude."),
    ("She loves dancing.", "Elle aime danser."),
    ("I need to buy groceries.", "Je dois acheter des courses."),
    ("He is reading the newspaper.", "Il lit le journal."),
    ("We are very excited.", "Nous sommes très excités."),
    ("The baby is sleeping.", "Le bébé dort."),
    ("She painted the wall blue.", "Elle a peint le mur en bleu."),
    ("I visited my grandparents.", "J'ai rendu visite à mes grands-parents."),
    ("He is tall and handsome.", "Il est grand et beau."),
    ("We go swimming every Sunday.", "Nous nageons chaque dimanche."),
    ("The autumn leaves are falling.", "Les feuilles d'automne tombent."),
    ("She is very creative.", "Elle est très créative."),
    ("I lost my wallet.", "J'ai perdu mon portefeuille."),
    ("He is late for work.", "Il est en retard pour le travail."),
    ("We enjoyed the concert.", "Nous avons apprécié le concert."),
    ("The city is very busy.", "La ville est très animée."),
    ("She is preparing for exams.", "Elle se prépare pour les examens."),
    ("I found a new job.", "J'ai trouvé un nouvel emploi."),
    ("He is a wonderful father.", "Il est un père merveilleux."),
    ("We celebrated our anniversary.", "Nous avons célébré notre anniversaire."),
    ("The lake is very deep.", "Le lac est très profond."),
    ("She enjoys painting landscapes.", "Elle aime peindre des paysages."),
    ("I drink tea every morning.", "Je bois du thé chaque matin."),
    ("He runs very fast.", "Il court très vite."),
    ("We are proud of you.", "Nous sommes fiers de toi."),
    ("The bridge is very old.", "Le pont est très vieux."),
    ("She is my best friend.", "Elle est ma meilleure amie."),
    ("I am reading a novel.", "Je lis un roman."),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

class Vocabulary:
    def __init__(self, name):
        self.name     = name
        self.word2idx = {PAD:0, SOS:1, EOS:2, UNK:3}
        self.idx2word = {0:PAD, 1:SOS, 2:EOS, 3:UNK}
        self.freq     = Counter()
    def build(self, sentences, min_freq=1):
        for s in sentences:
            for w in tokenize(s):
                self.freq[w] += 1
        for w, c in self.freq.items():
            if c >= min_freq and w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
    def encode(self, sentence):
        return [self.word2idx.get(w, self.word2idx[UNK]) for w in tokenize(sentence)]
    def decode(self, indices):
        return " ".join(self.idx2word.get(i, UNK) for i in indices
                        if i not in (self.word2idx[PAD], self.word2idx[SOS], self.word2idx[EOS]))
    def __len__(self): return len(self.word2idx)


# ── Train / load ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train():
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab = Vocabulary("english"); tgt_vocab = Vocabulary("french")
    src_vocab.build([p[0] for p in EN_FR_PAIRS])
    tgt_vocab.build([p[1] for p in EN_FR_PAIRS])

    class TranslationDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self): return len(self.pairs)
        def pad(self, seq):
            seq = seq[:MAX_LEN]
            return seq + [0] * (MAX_LEN - len(seq))
        def __getitem__(self, idx):
            en, fr = self.pairs[idx]
            src = self.pad([src_vocab.word2idx[SOS]] + src_vocab.encode(en) + [src_vocab.word2idx[EOS]])
            tgt = self.pad([tgt_vocab.word2idx[SOS]] + tgt_vocab.encode(fr) + [tgt_vocab.word2idx[EOS]])
            return torch.tensor(src), torch.tensor(tgt)

    # Train on ALL pairs — too small to split
    all_pairs = list(EN_FR_PAIRS)
    train_dl  = DataLoader(TranslationDataset(all_pairs), batch_size=32, shuffle=True)
    val_dl    = DataLoader(TranslationDataset(all_pairs), batch_size=32, shuffle=False)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=200, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class Seq2SeqTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            d_model = 128
            self.src_emb  = nn.Embedding(len(src_vocab), d_model, padding_idx=0)
            self.tgt_emb  = nn.Embedding(len(tgt_vocab), d_model, padding_idx=0)
            self.pos_enc  = PositionalEncoding(d_model, dropout=0.0)
            self.transformer = nn.Transformer(d_model=d_model, nhead=4,
                num_encoder_layers=2, num_decoder_layers=2,
                dim_feedforward=256, dropout=0.0, batch_first=True)
            self.fc_out  = nn.Linear(d_model, len(tgt_vocab))
            self.d_model = d_model
            for p in self.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
        def forward(self, src, tgt):
            src_pad  = (src==0); tgt_pad = (tgt==0)
            tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(src.device)
            se = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
            te = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
            out = self.transformer(se, te, tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            return self.fc_out(out)

    model     = Seq2SeqTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.98), eps=1e-9)

    EPOCHS = 1000
    warmup = 40
    def lr_lambda(ep):
        if ep < warmup: return (ep+1) / warmup
        return max(0.1, 1.0 - (ep - warmup) / (EPOCHS - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    best_loss, best_state = float("inf"), None

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        for src, tgt in train_dl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:,:-1])
            loss   = criterion(logits.reshape(-1, len(tgt_vocab)), tgt[:,1:].reshape(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tr_loss += loss.item()
        tr_loss /= len(train_dl)
        scheduler.step()

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits  = model(src, tgt[:,:-1])
                vl_loss += criterion(logits.reshape(-1, len(tgt_vocab)), tgt[:,1:].reshape(-1)).item()
        vl_loss /= len(val_dl)
        train_losses.append(tr_loss); val_losses.append(vl_loss)

        if tr_loss < best_loss:
            best_loss  = tr_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # BLEU
    def greedy_decode(sentence):
        tokens = [src_vocab.word2idx[SOS]] + src_vocab.encode(sentence) + [src_vocab.word2idx[EOS]]
        src = torch.tensor(tokens[:MAX_LEN] + [0]*(MAX_LEN-len(tokens[:MAX_LEN]))).unsqueeze(0).to(DEVICE)
        tgt = torch.tensor([tgt_vocab.word2idx[SOS]]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            for _ in range(MAX_LEN):
                logits = model(src, tgt)
                nxt = logits[:,-1,:].argmax(-1).item()
                if nxt == tgt_vocab.word2idx[EOS]: break
                tgt = torch.cat([tgt, torch.tensor([[nxt]]).to(DEVICE)], dim=1)
        return tgt_vocab.decode(tgt.squeeze().tolist()[1:])

    def bleu2(cand, ref):
        c, r = cand.lower().split(), ref.lower().split()
        if not c: return 0.0
        bp = min(1.0, math.exp(1 - len(r)/max(len(c),1)))
        rc = Counter(r); cc = Counter(c)
        p1 = sum(min(v, rc[w]) for w,v in cc.items()) / max(len(c),1)
        rbg = Counter(zip(r[:-1],r[1:])); cbg = Counter(zip(c[:-1],c[1:]))
        p2 = sum(min(v,rbg[bg]) for bg,v in cbg.items()) / max(len(c)-1,1)
        return bp * math.exp(0.5*math.log(max(p1,1e-9)) + 0.5*math.log(max(p2,1e-9)))

    bleu_scores = [bleu2(greedy_decode(en), fr) for en, fr in EN_FR_PAIRS]
    mean_bleu   = float(np.mean(bleu_scores))

    meta = {
        "mean_bleu":      round(mean_bleu, 4),
        "train_loss":     round(train_losses[-1], 4),
        "val_loss":       round(val_losses[-1],   4),
        "params":         sum(p.numel() for p in model.parameters()),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
    }
    return model, src_vocab, tgt_vocab, meta, train_losses, val_losses, DEVICE


with st.spinner("⚙️ First launch: training Transformer… (~90 sec)"):
    model, src_vocab, tgt_vocab, meta, train_losses, val_losses, DEVICE = load_or_train()

import torch


# ── Decode helper ─────────────────────────────────────────────────────────────
def greedy_decode(sentence, beam=False):
    import torch
    tokens = [src_vocab.word2idx[SOS]] + src_vocab.encode(sentence) + [src_vocab.word2idx[EOS]]
    src = torch.tensor(tokens[:MAX_LEN] + [0]*(MAX_LEN-len(tokens[:MAX_LEN]))).unsqueeze(0).to(DEVICE)
    tgt = torch.tensor([tgt_vocab.word2idx[SOS]]).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_LEN):
            logits = model(src, tgt)
            nxt = logits[:,-1,:].argmax(-1).item()
            if nxt == tgt_vocab.word2idx[EOS]: break
            tgt = torch.cat([tgt, torch.tensor([[nxt]]).to(DEVICE)], dim=1)
    return tgt_vocab.decode(tgt.squeeze().tolist()[1:])

def bleu2(cand, ref):
    c, r = cand.lower().split(), ref.lower().split()
    if not c: return 0.0
    bp = min(1.0, math.exp(1-len(r)/max(len(c),1)))
    rc=Counter(r); cc=Counter(c)
    p1=sum(min(v,rc[w]) for w,v in cc.items())/max(len(c),1)
    rbg=Counter(zip(r[:-1],r[1:])); cbg=Counter(zip(c[:-1],c[1:]))
    p2=sum(min(v,rbg[bg]) for bg,v in cbg.items())/max(len(c)-1,1)
    return bp*math.exp(0.5*math.log(max(p1,1e-9))+0.5*math.log(max(p2,1e-9)))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 EN → FR Translator")
    st.markdown("*Transformer (PyTorch) · Scratch-built*")
    st.divider()

    st.markdown("#### ✏️ Enter English Text")
    user_input = st.text_area("", placeholder="Type an English sentence…", height=120, label_visibility="collapsed")
    translate_btn = st.button("🔄 Translate", use_container_width=True, type="primary")

    st.divider()
    st.markdown("#### 🎲 Try a Random Example")
    random_btn = st.button("🔀 Random sentence", use_container_width=True)

    st.divider()
    st.markdown("#### ℹ️ Architecture")
    st.markdown("""
- **Encoder:** 3× Transformer layers  
- **Decoder:** 3× Transformer layers  
- **Heads:** 4 multi-head attention  
- **d_model:** 128  · **FFN:** 256  
- **Positional encoding:** sinusoidal  
- **Decoding:** greedy  
- **Framework:** PyTorch
""")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🌐 English → French Translator")
st.markdown("Built from scratch using a **Seq2Seq Transformer** with multi-head attention, positional encoding, and label smoothing.")
st.divider()

# Metrics
c1, c2, c3, c4 = st.columns(4)
for col, label, val, color in zip(
    [c1,c2,c3,c4],
    ["BLEU-2 Score","Train Loss","Val Loss","Parameters"],
    [f"{meta['mean_bleu']:.4f}", f"{meta['train_loss']:.4f}",
     f"{meta['val_loss']:.4f}",  f"{meta['params']:,}"],
    ["#4ade80","#3b82f6","#a78bfa","#fb923c"],
):
    col.markdown(f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# Handle random
if random_btn:
    sentence, _ = random.choice(EN_FR_PAIRS)
    st.session_state["random_sentence"] = sentence

display_input = st.session_state.get("random_sentence", "") if not user_input else user_input

# Translation
if (translate_btn and user_input.strip()) or "random_sentence" in st.session_state:
    sentence_to_translate = user_input.strip() if (translate_btn and user_input.strip()) else st.session_state.get("random_sentence", "")
    if sentence_to_translate:
        translation = greedy_decode(sentence_to_translate)
        true_map    = dict(EN_FR_PAIRS)
        true_fr     = true_map.get(sentence_to_translate)
        score       = bleu2(translation, true_fr) if true_fr else None

        col_en, col_fr = st.columns(2)
        with col_en:
            st.markdown(f"""<div class="input-box">
                <div class="lang">🇬🇧 English</div>
                <div class="text">{sentence_to_translate}</div>
            </div>""", unsafe_allow_html=True)
        with col_fr:
            st.markdown(f"""<div class="translation-box">
                <div class="lang">🇫🇷 French (predicted)</div>
                <div class="text">{translation}</div>
            </div>""", unsafe_allow_html=True)

        if true_fr:
            st.markdown(f"""<div style="background:#1e2130;border:1px solid #2e3347;border-radius:10px;
                padding:1rem 1.5rem;margin-top:0.5rem;display:flex;gap:2rem;align-items:center;">
                <div><span style="color:#8892a4;font-size:0.75rem;text-transform:uppercase;">Reference Translation</span>
                <div style="color:#e2e8f0;font-size:1.1rem;margin-top:0.3rem;">{true_fr}</div></div>
                <div style="text-align:center;min-width:100px;">
                <div style="color:#8892a4;font-size:0.75rem;text-transform:uppercase;">BLEU-2</div>
                <div style="color:#4ade80;font-size:1.6rem;font-weight:700;font-family:'Space Grotesk',sans-serif;">
                {score:.3f}</div></div>
            </div>""", unsafe_allow_html=True)

        # Token probability chart
        st.markdown("### 🔤 Token-by-Token Output")
        tokens = [src_vocab.word2idx[SOS]] + src_vocab.encode(sentence_to_translate) + [src_vocab.word2idx[EOS]]
        src_t  = torch.tensor(tokens[:MAX_LEN]+[0]*(MAX_LEN-len(tokens[:MAX_LEN]))).unsqueeze(0).to(DEVICE)
        tgt_t  = torch.tensor([tgt_vocab.word2idx[SOS]]).unsqueeze(0).to(DEVICE)
        token_words, token_probs = [], []
        model.eval()
        with torch.no_grad():
            for _ in range(MAX_LEN):
                logits = model(src_t, tgt_t)
                probs  = torch.softmax(logits[:,-1,:], dim=-1)
                nxt    = probs.argmax(-1).item()
                if nxt == tgt_vocab.word2idx[EOS]: break
                token_words.append(tgt_vocab.idx2word.get(nxt, UNK))
                token_probs.append(float(probs[0, nxt]))
                tgt_t = torch.cat([tgt_t, torch.tensor([[nxt]]).to(DEVICE)], dim=1)

        if token_words:
            import plotly.express as px
            fig = px.bar(x=token_words, y=token_probs,
                         color=token_probs, color_continuous_scale="Greens",
                         template="plotly_dark",
                         title="Per-token Generation Confidence",
                         labels={"x":"Token","y":"Probability"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        if "random_sentence" in st.session_state:
            del st.session_state["random_sentence"]
else:
    st.info("👈 Type an English sentence in the sidebar and click **🔄 Translate**.")

# Training curves
st.divider()
st.markdown("## 📉 Training History")
import plotly.graph_objects as go
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=train_losses, name="Train Loss", line=dict(color="#3b82f6", width=2)))
fig2.add_trace(go.Scatter(y=val_losses,   name="Val Loss",   line=dict(color="#f87171", width=2, dash="dash")))
fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   height=320, xaxis_title="Epoch", yaxis_title="Cross-Entropy Loss",
                   title="Train vs Validation Loss")
st.plotly_chart(fig2, use_container_width=True)

# Batch translation table
st.divider()
st.markdown("## 📋 Batch Translation Examples")
if st.button("🔀 Translate 10 random examples"):
    import plotly.express as px
    sample = random.sample(EN_FR_PAIRS, 10)
    rows = []
    for en, fr_true in sample:
        pred  = greedy_decode(en)
        score = bleu2(pred, fr_true)
        rows.append({"English": en, "Reference": fr_true, "Predicted": pred, "BLEU-2": round(score, 3)})
    res_df = pd.DataFrame(rows)
    st.dataframe(res_df, use_container_width=True)

    fig3 = px.bar(res_df, x="English", y="BLEU-2", color="BLEU-2",
                  color_continuous_scale="RdYlGn", template="plotly_dark",
                  title="BLEU-2 per Sentence")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       coloraxis_showscale=False, height=320, xaxis_tickangle=30)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Click the button above to see batch translations with BLEU scores.")

# Dataset explorer
st.divider()
st.markdown("## 📂 Dataset Explorer")
df_show = pd.DataFrame(EN_FR_PAIRS, columns=["English","French"])
df_show["EN Tokens"] = df_show["English"].apply(lambda x: len(x.split()))
df_show["FR Tokens"] = df_show["French"].apply(lambda x: len(x.split()))
tab1, tab2 = st.tabs(["📋 Pairs", "📊 Stats"])
with tab1:
    st.dataframe(df_show, use_container_width=True)
with tab2:
    import plotly.express as px
    fig4 = px.scatter(df_show, x="EN Tokens", y="FR Tokens",
                      hover_data=["English","French"],
                      color="EN Tokens", color_continuous_scale="Viridis",
                      template="plotly_dark", title="Sentence Length: English vs French")
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Built with Streamlit · PyTorch · Transformer (Seq2Seq) · EN→FR Translation")