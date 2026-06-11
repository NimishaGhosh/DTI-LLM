import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
device = "cuda" if torch.cuda.is_available() else "cpu"


esm_model_name = "ESM-2"
chem_model_name = "ChemBERTa"

esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name).to(device).eval()

chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
chem_model = AutoModel.from_pretrained(chem_model_name).to(device).eval()

def simple_seq_similarity(seq1, seq2):
    return SequenceMatcher(None, seq1, seq2).ratio()


def get_batch_embeddings(texts, tokenizer, model, batch_size=16):
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)



class ProjectionModel(nn.Module):
    def __init__(self, d_dim, p_dim, proj_dim=256):
        super().__init__()
        self.proj_d = nn.Linear(d_dim, proj_dim)
        self.proj_p = nn.Linear(p_dim, proj_dim)

    def forward(self, d, p):
        d = F.normalize(self.proj_d(d), dim=1)
        p = F.normalize(self.proj_p(p), dim=1)
        return (d * p).sum(dim=1)  # cosine similarity


class DTIDataset(Dataset):
    def __init__(self, d, p2, labels):
        self.d = d
        self.p = p2
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.d[idx], self.p[idx], self.y[idx]



def train_projection(d_emb, p2_emb, labels, epochs=5):

    
    d_dim = d_emb.shape[1]
    p_dim = p2_emb.shape[1]

    print(f"Drug dim: {d_dim}, Protein dim: {p_dim}")

    model = ProjectionModel(d_dim, p_dim).to(device)

    dataset = DTIDataset(d_emb, p2_emb, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0

        for d, p, y in loader:
            d = d.to(device)
            p = p.to(device)
            y = y.float().to(device)

            logits = model(d, p)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss:.4f}")

    return model


def compute_embeddings(df):
    print("Computing embeddings...")

    p1 = get_batch_embeddings(df["protein1_seq"].tolist(), esm_tokenizer, esm_model)
    p2 = get_batch_embeddings(df["protein2_seq"].tolist(), esm_tokenizer, esm_model)
    d  = get_batch_embeddings(df["drug_smiles"].tolist(), chem_tokenizer, chem_model)

    # Debug shapes (important)
    print("p1 shape:", p1.shape)
    print("p2 shape:", p2.shape)
    print("d shape:", d.shape)

    # Normalize
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    d  = F.normalize(d, dim=1)

    return p1, p2, d



def compute_features(df, p1, p2, d, proj_model):

    proj_model.eval()

    with torch.no_grad():
        sim_d_p2 = torch.sigmoid(proj_model(d.to(device), p2.to(device))).cpu()
        sim_d_p1 = torch.sigmoid(proj_model(d.to(device), p1.to(device))).cpu()

    df["emb_sim_p1_p2"] = (p1 * p2).sum(dim=1).numpy()
    df["emb_sim_d_p1"] = sim_d_p1.numpy()
    df["emb_sim_d_p2"] = sim_d_p2.numpy()

    df["seq_sim_simple"] = [
        simple_seq_similarity(a, b)
        for a, b in zip(df["protein1_seq"], df["protein2_seq"])
    ]

    return df



def normalize(train_df, test_df):

    stats = {}
    cols = ["emb_sim_p1_p2", "emb_sim_d_p1", "emb_sim_d_p2"]

    for col in cols:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-8
        stats[col] = (mean, std)

    def apply_norm(df):
        for col in cols:
            mean, std = stats[col]
            df[col] = (df[col] - mean) / std
        return df

    return apply_norm(train_df), apply_norm(test_df)



if __name__ == "__main__":

    print("Loading data...")
    train = pd.read_csv("dataset_train.csv")
    test  = pd.read_csv("ataset_test.csv")

  
    train_p1, train_p2, train_d = compute_embeddings(train)

  
    labels = torch.tensor(train["label"].values)
    print("Training projection model...")
    proj_model = train_projection(train_d, train_p2, labels)


    train = compute_features(train, train_p1, train_p2, train_d, proj_model)

  
    test_p1, test_p2, test_d = compute_embeddings(test)

 
    test = compute_features(test, test_p1, test_p2, test_d, proj_model)


    train, test = normalize(train, test)


    train.to_pickle("train_with_emb.pkl")
    test.to_pickle("test_with_emb.pkl")

    print("Done. Embeddings computed successfully.")