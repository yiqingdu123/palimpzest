import os
import time

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import datasets
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# NOTE: this script is meant to be run from the root of the repository
if __name__ == "__main__":
    # initialize openai client
    openai_client = OpenAI()

    # load reaction terms
    reaction_terms = []
    reaction_terms_path = "testdata/reaction_terms.txt"
    if os.path.exists(reaction_terms_path):
        with open(reaction_terms_path) as f:
            for line in f:
                term = line.strip()
                if term:
                    reaction_terms.append(term)
    else:
        print("testdata/reaction_terms.txt not found; deriving terms from BioDEX dataset...")
        ds = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="train")
        reaction_terms_set = set()
        for entry in ds:
            for term in entry["reactions"].split(","):
                normalized = term.strip().lower().replace("'", "").replace("^", "")
                if normalized:
                    reaction_terms_set.add(normalized)
        reaction_terms = sorted(reaction_terms_set)

        # persist for future runs
        os.makedirs("testdata", exist_ok=True)
        with open(reaction_terms_path, "w") as f:
            for term in reaction_terms:
                f.write(f"{term}\n")

    if len(reaction_terms) == 0:
        raise ValueError("No reaction terms were loaded/generated.")

    # create directory for embeddings
    os.makedirs("testdata/reaction-term-embeddings/", exist_ok=True)

    # generate embeddings in batches of 1000 at a time
    batch_size = 1000
    batch_ranges = [(start_idx, min(start_idx + batch_size, len(reaction_terms))) for start_idx in range(0, len(reaction_terms), batch_size)]
    total_embeds = len(batch_ranges)
    print(f"Generating {total_embeds} embeddings...")
    for start_idx, end_idx in tqdm(batch_ranges, total=total_embeds):
        filename = f"testdata/reaction-term-embeddings/{start_idx}_{end_idx}.npy"
        if not os.path.exists(filename):
            # generate embeddings
            batch = reaction_terms[start_idx:end_idx]
            resp = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings = [item.embedding for item in resp.data]

            # save embeddings to disk
            with open(filename, "wb") as f:
                np.save(f, np.array(embeddings))

            time.sleep(1)
    print("Done generating embeddings.")

    # initialize chroma client
    chroma_client = chromadb.PersistentClient(".chroma-biodex")

    # initialize embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    # create a collection
    collection = chroma_client.get_or_create_collection(
        name="biodex-reaction-terms",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    # insert documents in batches
    total_inserts = len(batch_ranges)
    print(f"Inserting {total_inserts} batches into the collection...")
    for start_idx, end_idx in tqdm(batch_ranges, total=total_inserts):
        embeddings = np.load(f"testdata/reaction-term-embeddings/{start_idx}_{end_idx}.npy")
        collection.add(
            documents=reaction_terms[start_idx:end_idx],
            embeddings=embeddings.tolist(),
            ids=[f"id{idx}" for idx in range(start_idx, end_idx)]
        )
