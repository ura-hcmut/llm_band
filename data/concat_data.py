import json
from datasets import load_dataset
import time
import argparse

concat_dataset = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for threshold')

    parser.add_argument('--n', type=int, required=True, help='n')
    args = parser.parse_args()
    n = int(args.n)
    
    data_len = 0
    start_time = time.time()
    print('[+] Starting...')
    
    # question_answering
    print('[+] Generating dataset google/boolq for question_answering...')
    boolq = load_dataset("google/boolq") # 3.68k rows
    for item in list(boolq["train"])[:n]:
        concat_dataset.append(
            {
                "passage": item["passage"],
                "question": item["question"],
                "scenario": "question_answering",
            }
        )

    length = len(list(boolq['train'])[:n])
    print(f"[+] Question_answering done with {length} records...")
    data_len += len(list(boolq['train'])[:n])

    # summarization
    print('[+] Generating dataset cnn_dailymail for summarization...')
    cnn_dailymail = load_dataset("cnn_dailymail", "1.0.0") # 312k rows
    for item in list(cnn_dailymail["train"])[:n]:
        concat_dataset.append(
            {
                "article": item["article"],
                "scenario": "summarization",
            }
        )

    length = len(list(cnn_dailymail['train'])[:n])
    print(f"[+] Summarization done with {length} records...")
    data_len += len(list(cnn_dailymail['train'])[:n])

    # text_classification
    print('[+] Generating dataset ought/raft for text_classification...')
    raft = load_dataset("ought/raft", "ade_corpus_v2") # 5.05k rows
    for item in list(raft["train"])[:n]:
        concat_dataset.append(
            {
                "sentence": item["Sentence"],
                "scenario": "text_classification",
            }
        )

    length = len(list(raft['train'])[:n])
    print(f"[+] Text classification done with {length} records...")
    data_len += len(list(raft['train'])[:n])

    # language
    print('[+] Generating dataset nyu-mll/blimp for language...')
    blimp1 = load_dataset("nyu-mll/blimp", "adjunct_island") # each subset is 1k rows
    for item in list(blimp1["train"])[:n]:
        concat_dataset.append(
            {
                "sentence_good": item["sentence_good"],
                "sentence_bad": item["sentence_bad"],
                "scenario": "language",
            }
        )

    length = len(list(blimp1['train'])[:n])
    print(f"[+] Language done with {length} records...")
    data_len += len(list(blimp1["train"])[:n])

    # knowledge
    print('[+] Generating dataset lighteval/wikifact for knowledge...')
    wikifact = load_dataset("lighteval/wikifact", "capital") # 950 rows
    for item in list(wikifact["train"])[:n]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "knowledge",
            }
        )

    length = len(list(wikifact['train'])[:n])
    print(f"[+] Knowledge done with {length} records...")
    data_len += len(list(wikifact["train"])[:n])

    # harms
    print('[+] Generating dataset lighteval/disinformation for harms...')
    disinformation1 = load_dataset("lighteval/disinformation", "reiteration_climate") # 49 rows
    for item in list(disinformation1["validation"])[:n]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "harms",
            }
        )

    length = len(list(disinformation1['validation'])[:n])
    print(f"[+] Harms done with {length} records...")
    data_len += len(list(disinformation1['validation'])[:n])

    # sentiment_analysis
    print('[+] Generating dataset stanfordnlp/imdb for sentiment_analysis...')
    imdb = load_dataset("stanfordnlp/imdb") # 25k rows
    for item in list(imdb["train"])[:n]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "sentiment_analysis",
            }
        )

    length = len(list(imdb['train'])[:n])
    print(f"[+] Sentiment analysis done with {length} records...")
    data_len += len(list(imdb['train'])[:n])

    # toxicity_detection
    print('[+] Generating dataset google/civil_comments for toxicity_detection...')
    civil_comments = load_dataset("google/civil_comments")
    for item in list(civil_comments["validation"])[:n]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "toxicity_detection"
            }
        )

    length = len(list(civil_comments['validation'])[:n])
    print(f"[+] Toxicity detection done with {length} records...")
    data_len += len(list(civil_comments['validation'])[:n])

    # reasoning
    print('[+] Generating dataset lighteval/synthetic_reasoning_natural for reasoning...')
    synthetic_reasoning_natural = load_dataset("lighteval/synthetic_reasoning_natural", "easy")
    for item in list(synthetic_reasoning_natural["train"])[:n]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "reasoning"
            }
        )

    length = len(list(synthetic_reasoning_natural['train'])[:n])
    print(f"[+] Reasoning done with {length} records...")
    data_len += len(list(synthetic_reasoning_natural['train'])[:n])

    # information_retrieval
    print('[+] Generating dataset msmarco for information_retrieval...')
    num_of_recs = 0
    mscoco = load_dataset("ms_marco", 'v1.1')
    
    for item in list(mscoco["validation"]):
        if num_of_recs == n:
            break
        try:
            index_of_1 = item['passages']['is_selected'].index(1)
        except ValueError:
            continue
        num_of_recs += 1
        concat_dataset.append(
            {
                "passage": item['passages']["passage_text"][index_of_1],
                "question": item['query'],
                "scenario": "information_retrieval"
            }
        )

    print(f"[+] Information retrieval done with {num_of_recs} records...")
    data_len += num_of_recs
    with open("data/concat_dataset.json", 'w',encoding="utf-8") as f:
        json.dump(concat_dataset, f, indent=4)

    print('DONE!!!')
    print(f'[+] DATA LENGTH: {data_len}')
    print("--- %s seconds ---" % (time.time() - start_time))
