import json
from datasets import load_dataset
import time
concat_dataset = []

if __name__ == "__main__":
    start_time = time.time()
    print('[+] Starting...')
    
    # question_answering
    print('[+] Generating dataset google/boolq for question_answering...')
    boolq = load_dataset("google/boolq") # 3.68k rows
    for item in boolq["train"]:
        concat_dataset.append(
            {
                "passage": item["passage"],
                "question": item["question"],
                "scenario": "question_answering",
            }
        )

    print(f"[+] Question_answering done with {len(boolq['train'])} records...")

    # summarization
    print('[+] Generating dataset cnn_dailymail for summarization...')
    cnn_dailymail = load_dataset("cnn_dailymail", "1.0.0") # 312k rows
    for item in cnn_dailymail["train"]:
        concat_dataset.append(
            {
                "article": item["article"],
                "scenario": "summarization",
            }
        )

    print(f"[+] Summarization done with {len(cnn_dailymail['train'])} records...")

    # text_classification
    print('[+] Generating dataset ought/raft for text_classification...')
    raft = load_dataset("ought/raft", "ade_corpus_v2") # 5.05k rows
    for item in raft["train"]:
        concat_dataset.append(
            {
                "sentence": item["Sentence"],
                "scenario": "text_classification",
            }
        )

    print(f"[+] Text classification done with {len(raft['train'])} records...")

    # language
    print('[+] Generating dataset nyu-mll/blimp for language...')
    blimp1 = load_dataset("nyu-mll/blimp", "adjunct_island") # each subset is 1k rows
    for item in blimp1["train"]:
        concat_dataset.append(
            {
                "sentence_good": item["sentence_good"],
                "sentence_bad": item["sentence_bad"],
                "scenario": "language",
            }
        )

    # language
    blimp2 = load_dataset("nyu-mll/blimp", "causative") # each subset is 1k rows
    for item in blimp2["train"]:
        concat_dataset.append(
            {
                "sentence_good": item["sentence_good"],
                "sentence_bad": item["sentence_bad"],
                "scenario": "language",
            }
        )

    # language
    blimp3 = load_dataset("nyu-mll/blimp", "drop_argument") # each subset is 1k rows
    for item in blimp3["train"]:
        concat_dataset.append(
            {
                "sentence_good": item["sentence_good"],
                "sentence_bad": item["sentence_bad"],
                "scenario": "language",
            }
        )

    print(f"[+] Language done with {len(raft['train']) + len(blimp1['train']) + len(blimp3['train'])} records...")

    # knowledge
    wikifact = load_dataset("lighteval/wikifact", "capital") # 950 rows
    for item in wikifact["train"]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "knowledge",
            }
        )

    # knowledge
    print('[+] Generating dataset truthful_qa for knowledge...')
    truthful_qa1 = load_dataset("truthful_qa", "generation") # 817 rows
    for item in truthful_qa1["validation"]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "knowledge",
            }
        )

    # knowledge
    truthful_qa2 = load_dataset("truthful_qa", "multiple_choice") # 817 rows
    for item in truthful_qa2["validation"]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "knowledge",
            }
        )

    print(f"[+] Knowledge done with {len(wikifact['train']) + len(truthful_qa1['validation']) + len(truthful_qa2['validation'])} records...")

    # harms
    print('[+] Generating dataset lighteval/disinformation for harms...')
    disinformation1 = load_dataset("lighteval/disinformation", "reiteration_climate") # 49 rows
    for item in disinformation1["validation"]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "harms",
            }
        )

    # harms
    disinformation2 = load_dataset("lighteval/disinformation", "reiteration_covid") # 38 rows
    for item in disinformation2["validation"]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "harms",
            }
        )

    print(f"[+] Harms done with {len(disinformation1['validation']) + len(disinformation2['validation'])} records...")

    # sentiment_analysis
    print('[+] Generating dataset stanfordnlp/imdb for sentiment_analysis...')
    imdb = load_dataset("stanfordnlp/imdb") # 25k rows
    for item in imdb["train"]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "sentiment_analysis",
            }
        )

    print(f"[+] Sentiment analysis done with {len(imdb['train'])} records...")

    # toxicity_detection
    print('[+] Generating dataset google/civil_comments for toxicity_detection...')
    civil_comments = load_dataset("google/civil_comments")
    for item in civil_comments["train"]:
        concat_dataset.append(
            {
                "text": item["text"],
                "scenario": "toxicity_detection"
            }
        )

    print(f"[+] Toxicity detection done with {len(civil_comments['train'])} records...")

    # reasoning
    print('[+] Generating dataset lighteval/synthetic_reasoning_natural for reasoning...')
    synthetic_reasoning_natural = load_dataset("lighteval/synthetic_reasoning_natural", "easy")
    for item in synthetic_reasoning_natural["train"]:
        concat_dataset.append(
            {
                "question": item["question"],
                "scenario": "reasoning"
            }
        )

    print(f"[+] Reasoning done with {len(synthetic_reasoning_natural['train'])} records...")

    # information_retrieval
    print('[+] Generating dataset msmarco for information_retrieval...')
    num_of_recs = 0
    mscoco = load_dataset("ms_marco", 'v1.1')
    
    for item in mscoco["validation"]:
        try:
            index_of_1 = item['passages']['is_selected'].index(1)
        except ValueError:
            continue
        num_of_recs += 1
        concat_dataset.append(
            {
                "passage": item['passages']["passage_text"][index_of_1],
                "query": item['query'],
                "scenario": "information_retrieval"
            }
        )

    print(f"[+] Information retrieval done with {num_of_recs} records...")
    with open("data/concat_dataset.json", 'w',encoding="utf-8") as f:
        json.dump(concat_dataset, f, indent=4)

    print('DONE!!!')
    print("--- %s seconds ---" % (time.time() - start_time))
