# ADL HW1: User State Prediction

Given the dialogue history (`context`) and the customer's current message (`utterance`),
predict which of the 10 intent labels apply. 

## Files

| File | Description |
|---|---|
| `train.jsonl` | 3,000 labeled training examples |
| `context.json` | dialogue history for each `public_test` example, keyed by `id` |
| `test.json` | the 500 `public_test` examples to predict (no labels) |
| `sample_prediction.csv` | example of the exact output format your `run.sh` must produce |
| `public_test_gold.csv` | ground-truth labels for `test.json`, same schema as `prediction.csv`|

## Data format

**`train.jsonl`** — one JSON object per line:

```json
{
  "id": "train_99999",
  "language": "zh",
  "context": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "utterance": "...",
  "labels": ["Ask_Service", "Ask_Spec"]
}
```

**`context.json` + `test.json`** — the test-time input is split into two files that your `run.sh` must join by `id`:

```json
// context.json 
{
  "public_test_90001": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "public_test_90002": []
}
```

```json
// test.json 
[
  {"id": "public_test_90001", "language": "zh", "utterance": "..."},
  {"id": "public_test_90002", "language": "en", "utterance": "..."}
]
```

For each entry in `test.json`, look up `context.json[id]` to get its dialogue history (an empty list `[]` if there is none). 

### Dataset Notes
- Class distribution: The dataset is naturally imbalanced, with per-label frequencies ranging from roughly 5-7% for the rarest labels to about 30-35% for the most common ones.
- Context: Some samples include dialogue context, while roughly 10% have an empty `context`, so the pipeline should handle both cases.

## Labels

10 intent labels: `Ask_Price`, `Ask_Service`, `Ask_Spec`, `Compare_Competitor`, `Confirm_Order`, `Doubt`, `Need_More_Evidence`, `Need_Time_To_Think`, `State_Need`, `Too_Expensive`.

**Annotation principles**

- Label only what is observable.
- Multi-label: a single utterance may carry several intents.
- Use `context` to disambiguate. 


### Label definitions
| Label | Definition |
|---|---|
| `Ask_Price` |  Asks about price, fees, or discounts. |
| `Ask_Service` |  Asks about delivery, warranty, returns, installation, or support. |
| `Ask_Spec` | Asks about features, specs, terms, or whether the product fits them.. |
| `Compare_Competitor` |  Compares with another brand, shop, or platform. |
| `Confirm_Order` | Committing to buy. |
| `Doubt` |  Distrusts the product's claims, the brand, or the seller. |
| `Need_More_Evidence` |  Asks for proof, such as reviews, reports, or certifications. |
| `Need_Time_To_Think` |  Puts off the decision or needs to consult someone. |
| `State_Need` | States a need or asks for a recommendation, without asking about a specific attribute. |
| `Too_Expensive` | Says this product is too pricey or not worth it. |


### Key distinctions 

| Message | Label | Why |
|---|---|---|
| "Is it really waterproof?" | `Doubt` | Questions whether the claim is true. |
| "What's the waterproof rating?" | `Ask_Spec` | Asks for a factual product attribute. |
| "Do you have a waterproof test report?" | `Need_More_Evidence` | Requests verifiable evidence to support the claim. |
| "Will the packaging get damaged? How many days will it take to arrive?" | `Ask_Service` | Raises a fulfilment concern, not a doubt about the product claim. |
| "I'm looking at these knives. What material are they made of?" | `Ask_Spec` | A concrete attribute is asked, so `State_Need` is not also assigned. |

**中文範例**

| Message | Label | Why |
|---|---|---|
| 「那我沒什麼顧慮了，直接去你們店下單吧。」 | `Confirm_Order` | Commits to buying. |
| 「報告我看過了，沒問題。要怎麼下單？」」 | `Compare_Competitor` | Compares with another shop, not a budget complaint. |
| 「好，那我下個月再來看看。」 | `Need_Time_To_Think` | Puts off the decision until next month. |


## Evaluation & Submission

### Metric
- Macro-F1/Micro-F1 

### Submit

We will grade by running:
```bash
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

- `${1}`: path to a `context.json` (same shape as the one given to you)
- `${2}`: path to a `test.json` (same shape as the one given to you)
- `${3}`: path to write your output `prediction.csv`

`run.sh` must work on the held-out private test set and finish within 2 hours. 

### Output format — `prediction.csv`
One row per example in `test.json`, in the same order:

```
id,Ask_Price,Ask_Service,Ask_Spec,Compare_Competitor,Confirm_Order,Doubt,Need_More_Evidence,Need_Time_To_Think,State_Need,Too_Expensive
public_test_90001,0,0,1,0,0,0,0,0,0,0
```

- Use the exact header and column order above.
- Each label is `1` if predicted, otherwise `0`.
- All-zero predictions are allowed.
- Every `id` must appear exactly once and in the original order. 

See `sample_prediction.csv` for an example.
