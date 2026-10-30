# Retrieval evaluation

21 questions over Bid1 and Bid2 (`eval/questions.json`), each with gold (file, page) passages. Bid filter applied where the question names a bid. MRR is computed over the top 10.

| Config | R@1 | R@3 | R@5 | MRR | ms/query |
|---|---|---|---|---|---|
| vector only | 0.62 | 0.86 | 0.86 | 0.733 | 37 |
| BM25 only | 0.71 | 0.90 | 0.90 | 0.794 | 24 |
| hybrid (RRF) | 0.67 | 0.90 | 0.90 | 0.770 | 23 |
| hybrid + rerank, no expansion | 0.76 | 0.86 | 0.95 | 0.838 | 2112 |
| hybrid + rerank | 0.81 | 0.86 | 0.90 | 0.850 | 2163 |

## Rank of first gold hit per question

`-` = not in top 10.

| Question | vector only | BM25 only | hybrid (RRF) | hybrid + rerank, no expansion | hybrid + rerank |
|---|---|---|---|---|---|
| When are proposals due after all changes? | 3 | 3 | 3 | 1 | 1 |
| When is the pre-proposal meeting? | 1 | 1 | 1 | 1 | 1 |
| Is a bid bond required with the submission? | 1 | 3 | 1 | 2 | 1 |
| Do vendors have to file Form 1295? | 1 | 1 | 1 | 1 | 1 |
| Must devices be etched or asset tagged before delivery? | 1 | 1 | 1 | 1 | 1 |
| Can other public entities purchase from this contract? | 2 | 1 | 1 | 1 | 1 |
| Who is the buyer and how do I contact them? | 8 | - | - | 4 | - |
| What kinds of devices are being procured? | 9 | - | - | 5 | 5 |
| What insurance coverage must the vendor carry? | 1 | 1 | 1 | 1 | 1 |
| Are there HUB subcontracting requirements? | 2 | 1 | 2 | 2 | 2 |
| How should devices be imaged and enrolled in management? | 3 | 1 | 1 | 1 | 1 |
| How soon after award must the laptops be delivered? | 1 | 1 | 1 | 1 | 1 |
| What processor and RAM does the laptop configuration have? | 1 | 1 | 1 | 1 | 1 |
| How many docking stations are requested? | 1 | 1 | 1 | 1 | 1 |
| Who is the procurement officer and what is their email? | 2 | 2 | 3 | 7 | 7 |
| What is the proposal due date and time? | 1 | 1 | 1 | 1 | 1 |
| How will the proposals be evaluated? | 1 | 1 | 1 | 1 | 1 |
| Do the laptops need to support Microsoft Copilot? | 1 | 1 | 1 | 1 | 1 |
| What is the eMMA solicitation number? | 1 | 1 | 1 | 1 | 1 |
| What does the contract affidavit require the bidder to certify? | 1 | 2 | 2 | 1 | 1 |
| Which bid includes part 210-BLYZ? | - | 1 | 2 | 1 | 1 |
