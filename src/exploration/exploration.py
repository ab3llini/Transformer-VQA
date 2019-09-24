import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# questions = VQALoader().questions
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
# # Stats
# longest_q, shortest_q, mean_q, n_q = 0, 100, 0, 0
# longest_a, shortest_a, mean_a, n_a = 0, 100, 0, 0
#
# counts_q, counts_a = {}, {}
#
# for question in tqdm(questions):
#     tk_q = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question.question))

    # l_q = len(tk_q)
    # if l_q > longest_q:
    #     longest_q = l_q
    # if l_q < shortest_q:
    #     shortest_q = l_q
    # mean_q += l_q
    # n_q += 1

    # Compute number of question of specified size

    # l_q = len(tk_q)
    # if l_q in counts_q:
    #     counts_q[l_q] += 1
    # else:
    #     counts_q[l_q] = 1

    # for a in question.answers:
    #     tk_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(a))
    #
    #     l_a = len(tk_a)
    #     if l_a > longest_a:
    #         longest_a = l_a
    #     if l_a < shortest_a:
    #         shortest_a = l_a
    #     mean_a += l_a
    #     n_a += 1

    # tk_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question.answers[0]))
    #
    # l_a = len(tk_a)
    # if l_a > longest_a:
    #     longest_a = l_a
    # if l_a < shortest_a:
    #     shortest_a = l_a
    # mean_a += l_a
    # n_a += 1

    # for ans in question.answers:
    #     tk_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans))
    #     l_a = len(tk_a)
    #     if l_a in counts_a:
    #         counts_a[l_a] += 1
    #     else:
    #         counts_a[l_a] = 1

# mean_q /= n_q
# `mean_a /= n_a

# print('Total questions:', n_q, '- longest:', longest_q, '- shortest', shortest_q, '- mean', mean_q)
# print('First answers number:', n_a, '- longest:', longest_a, '- shortest', shortest_a, '- mean', mean_a)


# Output :
# Total questions: 443757 - longest: 25 - shortest 3 - mean 7.413048132198478
# Total answers: 4437570 - longest: 34 - shortest 1 - mean 1.2981573248421996
# First answers number: 443757 - longest: 24 - shortest 1 - mean 1.2980752979671306

# print('question counts', counts_q)
# print('answer counts', counts_a)

# {8: 76310, 7: 87987, 6: 106342, 5: 54021, 13: 4327, 4: 9941, 10: 26516, 9: 50368, 14: 2376, 12: 8082, 16: 794,
# 11: 14148, 17: 448, 20: 132, 15: 1364, 18: 283, 21: 75, 19: 173, 24: 7, 3: 14, 22: 32, 23: 16, 25: 1}
#
# {1: 3645101,
# 2: 456056, 3: 236109, 4: 54081, 5: 25669, 7: 5059, 6: 9496, 13: 212, 10: 726, 11: 501, 8: 2367, 9: 1540, 14: 122,
# 15: 77, 12: 247, 17: 49, 21: 15, 18: 33, 26: 3, 20: 12, 23: 5, 34: 1, 28: 2, 33: 1, 16: 43, 19: 21, 30: 2, 24: 7,
# 22: 8, 25: 4, 27: 1}

counts_q = {8: 76310, 7: 87987, 6: 106342, 5: 54021, 13: 4327, 4: 9941, 10: 26516, 9: 50368, 14: 2376, 12: 8082,
            16: 794, 11: 14148, 17: 448, 20: 132, 15: 1364, 18: 283, 21: 75, 19: 173, 24: 7, 3: 14, 22: 32, 23: 16,
            25: 1}

counts_a = {1: 3645101, 2: 456056, 3: 236109, 4: 54081, 5: 25669, 7: 5059, 6: 9496, 13: 212, 10: 726, 11: 501, 8: 2367,
            9: 1540, 14: 122, 15: 77, 12: 247, 17: 49, 21: 15, 18: 33, 26: 3, 20: 12, 23: 5, 34: 1, 28: 2, 33: 1,
            16: 43, 19: 21, 30: 2, 24: 7, 22: 8, 25: 4, 27: 1}


q = pd.DataFrame({'Length': sorted(counts_q.keys()), 'Count': [counts_q[key] for key in sorted(counts_q.keys())]})
a = pd.DataFrame({'Length': sorted(counts_a.keys()), 'Count': [counts_a[key] for key in sorted(counts_a.keys())]})
plt.figure(figsize=(12,8))

ax = sns.barplot(x="Length", y= "Count",data=q)
ax.set(xlabel="Question length", ylabel='Count')

plt.tight_layout()
plt.show()

ay = sns.barplot(x="Length", y= "Count",data=a)
ay.set(xlabel="Answer length", ylabel='Count')

plt.tight_layout()
plt.show()

acc = 0
tot = 443757

for length, counts in counts_q.items():
   if length in [5, 6, 7, 8, 9]:
    acc += counts

print(acc )