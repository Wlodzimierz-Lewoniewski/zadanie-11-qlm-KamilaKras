import numpy as np
import sys

def delete_symbols(sentences_list_initial):
    symbols = [",", ".", ":", ";", "!", "?"]
    sentences_list = []
    for sentence in sentences_list_initial:
        for i in sentence:
            if i in symbols:
                sentence = sentence.replace(i, "")
        sentences_list.append(sentence)
    return sentences_list

def create_terms_list(terms_list_nested):
    terms_list = []
    for row in terms_list_nested:
        terms_list.extend(row)
    return list(set(terms_list))

def split_to_terms(sentences_list_clean):
    terms = []
    for sentence in sentences_list_clean:
        term = sentence.split()
        terms.append(term)
    return terms

def process_input():
    input_data = sys.stdin.read().strip().lower().split('\n')
    num_sentences = int(input_data[0])
    documents = delete_symbols(input_data[1:-1])
    query = input_data[-1].split()
    query_terms = split_to_terms(query)

    return num_sentences, documents, query, query_terms

def count_Ld(documents_list):
    Ld_list = []
    for document in documents_list:
        Ld = len(document.split())
        Ld_list.append(Ld)

    L_total = sum(Ld_list)
    return Ld_list, L_total

def count_matrix_terms(query_terms, documents_list):
    matrix_terms = np.zeros((len(query_terms), len(documents_list)), dtype=int)
    rows_sums = []

    for i in range(len(query_terms)):
        for j in range(len(documents_list)):
            for word in documents_list[j].split():
                if query_terms[i][0] in word:
                    matrix_terms[i][j] += 1
                else:
                    matrix_terms[i][j] += 0

    for i in range(len(query_terms)):
        rows_sums.append(sum(matrix_terms[i]))
    rows_probabilities = []
    for i in range(len(rows_sums)):
        rows_probabilities.append(rows_sums[i]/L_total)
    return matrix_terms, rows_probabilities

def count_matrix_terms_2(matrix_terms, Ld_list, L_total, documents_list, query_terms):
    matrix_terms_2 = np.zeros((len(query_terms), len(documents_list)), dtype=float)

    for i in range(len(query_terms)):
        for j in range(len(documents_list)):
            matrix_terms_2[i][j] = matrix_terms[i][j] / Ld_list[j]
    return matrix_terms_2

def count_matrix_probabilities(rows_probabilities,query_terms, documents_list, matrix_terms_2, Ld_list, L_total):
    matrix_probabilities = np.zeros((len(query_terms), len(documents_list)), dtype=float)

    for i in range(len(query_terms)):
        for j in range(len(documents_list)):
            matrix_probabilities[i][j] = (1/2 * matrix_terms_2[i][j]) + ((1-1/2) * rows_probabilities[i])
    return matrix_probabilities

def count_probabilities(matrix_probabilities, documents_list, query_terms):
    probabilities = np.zeros((1, len(documents_list)), dtype=float)
    for i in range(len(documents_list)):
        p = 1
        for j in range(len(query_terms)):
            p *= matrix_probabilities[j][i]
        probabilities[0][i] = p

    return probabilities

def rank(probabilities):
    sorted_indices = np.argsort(probabilities[0])[::-1]
    sorted_indices = sorted_indices.tolist()
    return sorted_indices

if __name__ == '__main__':
    num_sentences, documents, query, query_terms = process_input()
    Ld_list, L_total = count_Ld(documents)
    matrix_terms, rows_probabilities = count_matrix_terms(query_terms, documents)
    matrix_terms_2 = count_matrix_terms_2(matrix_terms, Ld_list, L_total, documents, query_terms)
    matrix_probabilities = count_matrix_probabilities(rows_probabilities, query_terms, documents, matrix_terms_2, Ld_list, L_total)
    probabilities = count_probabilities(matrix_probabilities, documents, query_terms)
    sorted_indices = rank(probabilities)
    print(sorted_indices)